# Hücre 11: Ek deneyler (seed 42)
#  A) D4 mini: orijinal Tiny GenImage (fake PNG / real JPEG) ile eğitim vs Unbiased (JPEG/JPEG) ile eğitim;
#     ikisi de hem Unbiased hem Original test setlerinde sınanır.
#  B) D5 mini: Unbiased ile eğitilen modeller, Unbiased test görsellerine JPEG q75 ve %50 küçültme uygulanınca.
import io

def jpeg75(im):
    b = io.BytesIO(); im.save(b, "JPEG", quality=75); b.seek(0)
    return Image.open(b).convert("RGB")
def half(im):
    return im.resize((max(1, im.width // 2), max(1, im.height // 2)), Image.BICUBIC)
PERTS = {"JPEG q75": jpeg75, "Resize 0.5x": half}

class PDS(Dataset):
    def __init__(self, items, tf, pert=None):
        self.items, self.tf, self.pert = items, tf, pert
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        p, y = self.items[i]
        im = Image.open(p).convert("RGB")
        if self.pert:
            im = self.pert(im)
        return self.tf(im), y

@torch.no_grad()
def ft_probs(model, items, pert=None, bs=64):
    model.eval(); out = []
    for x, _ in DataLoader(PDS(items, tf_test, pert), batch_size=bs, num_workers=4):
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(torch.sigmoid(model(x.to(DEVICE)).float().squeeze(1)).cpu())
    return torch.cat(out).numpy()

@torch.no_grad()
def clip_feats(items, pert=None, bs=64):
    feats = []
    for x, _ in DataLoader(PDS(items, clip_tf, pert), batch_size=bs, num_workers=4):
        f = clip.visual_projection(clip.vision_model(pixel_values=x.to(DEVICE, torch.float16)).pooler_output)
        feats.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu())
    return torch.cat(feats).numpy()

def train_ft(arch, items, seed=42, epochs=3, lr=1e-4, bs=32):
    torch.manual_seed(seed)
    items = items[:]; random.Random(seed).shuffle(items)
    tr = items[len(items) // 10:]
    model = timm.create_model(arch, pretrained=True, num_classes=1).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda"); lossf = torch.nn.BCEWithLogitsLoss()
    dl = DataLoader(PDS(tr, tf_train), batch_size=bs, shuffle=True, num_workers=4, drop_last=True)
    for ep in range(epochs):
        model.train()
        for x, y in dl:
            x, y = x.to(DEVICE), y.float().to(DEVICE)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = lossf(model(x).squeeze(1), y)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step()
    return model

def train_clip(items, seed=42):
    X = clip_feats(items); y = np.array([l for _, l in items])
    idx = list(range(len(items))); random.Random(seed).shuffle(idx); tr = idx[len(idx) // 10:]
    return LogisticRegression(max_iter=2000, C=1.0, random_state=seed).fit(X[tr], y[tr])

def evaluate(model_name, train_src, prob_fn, sets, test_name, pert_name="none"):
    rows = [{"Model": model_name, "Train": train_src, "Test": test_name, "Perturbation": pert_name,
             "Generator": g, **metrics([y for _, y in items], prob_fn(items))} for g, items in sets.items()]
    df = pd.DataFrame(rows)
    mean = df[df["Generator"] != "SD1.5"].mean(numeric_only=True).to_dict()
    mean.update({"Model": model_name, "Train": train_src, "Test": test_name, "Perturbation": pert_name, "Generator": "Mean (6 unseen)"})
    return pd.concat([df, pd.DataFrame([mean])], ignore_index=True)

TRAIN = {"Unbiased (JPEG/JPEG)": train_items, "Original (PNG/JPEG)": train_orig}
TESTS = {"Unbiased test": test_sets, "Original test": test_orig}
extra = []; t0 = time.time()
for src, items in TRAIN.items():
    is_unb = src.startswith("Unbiased")
    lr_m = train_clip(items)
    for tname, sets in TESTS.items():
        extra.append(evaluate("CLIP ViT-L/14 + LR", src, lambda it: lr_m.predict_proba(clip_feats(it))[:, 1], sets, tname))
    if is_unb:
        for pn, pf in PERTS.items():
            extra.append(evaluate("CLIP ViT-L/14 + LR", src, lambda it, pf=pf: lr_m.predict_proba(clip_feats(it, pf))[:, 1], test_sets, "Unbiased test", pn))
    print(f"CLIP | {src} bitti ({(time.time()-t0)/60:.1f} dk)")
    for name, arch in ARCHS:
        m = train_ft(arch, items)
        for tname, sets in TESTS.items():
            extra.append(evaluate(name, src, lambda it, m=m: ft_probs(m, it), sets, tname))
        if is_unb:
            for pn, pf in PERTS.items():
                extra.append(evaluate(name, src, lambda it, m=m, pf=pf: ft_probs(m, it, pf), test_sets, "Unbiased test", pn))
        del m; torch.cuda.empty_cache()
        print(f"{name} | {src} bitti ({(time.time()-t0)/60:.1f} dk)")

E = pd.concat(extra, ignore_index=True).rename(columns=REN)
E.to_csv("/kaggle/working/ek_deneyler.csv", index=False)
M = E[E["Generator"] == "Mean (6 unseen)"]
S = E[E["Generator"] == "SD1.5"]
pd.set_option("display.width", 200)
print("\n=== A) D4 mini — Mean (6 unseen), % ===")
print((M[M["Perturbation"] == "none"].pivot_table(index=["Train", "Test"], columns="Model", values=["Accuracy", "Real acc.", "Fake acc."]) * 100).round(1).to_string())
print("\n=== A) D4 mini — SD1.5 (eğitim üreticisi), Accuracy % ===")
print((S[S["Perturbation"] == "none"].pivot_table(index=["Train", "Test"], columns="Model", values="Accuracy") * 100).round(1).to_string())
print("\n=== B) D5 mini — Unbiased ile eğitim, Unbiased test, Mean (6 unseen), % ===")
R = E[(E["Train"].str.startswith("Unbiased")) & (E["Test"] == "Unbiased test")]
print((R[R["Generator"] == "Mean (6 unseen)"].pivot_table(index="Perturbation", columns="Model", values=["Accuracy", "ROC-AUC"]) * 100).round(1).to_string())
print("\n=== B) D5 mini — SD1.5, Accuracy % ===")
print((R[R["Generator"] == "SD1.5"].pivot_table(index="Perturbation", columns="Model", values="Accuracy") * 100).round(1).to_string())
print(f"Toplam süre: {(time.time()-t0)/60:.1f} dk")
