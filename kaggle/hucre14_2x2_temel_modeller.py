# Hücre 14: 2×2 deneyi — 1. adım: düzeltilmiş veri seti, aynı modeller (4k ve 8k)
#   (1) Düzeltilmiş veri, aynı model    (3) İki katı (düzeltilmiş) veri, aynı model
# Düzeltme (rapor Bölüm 3, ön işleme): tüm görseller aynı boyut ve aynı format —
#   özgün çözünürlükte 224×224 merkezden kırpma (kısa kenarı 224'ten küçükse, yalnız BigGAN 128 px, önce bicubic ile 224'e büyütülür)
#   ve JPEG kalite 96 ile yeniden kaydetme. Eğitimde ve testte artık yeniden boyutlandırma yok (eğitimde yalnız yatay çevirme).
# Hücre 12 ve 13 aynı oturumda çalışmış olmalı (train_items, SETS_B, test_sets, metrics, ImgDS, SEEDS, REN, UNSEEN5).
import hashlib
from concurrent.futures import ThreadPoolExecutor
from sklearn.ensemble import RandomForestClassifier

clip = None; gc.collect(); torch.cuda.empty_cache()   # Hücre 0'daki CLIP artık gerekmiyor; GPU'yu boşalt

FIXDIR = "/kaggle/temp/fixed"; os.makedirs(FIXDIR, exist_ok=True)
def fix_one(p):
    q = os.path.join(FIXDIR, hashlib.md5(p.encode()).hexdigest() + ".jpg")
    if not os.path.exists(q):
        im = Image.open(p).convert("RGB"); w, h = im.size
        if min(w, h) < 224:
            s = 224 / min(w, h)
            im = im.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BICUBIC); w, h = im.size
        l, t = (w - 224) // 2, (h - 224) // 2
        im.crop((l, t, l + 224, t + 224)).save(q, "JPEG", quality=96)
    return q

all_paths = sorted({p for it in [train_items, *SETS_B.values(), *test_sets.values()] for p, _ in it})
t0 = time.time()
with ThreadPoolExecutor(8) as ex:
    FIX = dict(zip(all_paths, ex.map(fix_one, all_paths)))
print(f"Düzeltilmiş görsel: {len(FIX)} ({time.time()-t0:.0f} sn)")
fx = lambda items: [(FIX[p], y) for p, y in items]
TRAIN_FIX = {"Fixed-4k (SD1.5)": fx(train_items), "Fixed-8k (SD1.5+Wukong)": fx(SETS_B["B-8k (SD1.5+Wukong)"])}
TEST_FIX = {g: fx(v) for g, v in test_sets.items()}

# --- Kestirme testi: düzeltmeden önce ve sonra, yalnız dosya özellikleriyle Random Forest (%50 = kestirme yok) ---
def meta(p):
    im = Image.open(p); w, h = im.size
    q = getattr(im, "quantization", None)
    return [w, h, w / h, os.path.getsize(p) / (w * h), int(im.format == "PNG"), float(np.mean(q[0])) if q else 0.0]
meta_X = lambda items: np.array([meta(p) for p, _ in items])
rf_rows = []
for name, tr, tests in [("Unbiased (before)", train_items, test_sets), ("Fixed (after)", TRAIN_FIX["Fixed-4k (SD1.5)"], TEST_FIX)]:
    rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(meta_X(tr), [y for _, y in tr])
    for g, items in tests.items():
        rf_rows.append({"Data": name, "Generator": g,
                        "Accuracy": metrics([y for _, y in items], rf.predict_proba(meta_X(items))[:, 1])["Doğruluk"] * 100})
RF = pd.DataFrame(rf_rows); RF.to_csv("/kaggle/working/kestirme_duzeltme.csv", index=False)
print("\n=== Metadata-only Random Forest, Accuracy (%) — %50 = kestirme yok ===")
print(RF.pivot(index="Generator", columns="Data", values="Accuracy").loc[list(GEN)].round(1).to_string())

# --- Eğitim ve değerlendirme (düzeltilmiş görseller 224×224 olduğu için yeniden boyutlandırma yok) ---
tf_tr_fix = T_.Compose([T_.RandomHorizontalFlip(), T_.ToTensor(), T_.Normalize(MEAN, STD)])
tf_te_fix = T_.Compose([T_.ToTensor(), T_.Normalize(MEAN, STD)])

@torch.no_grad()
def predict2(model, items, bs=64):
    model.eval(); out = []
    for x, _ in DataLoader(ImgDS(items, tf_te_fix), batch_size=bs, num_workers=4):
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(torch.sigmoid(model(x.to(DEVICE)).float().squeeze(1)).cpu())
    return torch.cat(out).numpy()

def make_model(arch):
    for a in ([arch] if isinstance(arch, str) else arch):  # liste verilirse ilk bulunan ağırlık kullanılır
        try:
            m = timm.create_model(a, pretrained=True, num_classes=1); print(f"  model: {a}"); return m
        except Exception as e:
            print(f"  {a} yüklenemedi: {type(e).__name__}")
    raise RuntimeError(f"Hiçbir ağırlık yüklenemedi: {arch}")

def finetune2(arch, items, seed, epochs=3, lr=1e-4, bs=32, accum=1):
    # bs × accum = 32 etkin batch; büyük modellerde bellek için bs=16, accum=2
    torch.manual_seed(seed)
    items = items[:]; random.Random(seed).shuffle(items)
    n_val = len(items) // 10
    val_items, tr_items = items[:n_val], items[n_val:]
    model = make_model(arch).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda")
    dl = DataLoader(ImgDS(tr_items, tf_tr_fix), batch_size=bs, shuffle=True, num_workers=4, drop_last=True)
    lossf = torch.nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        model.train(); t0 = time.time(); tot = 0; opt.zero_grad()
        for i, (x, y) in enumerate(dl):
            x, y = x.to(DEVICE), y.float().to(DEVICE)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = lossf(model(x).squeeze(1), y)
            scaler.scale(loss / accum).backward()
            if (i + 1) % accum == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad()
            tot += loss.item() * len(y)
        sched.step()
        va = metrics([y for _, y in val_items], predict2(model, val_items))["Doğruluk"]
        print(f"  epoch {ep+1}/{epochs}  kayıp {tot/len(tr_items):.4f}  doğrulama {va:.3f}  ({time.time()-t0:.0f} sn)")
    return model

@torch.no_grad()
def clip_feats2(cm, cp, items, bs=64):
    tf = lambda im: cp(images=im, return_tensors="pt")["pixel_values"][0]
    feats = []
    for x, _ in DataLoader(ImgDS(items, tf), batch_size=bs, num_workers=4):
        f = cm.visual_projection(cm.vision_model(pixel_values=x.to(DEVICE, torch.float16)).pooler_output)
        feats.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu())
    return torch.cat(feats).numpy()

def table2(model_name, prob_fn):
    rows = [{"Model": model_name, "Üretici": g, **metrics([y for _, y in it], prob_fn(g, it))} for g, it in TEST_FIX.items()]
    return pd.DataFrame(rows).rename(columns=REN)

GRID, GRID_RES = [], []   # tüm 2×2 koşularının sonuçları ve kaynak ölçümleri (Hücre 15 de buraya ekler)

def run_grid(size, models):
    # models = {"CNN": (ad, arch, bs, accum), "ViT": (...), "CLIP": (ad, hf_adı)}
    t_all = time.time()
    clip_name, clip_hf = models["CLIP"]
    cm = CLIPModel.from_pretrained(clip_hf, torch_dtype=torch.float16).to(DEVICE).eval()
    cp = CLIPImageProcessor.from_pretrained(clip_hf)
    paths = sorted({p for it in [*TRAIN_FIX.values(), *TEST_FIX.values()] for p, _ in it})
    torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    F = dict(zip(paths, clip_feats2(cm, cp, [(p, 0) for p in paths])))
    GRID_RES.append({"Size": size, "Family": "CLIP", "Model": clip_name, "Data": "öznitelik çıkarma", "Seed": "-",
                     "Süre (sn)": time.time() - t0, "GPU tepe (GB)": torch.cuda.max_memory_allocated() / 1e9,
                     "RAM süreç (GB)": psutil.Process().memory_info().rss / 1e9, "RAM sistem (GB)": psutil.virtual_memory().used / 1e9})
    print(f"{clip_name} öznitelikleri: {len(paths)} görsel, {time.time()-t0:.0f} sn, GPU tepe {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    del cm; gc.collect(); torch.cuda.empty_cache()
    for data, items_n in TRAIN_FIX.items():
        for s in SEEDS:
            it = items_n[:]; random.Random(s).shuffle(it); tr = it[len(it) // 10:]
            lr_ = LogisticRegression(max_iter=2000, C=1.0, random_state=s).fit(np.stack([F[p] for p, _ in tr]), [y for _, y in tr])
            t = table2(clip_name + " + LR", lambda g, items: lr_.predict_proba(np.stack([F[p] for p, _ in items]))[:, 1])
            GRID.append(t.assign(Size=size, Family="CLIP", Data=data, Seed=s))
            for fam in ["CNN", "ViT"]:
                name, arch, bs, accum = models[fam]
                print(f"{size} | {data} | seed={s} | {name}")
                base = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
                m = finetune2(arch, items_n, s, bs=bs, accum=accum)
                sure, gpu = time.time() - t0, (torch.cuda.max_memory_allocated() - base) / 1e9
                t = table2(name, lambda g, items, m=m: predict2(m, items))
                GRID.append(t.assign(Size=size, Family=fam, Data=data, Seed=s))
                GRID_RES.append({"Size": size, "Family": fam, "Model": name, "Data": data, "Seed": s, "Süre (sn)": sure,
                                 "GPU tepe (GB)": gpu, "RAM süreç (GB)": psutil.Process().memory_info().rss / 1e9,
                                 "RAM sistem (GB)": psutil.virtual_memory().used / 1e9})
                u5 = t[t["Generator"].isin(UNSEEN5)]["Accuracy"].mean() * 100
                print(f"  -> Mean Accuracy (5 unseen) {u5:.1f}  |  GPU {gpu:.2f} GB  |  {sure:.0f} sn")
                if s == 42:
                    torch.save(m.state_dict(), f"/kaggle/working/{size}_{fam}_{data.split()[0]}_seed42.pt")
                del m; gc.collect(); torch.cuda.empty_cache()
    print(f"{size} toplam süre: {(time.time()-t_all)/60:.1f} dk")

def summarize():
    G = pd.concat(GRID, ignore_index=True); G.to_csv("/kaggle/working/grid2x2.csv", index=False)
    K = pd.DataFrame(GRID_RES); K.to_csv("/kaggle/working/grid2x2_kaynak.csv", index=False)
    COLS = ["Accuracy", "ROC-AUC", "AP", "Fake acc.", "Real acc."]
    U5 = G[G["Generator"].isin(UNSEEN5)].groupby(["Size", "Family", "Data", "Seed"])[COLS].mean().reset_index()
    for col in COLS:
        g = U5.groupby(["Data", "Family", "Size"])[col]
        tab = ((g.mean() * 100).round(1).astype(str) + " ± " + (g.std() * 100).round(1).astype(str)).unstack(["Family", "Size"])
        print(f"\n=== Mean (5 unseen) {col} (%), 3 seeds ===\n" + tab.to_string())
    U6 = G[(G["Data"].str.startswith("Fixed-4k")) & (G["Generator"] != "SD1.5")].groupby(["Size", "Family", "Seed"])["Accuracy"].mean()
    g6 = U6.groupby(["Family", "Size"])
    print("\n=== Fixed-4k: Mean (6 unseen) Accuracy (%), 3 seeds ===\n" +
          ((g6.mean() * 100).round(1).astype(str) + " ± " + (g6.std() * 100).round(1).astype(str)).unstack().to_string())
    print("\n=== Per-generator Accuracy (%), mean of 3 seeds ===")
    print((G.groupby(["Generator", "Data", "Family", "Size"])["Accuracy"].mean().unstack(["Data", "Family", "Size"]) * 100).loc[list(GEN)].round(1).to_string())
    print("\n=== Kaynak kullanımı (seed ortalaması) ===")
    print(K.groupby(["Size", "Family", "Model", "Data"])[["Süre (sn)", "GPU tepe (GB)", "RAM süreç (GB)", "RAM sistem (GB)"]].mean().round(2).to_string())

SMALL = {"CNN": ("ResNet-50", "resnet50.a1_in1k", 32, 1),
         "ViT": ("ViT-B/16", "vit_base_patch16_224.augreg_in21k_ft_in1k", 32, 1),
         "CLIP": ("CLIP ViT-L/14", "openai/clip-vit-large-patch14")}
run_grid("Base", SMALL)
summarize()
