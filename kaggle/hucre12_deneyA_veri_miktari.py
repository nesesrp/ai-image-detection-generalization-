# Hücre 12: Deney A — veri miktarının etkisi (1.000 / 2.000 / 4.000 eğitim görseli), 3 seed
# Önce Hücre 0'ı (kurulum) çalıştırın. Alt kümeler iç içedir ve dengelidir (yarısı gerçek, yarısı SD 1.5):
# 1.000'lik küme 2.000'liğin, 2.000'lik küme 4.000'liğin içindedir. Test setleri değişmez.
# Ayrıca her koşuda GPU tepe belleği, RAM ve süre ölçülür; N=4000, seed 42 modelleri kaydedilir.
import psutil, gc

def finetune(arch, items, seed, epochs=3, lr=1e-4, bs=32):
    torch.manual_seed(seed)
    items = items[:]; random.Random(seed).shuffle(items)
    n_val = len(items) // 10
    val_items, tr_items = items[:n_val], items[n_val:]
    model = timm.create_model(arch, pretrained=True, num_classes=1).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda")
    dl = DataLoader(ImgDS(tr_items, tf_train), batch_size=bs, shuffle=True, num_workers=4, drop_last=True)
    lossf = torch.nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        model.train(); t0 = time.time(); tot = 0
        for x, y in dl:
            x, y = x.to(DEVICE), y.float().to(DEVICE)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = lossf(model(x).squeeze(1), y)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * len(y)
        sched.step()
        va = metrics([y for _, y in val_items], predict(model, val_items))["Doğruluk"]
        print(f"  {arch[:8]} epoch {ep+1}/{epochs}  kayıp {tot/len(tr_items):.4f}  doğrulama {va:.3f}  ({time.time()-t0:.0f} sn)")
    return model

@torch.no_grad()
def predict(model, items, bs=64):
    model.eval(); out = []
    for x, _ in DataLoader(ImgDS(items, tf_test), batch_size=bs, num_workers=4):
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(torch.sigmoid(model(x.to(DEVICE)).float().squeeze(1)).cpu())
    return torch.cat(out).numpy()

@torch.no_grad()
def clip_features(items, bs=64):
    feats = []
    for x, _ in DataLoader(ImgDS(items, clip_tf), batch_size=bs, num_workers=4):
        f = clip.visual_projection(clip.vision_model(pixel_values=x.to(DEVICE, torch.float16)).pooler_output)
        feats.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu())
    return torch.cat(feats).numpy()

# CLIP öznitelikleri bir kez çıkarılır (kodlayıcı donmuş), alt kümelerde yalnız LR yeniden eğitilir
torch.cuda.reset_peak_memory_stats(); t0 = time.time()
F_train = dict(zip([p for p, _ in train_items], clip_features(train_items)))
F_test = {g: clip_features(items) for g, items in test_sets.items()}
TID = {id(v): g for g, v in test_sets.items()}
clip_gpu, clip_sure = torch.cuda.max_memory_allocated() / 1e9, time.time() - t0
print(f"CLIP öznitelikleri hazır: {clip_sure:.0f} sn, GPU tepe {clip_gpu:.2f} GB")

def clip_run(items, seed):  # finetune() ile aynı %90 eğitim ayrımı
    items = items[:]; random.Random(seed).shuffle(items)
    tr = items[len(items) // 10:]
    lr_ = LogisticRegression(max_iter=2000, C=1.0, random_state=seed).fit(
        np.stack([F_train[p] for p, _ in tr]), [y for _, y in tr])
    return results_table("CLIP + LR", lambda items: lr_.predict_proba(F_test[TID[id(items)]])[:, 1])

rng = random.Random(42)
real_order = real_train[:]; rng.shuffle(real_order)
fake_order = FAKE["SD1.5"]["train"][:]; rng.shuffle(fake_order)
def subset(n):
    k = n // 2
    return [(p, 0) for p in real_order[:k]] + [(p, 1) for p in fake_order[:k]]

SIZES, SEEDS = [1000, 2000, 4000], [42, 43, 44]
rows, kaynak = [], []
t_all = time.time()
for n in SIZES:
    items_n = subset(n)
    for s in SEEDS:
        t = clip_run(items_n, s); t["N"] = n; t["Seed"] = s; rows.append(t)
        for name, arch in ARCHS:
            base = torch.cuda.memory_allocated()  # CLIP zaten bellekte; yalnız bu modelin payı ölçülür
            torch.cuda.reset_peak_memory_stats(); t0 = time.time()
            m = finetune(arch, items_n, s)
            sure, gpu = time.time() - t0, (torch.cuda.max_memory_allocated() - base) / 1e9
            t = results_table(name, lambda items, m=m: predict(m, items)); t["N"] = n; t["Seed"] = s; rows.append(t)
            kaynak.append({"Model": name, "N": n, "Seed": s, "Süre (sn)": sure, "GPU tepe (GB)": gpu,
                           "RAM süreç (GB)": psutil.Process().memory_info().rss / 1e9,
                           "RAM sistem (GB)": psutil.virtual_memory().used / 1e9})
            if n == 4000 and s == 42:
                torch.save(m.state_dict(), f"/kaggle/working/{arch}_N4000_seed42.pt")
            print(f"N={n} seed={s} {name}: Mean Accuracy (6 unseen) {t.iloc[-1]['Doğruluk']*100:.1f}  |  GPU {gpu:.2f} GB  |  {sure:.0f} sn")
            del m; gc.collect(); torch.cuda.empty_cache()
print(f"Toplam süre: {(time.time()-t_all)/60:.1f} dk")

A = pd.concat(rows, ignore_index=True).rename(columns=REN)
A.to_csv("/kaggle/working/deneyA_veri_miktari.csv", index=False)
K = pd.DataFrame(kaynak); K.to_csv("/kaggle/working/deneyA_kaynak.csv", index=False)
mean6, sd15 = A[A["Generator"] == "Ort. (SD1.5 hariç 6)"], A[A["Generator"] == "SD1.5"]

def ms(df, col):
    g = df.groupby(["N", "Model"])[col]
    return ((g.mean() * 100).round(1).astype(str) + " ± " + (g.std() * 100).round(1).astype(str)).unstack()

for col in ["Accuracy", "ROC-AUC", "AP", "Fake acc.", "Real acc."]:
    print(f"\n=== Mean (6 unseen) {col} (%), 3 seeds ===\n" + ms(mean6, col).to_string())
print("\n=== SD1.5 Accuracy (%), 3 seeds ===\n" + ms(sd15, "Accuracy").to_string())
print("\n=== Per-generator Accuracy (%), N=4000 vs N=1000, mean of 3 seeds ===")
pg = A[A["N"].isin([1000, 4000])].groupby(["Generator", "N", "Model"])["Accuracy"].mean().unstack([1, 2]) * 100
print(pg.round(1).to_string())
print("\n=== Kaynak kullanımı (3 seed ortalaması) ===")
print(K.groupby(["Model", "N"])[["Süre (sn)", "GPU tepe (GB)", "RAM süreç (GB)", "RAM sistem (GB)"]].mean().round(2).to_string())
print(f"CLIP öznitelik çıkarma (8.000 görsel): {clip_sure:.0f} sn, GPU tepe {clip_gpu:.2f} GB")

ax = (mean6.groupby(["N", "Model"])["Accuracy"].mean().unstack() * 100).plot(marker="o", figsize=(6, 4))
ax.set_xticks(SIZES); ax.set_xlabel("Number of training images"); ax.set_ylabel("Mean Accuracy, 6 unseen (%)")
ax.set_title("Experiment A: training set size"); plt.tight_layout()
plt.savefig("/kaggle/working/deneyA_accuracy.png", dpi=150); plt.show()
