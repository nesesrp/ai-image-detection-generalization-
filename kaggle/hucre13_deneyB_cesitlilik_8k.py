# Hücre 13: Deney B — üretici çeşitliliği ve 8.000 eğitim görseli, 3 seed
# Hücre 12 aynı oturumda çalışmış olmalı (finetune, predict, clip_features, clip_run, F_train, F_test, TID, A, real_order, fake_order).
#   B-4k (kontrol): 1.000 SD1.5 + 1.000 Wukong yapay + 2.000 gerçek  -> A'nın N=4000 setiyle aynı boyut ve aynı gerçekler, yalnız çeşitlilik farklı
#   B-8k:           2.000 SD1.5 + 2.000 Wukong yapay + 4.000 gerçek
# Wukong artık görülen üretici; karşılaştırma 5 görülmeyen üreticide (ADM, BigGAN, GLIDE, Midjourney, VQDM).
# A-4k → B-4k: çeşitliliğin etkisi; B-4k → B-8k: görsel sayısının etkisi.
UNSEEN5 = ["ADM", "BigGAN", "GLIDE", "Midjourney", "VQDM"]
pd.set_option("display.width", 250)

rng = random.Random(42)
extra_real = rng.sample(sorted(set(real_train_pool) - set(real_train)), 2000)  # eğitimde henüz kullanılmamış gerçekler
real8 = real_order + extra_real          # ilk 2.000 = A'daki gerçek eğitim seti
wk_f = FAKE["Wukong"]["train"][:]; rng.shuffle(wk_f)
SETS_B = {
    "B-4k (SD1.5+Wukong)": [(p, 0) for p in real8[:2000]] + [(p, 1) for p in fake_order[:1000] + wk_f[:1000]],
    "B-8k (SD1.5+Wukong)": [(p, 0) for p in real8] + [(p, 1) for p in fake_order + wk_f],
}
for k, v in SETS_B.items():
    print(f"{k}: {len(v)} görsel (gerçek {sum(y == 0 for _, y in v)}, yapay {sum(y == 1 for _, y in v)})")

new = sorted({p for it in SETS_B.values() for p, _ in it} - set(F_train))
t0 = time.time()
F_train.update(dict(zip(new, clip_features([(p, 0) for p in new]))))
print(f"Yeni CLIP öznitelikleri: {len(new)} görsel, {time.time()-t0:.0f} sn")

rowsB, kaynakB = [], []
t_all = time.time()
for setname, items_n in SETS_B.items():
    for s in SEEDS:
        t = clip_run(items_n, s); t["Set"] = setname; t["Seed"] = s; rowsB.append(t)
        for name, arch in ARCHS:
            base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats(); t0 = time.time()
            m = finetune(arch, items_n, s)
            sure, gpu = time.time() - t0, (torch.cuda.max_memory_allocated() - base) / 1e9
            t = results_table(name, lambda items, m=m: predict(m, items)); t["Set"] = setname; t["Seed"] = s; rowsB.append(t)
            kaynakB.append({"Model": name, "Set": setname, "Seed": s, "Süre (sn)": sure, "GPU tepe (GB)": gpu,
                            "RAM süreç (GB)": psutil.Process().memory_info().rss / 1e9,
                            "RAM sistem (GB)": psutil.virtual_memory().used / 1e9})
            if setname.startswith("B-8k") and s == 42:
                torch.save(m.state_dict(), f"/kaggle/working/{arch}_B8k_seed42.pt")
            u5 = t[t["Üretici"].isin(UNSEEN5)]["Doğruluk"].mean() * 100
            print(f"{setname} seed={s} {name}: Mean Accuracy (5 unseen) {u5:.1f}  |  GPU {gpu:.2f} GB  |  {sure:.0f} sn")
            del m; gc.collect(); torch.cuda.empty_cache()
print(f"Toplam süre: {(time.time()-t_all)/60:.1f} dk")

B = pd.concat(rowsB, ignore_index=True).rename(columns=REN)
A4 = A[A["N"] == 4000].drop(columns="N").assign(Set="A-4k (SD1.5)")
AB = pd.concat([A4, B], ignore_index=True)
AB.to_csv("/kaggle/working/deneyB.csv", index=False)
KB = pd.DataFrame(kaynakB); KB.to_csv("/kaggle/working/deneyB_kaynak.csv", index=False)
ORDER = ["A-4k (SD1.5)", "B-4k (SD1.5+Wukong)", "B-8k (SD1.5+Wukong)"]
COLS = ["Accuracy", "ROC-AUC", "AP", "Fake acc.", "Real acc."]
U5 = AB[AB["Generator"].isin(UNSEEN5)].groupby(["Set", "Model", "Seed"])[COLS].mean().reset_index()

def ms2(df, col):
    g = df.groupby(["Set", "Model"])[col]
    return ((g.mean() * 100).round(1).astype(str) + " ± " + (g.std() * 100).round(1).astype(str)).unstack().loc[ORDER]

for col in COLS:
    print(f"\n=== Mean (5 unseen) {col} (%), 3 seeds ===\n" + ms2(U5, col).to_string())
print("\n=== Per-generator Accuracy (%), mean of 3 seeds (SD1.5 ve Wukong B'de görülen) ===")
pg = AB.groupby(["Generator", "Set", "Model"])["Accuracy"].mean().unstack(["Set", "Model"]) * 100
print(pg.loc[list(GEN), ORDER].round(1).to_string())
print("\n=== Kaynak kullanımı (3 seed ortalaması) ===")
print(KB.groupby(["Model", "Set"])[["Süre (sn)", "GPU tepe (GB)", "RAM süreç (GB)", "RAM sistem (GB)"]].mean().round(2).to_string())

ax = (U5.groupby(["Set", "Model"])["Accuracy"].mean().unstack().loc[ORDER] * 100).plot.bar(rot=0, figsize=(8, 4))
ax.set_ylabel("Mean Accuracy, 5 unseen (%)"); ax.set_xlabel(""); ax.set_ylim(50, 100)
ax.set_title("Experiment B: generator diversity vs. data size"); plt.tight_layout()
plt.savefig("/kaggle/working/deneyB_accuracy.png", dpi=150); plt.show()
