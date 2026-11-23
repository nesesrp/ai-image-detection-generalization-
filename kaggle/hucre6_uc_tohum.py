# Hücre 6: Üç modeli 3 tohumla (42, 43, 44) eğit ve test et; ortalama ± standart sapma
# Tohum; doğrulama ayrımını (%10), veri sırasını, veri artırmayı ve ilk ağırlıkları belirler.
# CLIP'te kodlayıcı donmuş; tohum yalnızca doğrulama ayrımını değiştirir (LR, eğitim setinin %90'ıyla eğitilir).
SEEDS = [42, 43, 44]
ARCHS = [("ResNet-50", "resnet50.a1_in1k"), ("ViT-B/16", "vit_base_patch16_224.augreg_in21k_ft_in1k")]

def clip_seed(s):
    idx = list(range(len(train_items)))
    random.Random(s).shuffle(idx)          # finetune() ile aynı %10 doğrulama ayrımı
    tr = idx[len(idx) // 10:]
    lr_ = LogisticRegression(max_iter=2000, C=1.0, random_state=s).fit(Xtr[tr], ytr[tr])
    return results_table("CLIP ViT-L/14 (donmuş) + LR",
                         lambda items: lr_.predict_proba(clip_cache[id(items)])[:, 1])

seed_tables = []
t_all = time.time()
for s in SEEDS:
    t = clip_seed(s); t["Tohum"] = s; seed_tables.append(t)
    for name, arch in ARCHS:
        if s == 42 and name in res:        # 42 tohumu zaten eğitildi (Hücre 4)
            t = res[name].copy()
        else:
            SEED = s
            m = finetune(arch)
            t = results_table(name, lambda items, m=m: predict(m, items))
            del m; torch.cuda.empty_cache()
        t["Tohum"] = s; seed_tables.append(t)
        print(f"tohum {s} | {name} bitti | ort. doğruluk (6 üretici): {t.iloc[-1]['Doğruluk']*100:.1f}")
SEED = 42

seeds_all = pd.concat(seed_tables, ignore_index=True)
seeds_all.to_csv("/kaggle/working/pilot_3tohum.csv", index=False)
print(f"Toplam süre: {(time.time()-t_all)/60:.1f} dk")

order = list(GEN) + ["Ort. (SD1.5 hariç 6)"]
def mean_std(metric):
    g = seeds_all.groupby(["Üretici", "Model"])[metric]
    tab = (g.mean() * 100).round(1).astype(str) + " ± " + (g.std() * 100).round(1).astype(str)
    return tab.unstack().loc[order]

for metric in ["Doğruluk", "ROC-AUC", "AP"]:
    print(f"\n=== {metric} (%), 3 tohum, ortalama ± std ===")
    print(mean_std(metric).to_string())

# Grafik: 6 üretici ortalaması, hata çubuğu = std
avg = seeds_all[seeds_all["Üretici"] == "Ort. (SD1.5 hariç 6)"].groupby("Model")["Doğruluk"].agg(["mean", "std"]) * 100
tab_m = seeds_all[seeds_all["Üretici"] != "Ort. (SD1.5 hariç 6)"].groupby(["Üretici", "Model"])["Doğruluk"].mean().unstack().loc[list(GEN)] * 100
tab_s = seeds_all[seeds_all["Üretici"] != "Ort. (SD1.5 hariç 6)"].groupby(["Üretici", "Model"])["Doğruluk"].std().unstack().loc[list(GEN)] * 100
ax = tab_m.plot.bar(yerr=tab_s, capsize=3, figsize=(10, 4), rot=0)
ax.axhline(50, color="gray", ls="--", lw=1)
ax.set_ylabel("Doğruluk (%)"); ax.set_xlabel("Test üreticisi"); ax.set_ylim(0, 100)
ax.set_title("Pilot (3 tohum): SD 1.5 ile eğitim, üreticiler arası genelleme")
plt.tight_layout(); plt.savefig("/kaggle/working/pilot_3tohum.png", dpi=150); plt.show()
