# Hücre 5: Sonuçları birleştir, özet tablo ve grafik
import matplotlib.pyplot as plt

res_all = pd.concat([res_clip, res["ResNet-50"], res["ViT-B/16"]], ignore_index=True)
res_all.to_csv("/kaggle/working/pilot_sonuclar.csv", index=False)

order = list(GEN) + ["Ort. (SD1.5 hariç 6)"]
for metric in ["Doğruluk", "ROC-AUC"]:
    tab = res_all.pivot(index="Üretici", columns="Model", values=metric).loc[order]
    print(f"\n=== {metric} (eğitim: SD 1.5, eşik 0,5) ===")
    print((tab * 100).round(1).to_string())

tab = res_all.pivot(index="Üretici", columns="Model", values="Doğruluk").loc[list(GEN)] * 100
ax = tab.plot.bar(figsize=(10, 4), rot=0)
ax.axhline(50, color="gray", ls="--", lw=1)
ax.set_ylabel("Doğruluk (%)"); ax.set_xlabel("Test üreticisi"); ax.set_ylim(0, 100)
ax.set_title("Pilot: SD 1.5 ile eğitim, üreticiler arası genelleme (Unbiased Tiny GenImage)")
plt.tight_layout(); plt.savefig("/kaggle/working/pilot_dogruluk.png", dpi=150); plt.show()
