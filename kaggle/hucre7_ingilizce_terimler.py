# Hücre 7: Tablo ve grafikleri İngilizce ölçüt adlarıyla yeniden üret
REN = {"Doğruluk": "Accuracy", "Gerçek doğr.": "Real acc.", "Sahte doğr.": "Fake acc.",
       "Üretici": "Generator", "Tohum": "Seed"}
AVG = "Ort. (SD1.5 hariç 6)"; AVG_EN = "Mean (6 unseen)"
S = seeds_all.rename(columns=REN).replace({AVG: AVG_EN})
S.to_csv("/kaggle/working/pilot_3seed.csv", index=False)

order = list(GEN) + [AVG_EN]
def mean_std(metric):
    g = S.groupby(["Generator", "Model"])[metric]
    return ((g.mean() * 100).round(1).astype(str) + " ± " + (g.std() * 100).round(1).astype(str)).unstack().loc[order]

for metric in ["Accuracy", "ROC-AUC", "AP", "Real acc.", "Fake acc."]:
    print(f"\n=== {metric} (%), 3 seeds, mean ± std ===")
    print(mean_std(metric).to_string())

per = S[S["Generator"] != AVG_EN]
m = per.groupby(["Generator", "Model"])["Accuracy"].mean().unstack().loc[list(GEN)] * 100
s = per.groupby(["Generator", "Model"])["Accuracy"].std().unstack().loc[list(GEN)] * 100
ax = m.plot.bar(yerr=s, capsize=3, figsize=(10, 4), rot=0)
ax.axhline(50, color="gray", ls="--", lw=1)
ax.set_ylabel("Accuracy (%)"); ax.set_xlabel("Test üreticisi"); ax.set_ylim(0, 100)
ax.set_title("Pilot (3 seed): SD 1.5 ile eğitim, üreticiler arası genelleme")
ax.legend(loc="lower left")
plt.tight_layout(); plt.savefig("/kaggle/working/pilot_3seed_accuracy.png", dpi=150); plt.show()
