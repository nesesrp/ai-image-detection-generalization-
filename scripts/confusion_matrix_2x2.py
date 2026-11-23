"""2×2 deneyi için confusion matrix: her modelin Real acc. ve Fake acc. değerlerinden (5 görülmeyen üretici, 3 seed ortalaması; results/SONUCLAR.md) hesaplanır. Test: 500 gerçek + 5 × 500 yapay görsel. Çalıştırma: python scripts/confusion_matrix_2x2.py"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("blue", ["#ffffff","#cde2fb","#86b6ef","#3987e5","#1c5cab","#0d366b"])
D = {"Düzeltilmiş 4k (SD 1.5)": [("CLIP ViT-L/14",83.7,89.6),("CLIP ViT-H/14",83.2,95.3),("ResNet-50",79.6,86.8),("ResNet-101",83.8,83.9),("ViT-B/16",66.7,94.7),("ViT-L/16",69.5,92.8)],
     "İki katı 8k (SD 1.5 + Wukong)": [("CLIP ViT-L/14",85.0,91.7),("CLIP ViT-H/14",85.2,95.5),("ResNet-50",78.2,89.6),("ResNet-101",80.7,89.7),("ViT-B/16",63.2,94.4),("ViT-L/16",72.4,89.4)]}
NR, NF = 500, 2500
tr = lambda v: f"{v:.1f}".replace(".", ",")
th = lambda n: f"{n:,}".replace(",", ".")
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9})
fig, axes = plt.subplots(2, 6, figsize=(15, 6.2), gridspec_kw={"hspace": 0.45, "wspace": 0.08})
for r,(data,models) in enumerate(D.items()):
    for c,(m,fa,ra) in enumerate(models):
        ax = axes[r,c]
        pct = [[ra, 100-ra],[100-fa, fa]]
        tn = round(ra/100*NR); tp = round(fa/100*NF)
        cnt = [[tn, NR-tn],[NF-tp, tp]]
        ax.imshow(pct, cmap=cmap, vmin=0, vmax=100)
        for i in range(2):
            for j in range(2):
                v = pct[i][j]
                ax.text(j, i, f"%{tr(v)}\n({th(cnt[i][j])})", ha="center", va="center",
                        color="white" if v > 55 else "#1a1a1a", fontsize=9, fontweight="bold" if i==j else "normal")
        ax.set_xticks([0,1]); ax.set_xticklabels(["Gerçek","Yapay"]); ax.set_yticks([0,1])
        ax.set_yticklabels(["Gerçek\n(500)","Yapay\n(2.500)"] if c==0 else ["",""])
        ax.set_xlabel("Tahmin")
        if c==0: ax.set_ylabel(f"{data}\n\nGerçek sınıf", fontsize=9)
        ax.set_title(m, fontsize=10, fontweight="bold")
        for s in ax.spines.values(): s.set_visible(False)
        ax.tick_params(length=0)
fig.suptitle("Confusion matrix — 5 görülmeyen üretici (ADM, BigGAN, GLIDE, Midjourney, VQDM), 3 seed ortalaması, eşik 0,5", fontsize=11, y=0.98)
plt.savefig("results/confusion_matrix_2x2.png", dpi=200, facecolor="white", bbox_inches="tight")
