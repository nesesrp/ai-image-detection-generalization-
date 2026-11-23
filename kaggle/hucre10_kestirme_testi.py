# Hücre 10: Kestirme (shortcut) testi — görüntüye bakmadan, yalnız dosya özellikleriyle sınıflandırma
# Özellikler: genişlik, yükseklik, en/boy oranı, piksel başına bayt, PNG mi, JPEG kuantizasyon ortalaması (kalite göstergesi)
from sklearn.ensemble import RandomForestClassifier

def tg_items(t_dir, sp):  # orijinal Tiny GenImage: fake PNG, real JPEG
    base = os.path.join(T, t_dir, sp)
    return ([(os.path.join(base, "nature", f), 0) for f in sorted(os.listdir(os.path.join(base, "nature")))] +
            [(os.path.join(base, "ai", f), 1) for f in sorted(os.listdir(os.path.join(base, "ai")))])

train_orig = tg_items(GEN["SD1.5"][1], "train")
test_orig = {g: tg_items(GEN[g][1], "val") for g in GEN}

def meta(p):
    im = Image.open(p); w, h = im.size
    q = getattr(im, "quantization", None)
    qv = float(np.mean(q[0])) if q else 0.0
    return [w, h, w / h, os.path.getsize(p) / (w * h), int(im.format == "PNG"), qv]
meta_X = lambda items: np.array([meta(p) for p, _ in items])

rows = []
for src, tr_items, sets in [("Unbiased", train_items, test_sets), ("Original", train_orig, test_orig)]:
    rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(meta_X(tr_items), [y for _, y in tr_items])
    for g, items in sets.items():
        rows.append({"Data": src, "Generator": g, **metrics([y for _, y in items], rf.predict_proba(meta_X(items))[:, 1])})
shortcut = pd.DataFrame(rows).rename(columns=REN)
shortcut.to_csv("/kaggle/working/kestirme_testi.csv", index=False)
print("=== Metadata-only classifier, Accuracy (%) — %50 = kestirme yok ===")
print((shortcut.pivot(index="Generator", columns="Data", values="Accuracy").loc[list(GEN)] * 100).round(1).to_string())
