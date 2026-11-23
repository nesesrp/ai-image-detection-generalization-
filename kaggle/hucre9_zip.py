# Hücre 9: Eğitim verisini indirmek için tek zip (SD 1.5 fake + Nature real + kullanılan görsellerin listesi)
import shutil, subprocess
tmp = "/kaggle/temp/egitim_verisi"
shutil.rmtree("/kaggle/temp", ignore_errors=True); os.makedirs(tmp)
os.symlink(os.path.join(U, "stable_diffusion_v_1_5"), os.path.join(tmp, "fake_SD1.5"))
os.symlink(os.path.join(U, "Nature"), os.path.join(tmp, "real_Nature"))
shutil.copy(os.path.join(U, "nature_metadata.csv"), tmp)

# Pilotta hangi görsel nerede kullanıldı (seed 42: ilk %10 validation)
items = train_items[:]; random.Random(42).shuffle(items)
rows = [(os.path.basename(p), "fake" if y else "real", "validation" if i < len(items) // 10 else "train") for i, (p, y) in enumerate(items)]
rows += [(os.path.basename(p), "real", "test") for p in real_test]
rows += [(os.path.basename(p), "fake", "test") for p in FAKE["SD1.5"]["val"]]
pd.DataFrame(rows, columns=["file", "label", "split_seed42"]).to_csv(os.path.join(tmp, "pilotta_kullanilanlar.csv"), index=False)

out = "/kaggle/working/egitim_verisi.zip"
subprocess.run(["zip", "-r", "-0", "-q", out, "egitim_verisi"], cwd="/kaggle/temp", check=True)
print(f"{out}: {os.path.getsize(out)/1e9:.2f} GB")
