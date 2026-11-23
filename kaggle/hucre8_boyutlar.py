# Hücre 8: Klasör boyutları ve örnek görsel çözünürlükleri
import subprocess
print(subprocess.run(["du", "-sh"] + [os.path.join(U, d) for d in sorted(os.listdir(U))] + [U, T],
                     capture_output=True, text=True).stdout)
for name, folder in [("SD1.5 fake", os.path.join(U, "stable_diffusion_v_1_5")), ("Nature real", os.path.join(U, "Nature"))]:
    fs = sorted(os.listdir(folder))[:200]
    sizes = collections.Counter(Image.open(os.path.join(folder, f)).size for f in fs)
    print(name, "| ilk 200 görselde en sık boyutlar:", sizes.most_common(5))
