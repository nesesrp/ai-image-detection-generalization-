# Hücre 1: Veri indeksi (Unbiased Tiny GenImage)
# Eğitim: SD 1.5 sahte (Tiny GenImage train ayrımı) + Nature gerçek (ImageNet train adları)
# Test: her üreticinin sahte val görselleri + ortak gerçek test seti (ImageNet val adları)
import os, random, numpy as np, torch

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

U = "/kaggle/input/datasets/cartografia/unbiased-tiny-genimage"
T = "/kaggle/input/datasets/yangsangtai/tiny-genimage"
GEN = {  # ad: (Unbiased klasörü, orijinal Tiny GenImage klasörü)
    "SD1.5": ("stable_diffusion_v_1_5", "imagenet_ai_0424_sdv5"),
    "ADM": ("ADM", "imagenet_ai_0508_adm"),
    "BigGAN": ("BigGAN", "imagenet_ai_0419_biggan"),
    "GLIDE": ("glide", "imagenet_glide"),
    "Midjourney": ("Midjourney", "imagenet_midjourney"),
    "VQDM": ("VQDM", "imagenet_ai_0419_vqdm"),
    "Wukong": ("wukong", "imagenet_ai_0424_wukong"),
}
stem = lambda f: os.path.splitext(f)[0]

def fake_split(g):
    u_dir, t_dir = GEN[g]
    files = {stem(f): os.path.join(U, u_dir, f) for f in os.listdir(os.path.join(U, u_dir))}
    out = {}
    for sp in ["train", "val"]:
        names = {stem(f) for f in os.listdir(os.path.join(T, t_dir, sp, "ai"))}
        out[sp] = sorted(files[n] for n in names if n in files)
    return out

FAKE = {g: fake_split(g) for g in GEN}
nat = sorted(os.listdir(os.path.join(U, "Nature")))
real_train_pool = [os.path.join(U, "Nature", f) for f in nat if not f.startswith("ILSVRC2012_val")]
real_test_pool = [os.path.join(U, "Nature", f) for f in nat if f.startswith("ILSVRC2012_val")]

rng = random.Random(SEED)
real_train = rng.sample(real_train_pool, min(2000, len(real_train_pool)))
real_test = rng.sample(real_test_pool, min(500, len(real_test_pool)))

train_items = [(p, 0) for p in real_train] + [(p, 1) for p in FAKE["SD1.5"]["train"]]
test_sets = {g: [(p, 0) for p in real_test] + [(p, 1) for p in FAKE[g]["val"]] for g in GEN}

print("Gerçek havuz  train:", len(real_train_pool), " test:", len(real_test_pool))
for g in GEN:
    print(f"{g:11s} sahte train: {len(FAKE[g]['train']):4d}  sahte val: {len(FAKE[g]['val']):4d}")
print("Eğitim seti:", len(train_items), "(gerçek", len(real_train), "+ sahte", len(FAKE["SD1.5"]["train"]), ")")
