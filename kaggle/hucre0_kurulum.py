# Hücre 0: Kurulum (oturum yeniden başlarsa önce bunu çalıştır) = Hücre 1 + 2 + model tanımları
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

# Hücre 2: Ortak araçlar (veri yükleyici + ölçütler)
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ImgDS(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        p, y = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), y

def metrics(y, prob, thr=0.5):
    # Eşik 0,5 sabit; test setinde yeniden ayarlanmaz (Bölüm 3.2.2)
    y, prob = np.asarray(y), np.asarray(prob)
    pred = (prob >= thr).astype(int)
    return {
        "Doğruluk": accuracy_score(y, pred),
        "Gerçek doğr.": accuracy_score(y[y == 0], pred[y == 0]),
        "Sahte doğr.": accuracy_score(y[y == 1], pred[y == 1]),
        "ROC-AUC": roc_auc_score(y, prob),
        "AP": average_precision_score(y, prob),
    }

def results_table(model_name, prob_fn):
    # prob_fn(items) -> sahte olasılıkları
    rows = []
    for g, items in test_sets.items():
        m = metrics([y for _, y in items], prob_fn(items))
        rows.append({"Model": model_name, "Üretici": g, **m})
    df = pd.DataFrame(rows)
    others = df[df["Üretici"] != "SD1.5"]
    mean = {"Model": model_name, "Üretici": "Ort. (SD1.5 hariç 6)", **others.mean(numeric_only=True).to_dict()}
    df = pd.concat([df, pd.DataFrame([mean])], ignore_index=True)
    return df

# --- Model tanımları (eğitim yapmaz) ---
import time, timm
import matplotlib.pyplot as plt
from torchvision import transforms as T_
from transformers import CLIPModel, CLIPImageProcessor
from sklearn.linear_model import LogisticRegression

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
tf_train = T_.Compose([T_.Resize(256), T_.RandomCrop(224), T_.RandomHorizontalFlip(), T_.ToTensor(), T_.Normalize(MEAN, STD)])
tf_test = T_.Compose([T_.Resize(256), T_.CenterCrop(224), T_.ToTensor(), T_.Normalize(MEAN, STD)])
ARCHS = [("ResNet-50", "resnet50.a1_in1k"), ("ViT-B/16", "vit_base_patch16_224.augreg_in21k_ft_in1k")]
REN = {"Doğruluk": "Accuracy", "Gerçek doğr.": "Real acc.", "Sahte doğr.": "Fake acc.", "Üretici": "Generator", "Tohum": "Seed"}

clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).to(DEVICE).eval()
proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
clip_tf = lambda im: proc(images=im, return_tensors="pt")["pixel_values"][0]
print("Kurulum tamam.")
