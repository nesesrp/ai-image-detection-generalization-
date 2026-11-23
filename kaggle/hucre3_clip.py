# Hücre 3: Donmuş CLIP ViT-L/14 + doğrusal sınıflandırıcı (lojistik regresyon)
import time
from transformers import CLIPModel, CLIPImageProcessor
from sklearn.linear_model import LogisticRegression

t0 = time.time()
clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).to(DEVICE).eval()
proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
clip_tf = lambda im: proc(images=im, return_tensors="pt")["pixel_values"][0]

@torch.no_grad()
def clip_features(items, bs=64):
    dl = DataLoader(ImgDS(items, clip_tf), batch_size=bs, num_workers=4)
    feats = []
    for x, _ in dl:
        pooled = clip.vision_model(pixel_values=x.to(DEVICE, torch.float16)).pooler_output
        f = clip.visual_projection(pooled)  # CLIP görüntü gömmesi (768 boyut)
        feats.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu())
    return torch.cat(feats).numpy()

Xtr = clip_features(train_items)
ytr = np.array([y for _, y in train_items])
lin = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr)
print(f"Eğitim doğruluğu: {lin.score(Xtr, ytr):.3f}  |  süre: {time.time()-t0:.0f} sn")

clip_cache = {}
def clip_prob(items):
    key = id(items)
    if key not in clip_cache:
        clip_cache[key] = clip_features(items)
    return lin.predict_proba(clip_cache[key])[:, 1]

res_clip = results_table("CLIP ViT-L/14 (donmuş) + LR", clip_prob)
print(f"Toplam süre: {time.time()-t0:.0f} sn")
res_clip.round(3)
