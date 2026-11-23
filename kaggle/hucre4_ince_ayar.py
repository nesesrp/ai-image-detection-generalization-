# Hücre 4: ResNet-50 ve ViT-B/16 ince ayar (ImageNet ön eğitimli, timm)
# Pilot: tek tohum (42), 3 epoch, AdamW, 224x224, doğrulama için SD1.5 eğitim setinin %10'u ayrılır
import timm, time
from torchvision import transforms as T_

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
tf_train = T_.Compose([T_.Resize(256), T_.RandomCrop(224), T_.RandomHorizontalFlip(), T_.ToTensor(), T_.Normalize(MEAN, STD)])
tf_test = T_.Compose([T_.Resize(256), T_.CenterCrop(224), T_.ToTensor(), T_.Normalize(MEAN, STD)])

def finetune(arch, epochs=3, lr=1e-4, bs=32):
    torch.manual_seed(SEED)
    items = train_items[:]
    random.Random(SEED).shuffle(items)
    n_val = len(items) // 10
    val_items, tr_items = items[:n_val], items[n_val:]
    model = timm.create_model(arch, pretrained=True, num_classes=1).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler()
    dl = DataLoader(ImgDS(tr_items, tf_train), batch_size=bs, shuffle=True, num_workers=4, drop_last=True)
    lossf = torch.nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        model.train(); t0 = time.time(); tot = 0
        for x, y in dl:
            x, y = x.to(DEVICE), y.float().to(DEVICE)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = lossf(model(x).squeeze(1), y)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * len(y)
        sched.step()
        vm = metrics([y for _, y in val_items], predict(model, val_items))
        print(f"{arch} epoch {ep+1}/{epochs}  kayıp {tot/len(tr_items):.4f}  doğrulama doğr. {vm['Doğruluk']:.3f}  ({time.time()-t0:.0f} sn)")
    return model

@torch.no_grad()
def predict(model, items, bs=64):
    model.eval(); out = []
    for x, _ in DataLoader(ImgDS(items, tf_test), batch_size=bs, num_workers=4):
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(torch.sigmoid(model(x.to(DEVICE)).float().squeeze(1)).cpu())
    return torch.cat(out).numpy()

res = {}
for name, arch in [("ResNet-50", "resnet50.a1_in1k"), ("ViT-B/16", "vit_base_patch16_224.augreg_in21k_ft_in1k")]:
    m = finetune(arch)
    res[name] = results_table(name, lambda items, m=m: predict(m, items))
    del m; torch.cuda.empty_cache()

res_all = pd.concat([res_clip, res["ResNet-50"], res["ViT-B/16"]], ignore_index=True)
res_all.to_csv("/kaggle/working/pilot_sonuclar.csv", index=False)
res_all.round(3)
