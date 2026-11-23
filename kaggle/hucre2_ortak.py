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
