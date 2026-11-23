# AI Image Detection — Generalization

Yapay zekâ ile üretilmiş görsellerin tespiti: bir üreticide (Stable Diffusion 1.5) eğitilen modellerin görülmemiş üreticilere (ADM, BigGAN, GLIDE, Midjourney, VQDM, Wukong) genellenmesi.

**Modeller:** ResNet-50 / ResNet-101, ViT-B/16 / ViT-L/16 (ince ayar), dondurulmuş CLIP ViT-L/14 / ViT-H/14 + Logistic Regression.
**Veri:** [Unbiased Tiny GenImage](https://www.kaggle.com/datasets/cartografia/unbiased-tiny-genimage) ve [Tiny GenImage](https://www.kaggle.com/datasets/yangsangtai/tiny-genimage) (Kaggle).
**Ortam:** Kaggle, Tesla T4 (15 GB).

## Klasörler
| Klasör | İçerik |
|---|---|
| `kaggle/` | Kaggle not defteri hücreleri (`hucreN_*.py`) |
| `scripts/veri_duzelt.py` | Yerel veri setlerini 224×224 JPEG (kalite 96) formatına getirir |
| `results/SONUCLAR.md` | Tüm deney sonuçları (Accuracy, ROC-AUC, AP, kaynak kullanımı) |

## Hücre sırası
Hücreler aynı kernel oturumunda sırayla çalışmalıdır; oturum yeniden başlarsa önce `hucre0_kurulum.py`.
- `hucre1`–`hucre9`: pilot (veri indeksi, CLIP, ince ayar, 3 seed, özet)
- `hucre10`: metadata kestirme testi (Random Forest)
- `hucre11`: D4/D5 mini deney
- `hucre12`: Deney A — veri miktarı (1k / 2k / 4k)
- `hucre13`: Deney B — üretici çeşitliliği (SD1.5 + Wukong, 8k)
- `hucre14`: 2×2 deney, düzeltilmiş veri + temel modeller (4k ve 8k)
- `hucre15`: 2×2 deney, düzeltilmiş veri + büyük modeller (4k ve 8k)

## Veri düzeltme
Her görsel özgün çözünürlükte 224×224 merkezden kırpılır (kısa kenarı 224'ten küçükse önce bicubic büyütme), JPEG kalite 96 ile kaydedilir. Böylece gerçek ve yapay görseller arasında boyut/format kestirmesi kalmaz.

```bash
python scripts/veri_duzelt.py "<kaynak klasör>" "<hedef klasör>"
```

## Özet sonuç (5 görülmeyen üretici, Mean Accuracy %, 3 seed)
| Veri | CLIP L/14 → H/14 | ResNet-50 → 101 | ViT-B/16 → L/16 |
|---|---|---|---|
| Düzeltilmiş 4k (SD1.5) | 86,6 → 89,2 | 83,2 → 83,8 | 80,7 → 81,2 |
| Düzeltilmiş 8k (SD1.5 + Wukong) | 88,4 → 90,3 | 83,9 → 85,2 | 78,8 → 80,9 |
