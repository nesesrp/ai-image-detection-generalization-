# Pilot deney sonuçları (16.11.2026)

Kaggle not defteri: `neeesarp/notebook9f7a80ae73` (GPU: Tesla T4). Kod: bu klasördeki `hucre1–7_*.py`.
Çıktılar (Kaggle `/kaggle/working`): `pilot_3seed.csv`, `pilot_3seed_accuracy.png`.

## Kurulum
- **Veri:** Unbiased Tiny GenImage (Kaggle: cartografia/unbiased-tiny-genimage). Train/val ayrımı orijinal Tiny GenImage'dan (yangsangtai/tiny-genimage) dosya adıyla alındı.
- **Eğitim:** SD 1.5, 2.000 fake + 2.000 real (Nature, ImageNet train adları); %10'u validation.
- **Test:** her üretici için 500 fake (val) + ortak 500 real (Nature, ImageNet val adları). Eğitimle çakışma yok.
- **Modeller:** donmuş CLIP ViT-L/14 + logistic regression; ResNet-50 (timm `resnet50.a1_in1k`); ViT-B/16 (timm `vit_base_patch16_224.augreg_in21k_ft_in1k`). Fine-tuning: 3 epoch, AdamW lr 1e-4, wd 0,05, cosine, 224×224, fp16.
- **Eşik:** 0,5 sabit. **3 seed (42, 43, 44)**; değerler mean ± std. Seed; validation ayrımını, veri sırasını, augmentation'ı ve ilk ağırlıkları belirler. CLIP'te kodlayıcı donmuş olduğu için std çok küçük.
- 2 ek seed (4 fine-tuning koşusu) 7,3 dk sürdü.

## Accuracy (%)
| Generator | CLIP + LR | ResNet-50 | ViT-B/16 |
|---|---|---|---|
| SD 1.5 (eğitim) | 92,0 ± 0,1 | 95,7 ± 0,9 | 96,0 ± 2,1 |
| ADM | 74,5 ± 0,3 | 73,8 ± 2,0 | 75,6 ± 0,9 |
| BigGAN | 75,6 ± 0,3 | 94,9 ± 0,8 | 80,1 ± 2,7 |
| GLIDE | 87,7 ± 0,3 | 93,2 ± 0,4 | 82,0 ± 1,7 |
| Midjourney | 89,4 ± 0,1 | 87,7 ± 0,7 | 77,4 ± 2,0 |
| VQDM | 78,8 ± 0,4 | 78,3 ± 2,3 | 77,4 ± 1,2 |
| Wukong | 91,3 ± 0,1 | 95,5 ± 0,6 | 94,9 ± 1,7 |
| **Mean (6 unseen)** | **82,9 ± 0,2** | **87,2 ± 0,6** | **81,2 ± 0,9** |

## ROC-AUC (%)
| Generator | CLIP + LR | ResNet-50 | ViT-B/16 |
|---|---|---|---|
| SD 1.5 | 98,1 ± 0,0 | 98,9 ± 0,2 | 99,4 ± 0,1 |
| ADM | 85,9 ± 0,3 | 87,0 ± 1,6 | 88,1 ± 3,4 |
| BigGAN | 87,0 ± 0,3 | 97,7 ± 0,5 | 91,6 ± 3,9 |
| GLIDE | 94,9 ± 0,2 | 97,3 ± 0,2 | 93,2 ± 2,8 |
| Midjourney | 96,4 ± 0,1 | 95,4 ± 0,2 | 89,7 ± 1,6 |
| VQDM | 89,1 ± 0,2 | 90,4 ± 0,7 | 89,5 ± 3,1 |
| Wukong | 97,5 ± 0,1 | 98,6 ± 0,1 | 99,1 ± 0,1 |
| **Mean (6 unseen)** | **91,8 ± 0,2** | **94,4 ± 0,2** | **91,8 ± 2,5** |

## AP (%)
| Generator | CLIP + LR | ResNet-50 | ViT-B/16 |
|---|---|---|---|
| SD 1.5 | 98,1 ± 0,1 | 98,2 ± 0,4 | 99,4 ± 0,1 |
| ADM | 84,7 ± 0,4 | 85,6 ± 1,0 | 87,9 ± 3,5 |
| BigGAN | 83,5 ± 0,5 | 95,2 ± 0,9 | 91,6 ± 3,5 |
| GLIDE | 94,4 ± 0,2 | 95,6 ± 0,3 | 92,9 ± 2,5 |
| Midjourney | 96,3 ± 0,1 | 93,7 ± 0,4 | 89,9 ± 1,1 |
| VQDM | 87,9 ± 0,3 | 88,6 ± 0,5 | 89,5 ± 2,6 |
| Wukong | 97,3 ± 0,1 | 98,0 ± 0,3 | 99,0 ± 0,1 |
| **Mean (6 unseen)** | **90,7 ± 0,3** | **92,8 ± 0,1** | **91,8 ± 2,2** |

## Real acc. ve Fake acc. (%)
Real acc. (ortak 500 real, her üreticide aynı): CLIP 88,1 ± 0,1 · ResNet-50 92,7 ± 1,5 · ViT-B/16 93,4 ± 4,8

| Fake acc. | CLIP + LR | ResNet-50 | ViT-B/16 |
|---|---|---|---|
| SD 1.5 | 95,9 ± 0,1 | 98,8 ± 0,2 | 98,5 ± 0,8 |
| ADM | 60,8 ± 0,7 | 54,9 ± 5,6 | 57,8 ± 3,2 |
| BigGAN | 63,0 ± 0,5 | 97,1 ± 0,1 | 66,7 ± 3,3 |
| GLIDE | 87,2 ± 0,4 | 93,7 ± 2,2 | 70,7 ± 2,3 |
| Midjourney | 90,7 ± 0,1 | 82,8 ± 2,6 | 61,4 ± 8,8 |
| VQDM | 69,5 ± 0,7 | 63,9 ± 6,0 | 61,4 ± 4,0 |
| Wukong | 94,4 ± 0,0 | 98,3 ± 0,4 | 96,4 ± 1,4 |
| **Mean (6 unseen)** | **77,6 ± 0,4** | **81,8 ± 2,7** | **69,1 ± 3,2** |

## Gözlemler
- Eğitim üreticisinde (SD 1.5) accuracy %92–96; görülmeyen 6 üreticide ortalama %81–87.
- ResNet-50, 3 seed'de de en yüksek mean accuracy ve ROC-AUC'ye sahip; std küçük (0,6 ve 0,2).
- ViT-B/16 en kararsız model (ROC-AUC std 2,5; Midjourney fake acc. std 8,8). Eğitim kaybı 0,01'in altına iniyor.
- Accuracy, real tarafın yüksekliğiyle şişiyor: ADM'de fake acc. %55–61, VQDM'de %61–70 (rastgeleye yakın).
- CLIP Midjourney'de, ResNet-50 BigGAN ve GLIDE'da öne çıkıyor.

## Dikkat
- GenImage Tablo 2'deki %62,3 temel çizgisiyle **doğrudan karşılaştırılamaz** (Tiny veri, iki sınıf da JPEG, 1.000 görsellik test, 224 px).
- Real test seti tüm üreticilerde ortak (874 ImageNet-val real görselden 500'ü); rapordaki "6 × (500 fake + 500 real)" planından farklı.

## Kestirme (shortcut) testi — yalnız dosya özellikleriyle sınıflandırma
Özellikler: genişlik, yükseklik, en/boy oranı, piksel başına bayt, PNG mi, JPEG kuantizasyon ortalaması. Random Forest (200 ağaç), SD 1.5 eğitim seti, eşik 0,5.

| Generator | Original (PNG/JPEG) | Unbiased (JPEG/JPEG) |
|---|---|---|
| SD1.5 | 100,0 | 100,0 |
| ADM | 100,0 | 50,0 |
| BigGAN | 100,0 | 50,0 |
| GLIDE | 100,0 | 50,0 |
| Midjourney | 100,0 | 100,0 |
| VQDM | 100,0 | 50,0 |
| Wukong | 100,0 | 100,0 |

Yorum: Orijinal sette biçim (PNG/JPEG) tek başına sınıfı ele veriyor. Unbiased sette de **boyut kestirmesi** var:
SD 1.5 fake'leri 512×512, real'ler ~500 px (yerel kopyadan doğrulandı: ADM/GLIDE/VQDM 256×256, BigGAN 128×128,
Midjourney 1024×1024, Wukong 512×512). Sınıflandırıcı "512 ve üzeri ise fake" kuralını öğrenmiş; 512+ üreticilerde %100,
256 ve altında %50 (hepsine real diyor). İnce ayarlı modeller görüntüyü 256'ya küçültüp 224 kırptığı için boyut doğrudan
girdi değil, ama yeniden örnekleme izleri üzerinden etkisi olabilir; tam deneylerde native çözünürlükte rastgele kırpma
gibi önlemler danışmanla değerlendirilecek.

## D4 mini — biçim yanlılığının etkisi (seed 42), Mean (6 unseen), %
| Train → Test | CLIP Acc / Fake | ResNet-50 Acc / Fake | ViT-B/16 Acc / Fake |
|---|---|---|---|
| Original → Original | 77,1 / 62,4 | 62,3 / 33,0 | 59,6 / 25,2 |
| Original → Unbiased | 62,7 / 46,3 | 58,8 / 32,1 | 58,2 / 24,8 |
| Unbiased → Original | 62,4 / 64,9 | 66,4 / 64,1 | 62,4 / 45,2 |
| Unbiased → Unbiased | 82,7 / 77,2 | 86,5 / 79,7 | 80,1 / 64,7 |

Real acc.: Orig→Orig 91,8 / 91,5 / 94,0 · Orig→Unb 79,0 / 85,6 / 91,6 · Unb→Orig 59,9 / 68,8 / 79,7 · Unb→Unb 88,2 / 93,2 / 95,4
SD1.5 Accuracy: Orig→Orig 93,1 / 90,7 / 94,3 · Unb→Unb 92,0 / 96,2 / 97,1
Not: Original ile eğitilen ResNet-50'nin 6 üretici ortalaması %62,3 = GenImage Tablo 2 temel çizgisi.

## D5 mini — dönüşümler (Unbiased eğitim, seed 42), Mean (6 unseen), Accuracy (ROC-AUC) %
| Dönüşüm | CLIP | ResNet-50 | ViT-B/16 |
|---|---|---|---|
| yok | 82,7 (91,7) | 86,5 (94,1) | 80,1 (91,6) |
| JPEG q75 | 76,7 (83,9) | 73,8 (85,4) | 77,4 (85,9) |
| Resize 0,5× | 82,8 (91,6) | 84,8 (91,9) | 76,8 (88,7) |

SD1.5 Accuracy: yok 92,0 / 96,2 / 97,1 · JPEG q75 88,1 / 77,5 / 91,3 · Resize 90,4 / 87,1 / 95,7

## Deney A — veri miktarı (21.11.2026)
Kod: `hucre12_deneyA_veri_miktari.py` (Hücre 0'dan sonra). İç içe, dengeli alt kümeler (yarısı gerçek, yarısı SD 1.5); her boyutta %10 validation; 3 epoch; 3 seed (42, 43, 44); test setleri değişmedi. Toplam süre 25,0 dk (T4).

### Mean (6 unseen), %, 3 seeds
| N | Metric | CLIP + LR | ResNet-50 | ViT-B/16 |
|---|---|---|---|---|
| 1000 | Accuracy | 81,2 ± 0,1 | 73,4 ± 0,7 | 79,7 ± 0,9 |
| 2000 | Accuracy | 83,0 ± 0,1 | 83,1 ± 1,0 | 83,0 ± 3,4 |
| 4000 | Accuracy | 82,6 ± 0,1 | 87,5 ± 0,2 | 78,8 ± 3,7 |
| 1000 | ROC-AUC | 89,0 ± 0,1 | 85,0 ± 1,8 | 87,9 ± 0,9 |
| 2000 | ROC-AUC | 90,8 ± 0,0 | 92,0 ± 0,4 | 92,1 ± 2,2 |
| 4000 | ROC-AUC | 91,7 ± 0,1 | 94,8 ± 0,1 | 91,0 ± 2,3 |
| 1000 | AP | 86,9 ± 0,1 | 82,0 ± 2,0 | 86,2 ± 0,6 |
| 2000 | AP | 89,3 ± 0,1 | 89,6 ± 0,9 | 91,1 ± 2,4 |
| 4000 | AP | 90,5 ± 0,2 | 93,1 ± 0,3 | 91,3 ± 1,8 |
| 1000 | Fake acc. | 80,3 ± 0,4 | 57,3 ± 3,5 | 75,2 ± 1,2 |
| 2000 | Fake acc. | 80,3 ± 0,4 | 76,1 ± 3,8 | 74,8 ± 6,6 |
| 4000 | Fake acc. | 77,3 ± 0,3 | 81,7 ± 0,5 | 62,7 ± 8,8 |
| 1000 | Real acc. | 82,1 ± 0,3 | 89,4 ± 2,4 | 84,2 ± 1,5 |
| 2000 | Real acc. | 85,8 ± 0,2 | 90,1 ± 1,8 | 91,2 ± 0,5 |
| 4000 | Real acc. | 87,9 ± 0,1 | 93,3 ± 0,3 | 94,9 ± 1,4 |

SD1.5 Accuracy: N=1000 87,0 / 82,3 / 90,1 · N=2000 90,3 / 91,5 / 94,9 · N=4000 91,8 / 96,1 / 96,2 (CLIP / ResNet / ViT)

### Per-generator Accuracy (%), mean of 3 seeds — N=1000 → N=4000
| Generator | CLIP + LR | ResNet-50 | ViT-B/16 |
|---|---|---|---|
| ADM | 76,4 → 73,8 | 65,7 → 74,5 | 75,6 → 71,0 |
| BigGAN | 76,7 → 75,3 | 71,4 → 94,7 | 77,4 → 80,2 |
| GLIDE | 84,0 → 87,6 | 75,3 → 93,7 | 79,4 → 77,5 |
| Midjourney | 86,0 → 89,3 | 72,9 → 87,6 | 78,4 → 74,3 |
| VQDM | 77,0 → 78,6 | 71,9 → 78,6 | 77,9 → 74,8 |
| Wukong | 87,2 → 91,0 | 82,9 → 95,8 | 89,4 → 95,0 |

### Kaynak kullanımı (3 seed ortalaması, ölçüldü)
GPU tepe = `torch.cuda.max_memory_allocated()`, bellekteki CLIP payı çıkarılmış (PyTorch'un ayırdığı en yüksek bellek; nvidia-smi CUDA bağlamı ve önbellek yüzünden biraz daha fazla gösterir).
| Model | N | Süre (sn) | GPU tepe (GB) | RAM süreç (GB) | RAM sistem (GB) |
|---|---|---|---|---|---|
| ResNet-50 | 1000 | 19 | 1,88 | 3,32 | 4,21 |
| ResNet-50 | 2000 | 32 | 1,88 | 3,47 | 4,26 |
| ResNet-50 | 4000 | 60 | 1,88 | 3,48 | 4,30 |
| ViT-B/16 | 1000 | 35 | 3,73 | 3,45 | 4,18 |
| ViT-B/16 | 2000 | 60 | 3,73 | 3,47 | 4,20 |
| ViT-B/16 | 4000 | 113 | 3,73 | 3,48 | 4,25 |
CLIP öznitelik çıkarma (8.000 görsel, fp16, batch 64): 195 sn, GPU tepe 1,43 GB.
N=4000, seed 42 modelleri `/kaggle/working/*_N4000_seed42.pt` olarak kaydedildi.

### Gözlemler
- ResNet-50 veri arttıkça düzenli iyileşiyor (73,4 → 83,1 → 87,5) ve 4.000'de hâlâ yükseliyor; Fake acc. 57 → 82.
- CLIP + LR 2.000'de doyuyor (Accuracy ~83); ROC-AUC küçük artıyor (89,0 → 91,7). Fake acc. hafif düşüp Real acc. artıyor: sabit 0,5 eşiğinde karar sınırı gerçek tarafa kayıyor.
- ViT-B/16'da veri artınca SD 1.5 başarısı artarken (90 → 96) görülmeyen üreticilerde düşüş ve büyük std var (Fake acc. 62,7 ± 8,8): tek üreticiye aşırı uyum.
- GPU belleği veri miktarına değil batch size ve modele bağlı; süre veriyle doğrusal artıyor.
- N=4000 değerleri pilotla tutarlı (CLIP 82,6 vs 82,9; ResNet 87,5 vs 87,2); ViT 78,8 vs 81,2 (yüksek std, seed 44 = 76,9).

## Deney B — üretici çeşitliliği ve 8.000 görsel (21.11.2026)
Kod: `hucre13_deneyB_cesitlilik_8k.py` (Hücre 12'den sonra, aynı oturumda). 3 epoch, 3 seed. Toplam 31,0 dk.
- A-4k: 2.000 SD1.5 + 2.000 gerçek (Deney A, N=4000)
- B-4k: 1.000 SD1.5 + 1.000 Wukong + 2.000 gerçek (aynı gerçekler; yalnız çeşitlilik farklı)
- B-8k: 2.000 SD1.5 + 2.000 Wukong + 4.000 gerçek (ek 2.000 gerçek, eğitimde kullanılmamış ImageNet-train adlı Nature görselleri)
Wukong B'de görülen üretici olduğu için karşılaştırma 5 görülmeyen üreticide (ADM, BigGAN, GLIDE, Midjourney, VQDM).

### Mean (5 unseen), %, 3 seeds
| Set | Metric | CLIP + LR | ResNet-50 | ViT-B/16 |
|---|---|---|---|---|
| A-4k | Accuracy | 80,9 ± 0,1 | 85,8 ± 0,3 | 75,6 ± 4,3 |
| B-4k | Accuracy | 83,6 ± 0,3 | 85,3 ± 0,0 | 76,4 ± 1,1 |
| B-8k | Accuracy | 84,2 ± 0,0 | 81,4 ± 1,6 | 74,1 ± 1,6 |
| A-4k | ROC-AUC | 90,5 ± 0,2 | 94,0 ± 0,2 | 89,4 ± 2,7 |
| B-4k | ROC-AUC | 91,9 ± 0,1 | 93,5 ± 0,2 | 88,1 ± 0,5 |
| B-8k | ROC-AUC | 93,1 ± 0,0 | 92,8 ± 0,7 | 87,9 ± 0,5 |
| A-4k | AP | 89,2 ± 0,2 | 92,1 ± 0,3 | 89,8 ± 2,1 |
| B-4k | AP | 91,0 ± 0,1 | 91,5 ± 0,5 | 87,9 ± 0,2 |
| B-8k | AP | 92,6 ± 0,0 | 91,1 ± 0,7 | 86,8 ± 0,5 |
| A-4k | Fake acc. | 74,0 ± 0,4 | 78,3 ± 0,7 | 56,2 ± 9,9 |
| B-4k | Fake acc. | 80,7 ± 0,4 | 78,2 ± 0,3 | 61,1 ± 2,1 |
| B-8k | Fake acc. | 80,2 ± 0,3 | 68,9 ± 4,3 | 54,0 ± 5,4 |
| A-4k | Real acc. | 87,9 ± 0,1 | 93,3 ± 0,3 | 94,9 ± 1,4 |
| B-4k | Real acc. | 86,4 ± 0,2 | 92,5 ± 0,3 | 91,6 ± 0,2 |
| B-8k | Real acc. | 88,3 ± 0,2 | 93,8 ± 1,3 | 94,2 ± 2,3 |

### Per-generator Accuracy (%), mean of 3 seeds — A-4k / B-4k / B-8k
| Generator | CLIP + LR | ResNet-50 | ViT-B/16 |
|---|---|---|---|
| SD1.5 (görülen) | 91,8 / 91,0 / 92,6 | 96,1 / 95,9 / 96,7 | 96,2 / 95,2 / 96,4 |
| Wukong (B'de görülen) | 91,0 / 91,4 / 92,8 | 95,8 / 95,6 / 96,7 | 95,0 / 95,2 / 96,8 |
| ADM | 73,8 / 77,7 / 77,8 | 74,5 / 73,5 / 70,2 | 71,0 / 73,0 / 71,1 |
| BigGAN | 75,3 / 81,3 / 82,3 | 94,7 / 93,8 / 86,8 | 80,2 / 77,5 / 73,9 |
| GLIDE | 87,6 / 87,9 / 89,1 | 93,7 / 92,6 / 86,1 | 77,5 / 79,8 / 75,6 |
| Midjourney | 89,3 / 88,7 / 89,2 | 87,6 / 88,3 / 89,0 | 74,3 / 75,4 / 74,1 |
| VQDM | 78,6 / 82,0 / 82,9 | 78,6 / 78,4 / 74,7 | 74,8 / 76,1 / 75,8 |

### Kaynak kullanımı (3 seed ortalaması)
| Model | Set | Süre (sn) | GPU tepe (GB) | RAM süreç (GB) | RAM sistem (GB) |
|---|---|---|---|---|---|
| ResNet-50 | B-4k | 59 | 1,87 | 3,50 | 4,35 |
| ResNet-50 | B-8k | 113 | 1,88 | 3,50 | 4,32 |
| ViT-B/16 | B-4k | 112 | 3,73 | 3,51 | 4,32 |
| ViT-B/16 | B-8k | 217 | 3,73 | 3,51 | 4,28 |
Yeni CLIP öznitelikleri (4.000 görsel): 84 sn. B-8k seed 42 modelleri `/kaggle/working/*_B8k_seed42.pt`.

### Gözlemler
- CLIP çeşitlilikten yararlanıyor (80,9 → 83,6; Fake acc. 74 → 81); 8k'ya çıkmak küçük ek kazanç (84,2; ROC-AUC 93,1).
- ResNet-50'de çeşitlilik etkisiz (85,8 → 85,3); 8k'da düşüyor (81,4 ± 1,6; Fake acc. 78 → 69). Görülen üreticilerde artış (SD1.5 ve Wukong 96,7), BigGAN (94,7 → 86,8) ve GLIDE'da (93,7 → 86,1) düşüş: görülen üreticilere aşırı uyum. Epoch sayısı sabit olduğu için 8k'da güncelleme adımı iki katı; bu da aşırı uyumu artırmış olabilir.
- ViT-B/16 her sette en zayıf (74–76), Fake acc. %54–61.
- Wukong, A'da hiç görülmeden %95,8 alıyordu; SD 1.5'e çok benzediği için eklediği çeşitlilik sınırlı. Daha farklı bir ikinci üretici (ör. GAN tabanlı) daha bilgilendirici olabilir.
- GPU belleği 8k'da da aynı (batch size belirliyor); süre iki katı.

## Danışmanın 2×2 deneyi — adım 1: düzeltilmiş veri, aynı modeller (21.11.2026)
Danışmanın istediği dört koşul: (1) düzeltilmiş veri + aynı model, (2) düzeltilmiş veri + büyük model, (3) iki katı veri + aynı model, (4) iki katı veri + büyük model.
Kod: `hucre14_hoca_duzeltilmis_kucuk.py` (Hücre 12–13'ten sonra). Toplam 30,9 dk.
- **Düzeltme** (rapor Bölüm 3 ön işleme kararı): tüm görseller özgün çözünürlükte 224×224 merkezden kırpıldı ve JPEG kalite 96 ile yeniden kaydedildi; kısa kenarı 224'ten küçük olan BigGAN (128 px) önce bicubic ile 224'e büyütüldü. Eğitimde ve testte yeniden boyutlandırma yok (eğitimde yalnız yatay çevirme).
- Fixed-4k = 2.000 SD1.5 + 2.000 gerçek; Fixed-8k = 2.000 SD1.5 + 2.000 Wukong + 4.000 gerçek (B-8k ile aynı görseller). Karşılaştırma 5 görülmeyen üreticide (Wukong 8k'da görülen).

### Kestirme testi (yalnız dosya özellikleri, Random Forest; %50 = kestirme yok)
| Generator | Unbiased (önce) | Fixed (sonra) |
|---|---|---|
| SD1.5 | 100,0 | 56,2 |
| ADM | 50,0 | 54,7 |
| BigGAN | 50,0 | 56,6 |
| GLIDE | 50,0 | 57,5 |
| Midjourney | 100,0 | 56,5 |
| VQDM | 50,0 | 53,8 |
| Wukong | 100,0 | 52,9 |
Boyut kestirmesi büyük ölçüde giderildi; kalan %53–57 muhtemelen içerikten kaynaklanan piksel başına bayt farkı.

### Mean (5 unseen), %, 3 seeds — düzeltilmiş veri, Base modeller
| Data | Metric | CLIP ViT-L/14 + LR | ResNet-50 | ViT-B/16 |
|---|---|---|---|---|
| Fixed-4k | Accuracy | 86,6 ± 0,2 | 83,2 ± 0,4 | 80,7 ± 0,7 |
| Fixed-8k | Accuracy | 88,4 ± 0,2 | 83,9 ± 0,9 | 78,8 ± 2,9 |
| Fixed-4k | ROC-AUC | 94,8 ± 0,1 | 91,5 ± 0,4 | 92,3 ± 0,1 |
| Fixed-8k | ROC-AUC | 95,8 ± 0,1 | 93,2 ± 0,3 | 91,4 ± 0,9 |
| Fixed-4k | AP | 94,4 ± 0,2 | 89,7 ± 0,7 | 91,9 ± 0,4 |
| Fixed-8k | AP | 95,4 ± 0,1 | 92,0 ± 0,2 | 89,9 ± 1,1 |
| Fixed-4k | Fake acc. | 83,7 ± 0,3 | 79,6 ± 0,6 | 66,7 ± 2,0 |
| Fixed-8k | Fake acc. | 85,0 ± 0,2 | 78,2 ± 2,7 | 63,2 ± 6,1 |
| Fixed-4k | Real acc. | 89,6 ± 0,2 | 86,8 ± 0,3 | 94,7 ± 1,1 |
| Fixed-8k | Real acc. | 91,7 ± 0,2 | 89,6 ± 0,9 | 94,4 ± 0,3 |
Fixed-4k, Mean (6 unseen) Accuracy: CLIP 87,8 ± 0,2 · ResNet-50 84,2 ± 0,3 · ViT-B/16 82,6 ± 0,7 (düzeltmeden önce, Deney A N=4000: 82,6 / 87,5 / 78,8).

Düzeltmeden önce ve sonra, Mean (5 unseen) Accuracy: 4k → CLIP 80,9 → 86,6 · ResNet 85,8 → 83,2 · ViT 75,6 → 80,7; 8k → CLIP 84,2 → 88,4 · ResNet 81,4 → 83,9 · ViT 74,1 → 78,8.

### Per-generator Accuracy (%), mean of 3 seeds — Fixed-4k / Fixed-8k
| Generator | CLIP + LR | ResNet-50 | ViT-B/16 |
|---|---|---|---|
| SD1.5 | 93,3 / 94,8 | 91,1 / 94,2 | 95,9 / 96,1 |
| Wukong | 93,3 / 95,2 | 89,0 / 94,0 | 92,2 / 96,2 |
| ADM | 82,0 / 84,1 | 71,9 / 77,1 | 70,5 / 69,6 |
| BigGAN | 84,8 / 86,1 | 89,0 / 81,6 | 85,8 / 80,0 |
| GLIDE | 88,1 / 89,0 | 90,2 / 87,7 | 86,6 / 82,0 |
| Midjourney | 92,2 / 93,2 | 86,7 / 90,0 | 85,9 / 86,8 |
| VQDM | 86,1 / 89,3 | 78,2 / 83,1 | 74,9 / 75,4 |

### Kaynak kullanımı (seed ortalaması)
| Model | Data | Süre (sn) | GPU tepe (GB) | RAM süreç (GB) | RAM sistem (GB) |
|---|---|---|---|---|---|
| CLIP ViT-L/14 | öznitelik çıkarma (12.000 görsel) | 141 | 1,44 | 4,27 | 4,69 |
| ResNet-50 | Fixed-4k | 55 | 1,79 | 3,92 | 4,88 |
| ResNet-50 | Fixed-8k | 107 | 1,79 | 3,93 | 4,96 |
| ViT-B/16 | Fixed-4k | 109 | 3,44 | 3,97 | 4,90 |
| ViT-B/16 | Fixed-8k | 214 | 3,44 | 3,96 | 4,88 |

### Gözlemler
- Düzeltme CLIP'i (+4–6 puan) ve ViT'i (+5 puan) belirgin iyileştirdi; CLIP artık en iyi model (5 unseen 86,6 / 88,4; ADM 73,8 → 82,0, BigGAN 75,3 → 84,8, VQDM 78,6 → 86,1).
- ResNet-50 4k'da düştü (85,8 → 83,2), özellikle BigGAN (94,7 → 89,0): ResNet'in önceki başarısının bir kısmı yeniden boyutlandırma izlerine dayanıyor olabilir. 8k'da düzeltme ResNet'i iyileştirdi (81,4 → 83,9).
- Düzeltilmiş veride iki katı veri CLIP'e +1,8, ResNet'e +0,7 kazandırdı; ViT'i yine düşürdü (80,7 → 78,8, std 2,9).

## Yerel veri setlerinin düzeltilmesi (21.11.2026)
Betik: `deneyler/veri_duzelt.py` (Hücre 14 ile aynı işlem: 224×224 merkezden kırpma, kısa kenarı 224'ten küçükse önce bicubic büyütme, JPEG kalite 96). Orijinal klasörler değişmedi.
| Kaynak (Masaüstü) | Görsel | Çıktı | Boyut | Büyütülen |
|---|---|---|---|---|
| Unbiased Tiny GenImage (2,4 GB) | 23.328 | `Unbiased Tiny GenImage (224 JPEG96)` | 594 MB | 2.500 (BigGAN) |
| Tiny GenImage (7,8 GB; ai PNG, nature JPEG) | 35.000 | `Tiny GenImage (224 JPEG96)` | 929 MB | 3.631 (BigGAN 2.500 + kısa kenarı <224 olan 1.131 gerçek görsel) |
Doğrulama: çıktıdaki tüm görseller JPEG, 224×224. Tiny GenImage'ın gerçek görsellerinde 224'ten küçük ImageNet görselleri var (üretici başına ~160); Unbiased sürümde yok (gerçekler ~500 px seçilmiş).

## Danışmanın 2×2 deneyi — adım 2: büyük modeller (Hücre 15, 23.11.2026)
Büyük modeller: ResNet-101 (CNN), ViT-L/16 (ViT; batch 16 × gradient accumulation 2), CLIP ViT-H/14 + LR (ağırlıklar LAION-2B, OpenAI değil). Veri ve eğitim ayarları Hücre 14 ile aynı (düzeltilmiş 224 JPEG96, 3 epoch, 3 seed). Large toplam süre 79,6 dk.

### Mean (5 unseen) Accuracy (%), 3 seeds — tam 2×2
| Data | CLIP Base (L/14) | CLIP Large (H/14) | CNN Base (R-50) | CNN Large (R-101) | ViT Base (B/16) | ViT Large (L/16) |
|---|---|---|---|---|---|---|
| Fixed-4k (SD1.5) | 86,6 ± 0,2 | **89,2 ± 0,1** | 83,2 ± 0,4 | 83,8 ± 0,2 | 80,7 ± 0,7 | 81,2 ± 1,3 |
| Fixed-8k (SD1.5+Wukong) | 88,4 ± 0,2 | **90,3 ± 0,1** | 83,9 ± 0,9 | 85,2 ± 0,1 | 78,8 ± 2,9 | 80,9 ± 1,0 |

### Diğer metrikler (5 unseen, 3 seed ortalaması; Base / Large)
| Data | Metrik | CLIP | CNN | ViT |
|---|---|---|---|---|
| Fixed-4k | ROC-AUC | 94,8 / 97,7 | 91,5 / 91,5 | 92,3 / 90,8 |
| Fixed-8k | ROC-AUC | 95,8 / 97,9 | 93,2 / 93,9 | 91,4 / 90,7 |
| Fixed-4k | AP | 94,4 / 97,2 | 89,7 / 89,4 | 91,9 / 89,9 |
| Fixed-8k | AP | 95,4 / 97,5 | 92,0 / 93,0 | 89,9 / 88,9 |
| Fixed-4k | Fake acc. | 83,7 / 83,2 | 79,6 / 83,8 | 66,7 / 69,5 |
| Fixed-8k | Fake acc. | 85,0 / 85,2 | 78,2 / 80,7 | 63,2 / 72,4 |
| Fixed-4k | Real acc. | 89,6 / 95,3 | 86,8 / 83,9 | 94,7 / 92,8 |
| Fixed-8k | Real acc. | 91,7 / 95,5 | 89,6 / 89,7 | 94,4 / 89,4 |

Fixed-4k, Mean (6 unseen) Accuracy: CLIP 87,8 → 90,2; CNN 84,2 → 84,7; ViT 82,6 → 83,1 (Base → Large).

### Üretici bazında Accuracy (%), 3 seed ortalaması (Base / Large)
| Üretici | 4k CLIP | 4k CNN | 4k ViT | 8k CLIP | 8k CNN | 8k ViT |
|---|---|---|---|---|---|---|
| SD1.5 | 93,3 / 97,1 | 91,1 / 90,0 | 95,9 / 95,1 | 94,8 / 97,3 | 94,2 / 93,9 | 96,1 / 94,1 |
| Wukong | 93,3 / 95,2 | 89,0 / 88,8 | 92,2 / 93,1 | 95,2 / 97,3 | 94,0 / 94,0 | 96,2 / 94,3 |
| ADM | 82,0 / 93,1 | 71,9 / 74,6 | 70,5 / 71,1 | 84,1 / 94,6 | 77,1 / 76,8 | 69,6 / 71,3 |
| BigGAN | 84,8 / 73,5 | 89,0 / 88,6 | 85,8 / 83,9 | 86,1 / 76,6 | 81,6 / 86,9 | 80,0 / 83,7 |
| GLIDE | 88,1 / 92,1 | 90,2 / 89,0 | 86,6 / 85,9 | 89,0 / 91,8 | 87,7 / 89,5 | 82,0 / 83,4 |
| Midjourney | 92,2 / 94,6 | 86,7 / 87,2 | 85,9 / 88,6 | 93,2 / 95,1 | 90,0 / 89,3 | 86,8 / 88,6 |
| VQDM | 86,1 / 92,9 | 78,2 / 79,8 | 74,9 / 76,4 | 89,3 / 93,7 | 83,1 / 83,4 | 75,4 / 77,7 |

### Kaynak kullanımı (seed ortalaması)
| Model | Data | Süre (sn) | GPU tepe (GB) | RAM süreç (GB) | RAM sistem (GB) |
|---|---|---|---|---|---|
| CLIP ViT-H/14 | öznitelik çıkarma | 267 | 2,52 | 5,07 | 5,55 |
| ResNet-101 | Fixed-4k | 88 | 2,72 | 4,81 | 5,69 |
| ResNet-101 | Fixed-8k | 171 | 2,72 | 5,26 | 6,27 |
| ViT-L/16 | Fixed-4k | 357 | 8,35 | 5,32 | 5,87 |
| ViT-L/16 | Fixed-8k | 699 | 8,35 | 5,31 | 5,91 |

### Gözlemler
- En iyi sonuç: CLIP ViT-H/14 + Fixed-8k, 90,3 ± 0,1. Büyük model kazancı (+2,6 / +1,9) veri iki katına çıkarmanın kazancından (+1,8 / +1,1) biraz fazla; ikisi birlikte +3,7.
- CLIP-H'nin kazancı ADM (82 → 93) ve VQDM'de (86 → 93) büyük; BigGAN'da ise düşüş var (84,8 → 73,5): büyütülmüş 128 px görseller CLIP-H'yi şaşırtıyor olabilir.
- ResNet-101 küçük ama tutarlı kazanç (4k +0,6, 8k +1,3). ViT-L ViT-B'den az farkla iyi (+0,5 / +2,1) ve 8k'daki düşüşü azaltıyor, ama ViT ailesi hâlâ en zayıf; Fake acc. düşük (≈63–72), gerçeğe yatkın.
- Maliyet: ViT-L 8,35 GB GPU ve ViT-B'nin ~3,3 katı süre; CLIP-H tek seferlik öznitelik çıkarma 267 sn. Hepsi T4'e (15 GB) sığıyor.
