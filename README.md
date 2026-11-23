# AI Image Detection — Generalization

Yapay zekâ ile üretilmiş görsellerin tespiti: bir üreticide (Stable Diffusion 1.5) eğitilen modellerin görülmemiş üreticilere (ADM, BigGAN, GLIDE, Midjourney, VQDM, Wukong) genellenmesi.

**Modeller:** ResNet-50 / ResNet-101, ViT-B/16 / ViT-L/16 (ince ayar), dondurulmuş CLIP ViT-L/14 / ViT-H/14 + Logistic Regression.
**Ortam:** Kaggle, Tesla T4 (15 GB).

## Veri setleri
| Veri seti | Kaynak | Görsel | Boyut | Format (yapay / gerçek) |
|---|---|---|---|---|
| Tiny GenImage | [yangsangtai/tiny-genimage](https://www.kaggle.com/datasets/yangsangtai/tiny-genimage) | 35.000 | 7,8 GB | PNG / JPEG |
| Unbiased Tiny GenImage | [cartografia/unbiased-tiny-genimage](https://www.kaggle.com/datasets/cartografia/unbiased-tiny-genimage) | 23.328 | 2,4 GB | JPEG / JPEG |

Her iki set de GenImage'ın küçük sürümüdür: 7 üretici (SD 1.5, ADM, BigGAN, GLIDE, Midjourney, VQDM, Wukong) ve ImageNet tabanlı gerçek görseller (Nature).

Özgün görsel boyutları üreticiye göre farklı:

| Üretici | Çözünürlük |
|---|---|
| BigGAN | 128×128 |
| ADM, GLIDE, VQDM | 256×256 |
| SD 1.5, Wukong | 512×512 |
| Midjourney | 1024×1024 |
| Gerçek (ImageNet) | değişken, ~500 px |

**Eğitim / test ayrımı**
- Eğitim 4k: 2.000 SD 1.5 yapay + 2.000 gerçek (%10 validation)
- Eğitim 8k: 2.000 SD 1.5 + 2.000 Wukong yapay + 4.000 gerçek
- Test: her üretici için 500 yapay + ortak 500 gerçek (ImageNet val adlı); eğitimle çakışma yok

## Veri düzeltme (aynı boyut ve format)
**Sorun:** Modelin görselin içeriğine değil dosya özelliklerine bakarak karar verebilmesi (kestirme / shortcut).
- Tiny GenImage'da yapaylar PNG, gerçekler JPEG: yalnız format bile sınıfı ele veriyor.
- Unbiased sürümde ikisi de JPEG, ama boyut farkı kalıyor (SD 1.5 512 px, gerçekler ~500 px).

Bunu ölçmek için yalnız dosya özellikleriyle (genişlik, yükseklik, en/boy oranı, piksel başına bayt, PNG mi, JPEG kuantizasyonu) bir Random Forest eğitildi:

| Veri | SD 1.5 | ADM | BigGAN | GLIDE | Midjourney | VQDM | Wukong |
|---|---|---|---|---|---|---|---|
| Tiny GenImage (özgün) | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| Unbiased (özgün) | 100 | 50 | 50 | 50 | 100 | 50 | 100 |
| **Düzeltilmiş** | 56 | 55 | 57 | 58 | 57 | 54 | 53 |

Accuracy (%); %50 = kestirme yok. Özgün sette Random Forest "512 px ve üzeri ise yapay" kuralını öğreniyor.

**Düzeltme:** Tüm görseller (gerçek ve yapay, eğitim ve test) aynı biçime getirildi:
1. RGB'ye çevir
2. Kısa kenarı 224'ten küçükse (BigGAN 128 px, bazı ImageNet görselleri) bicubic ile 224'e büyüt
3. Özgün çözünürlükte 224×224 merkezden kırp (yeniden boyutlandırma yok)
4. JPEG kalite 96 ile kaydet

Düzeltmeden sonra kestirme testi %53–57'ye düştü. Eğitimde yeniden boyutlandırma yapılmıyor; yalnız yatay çevirme kullanılıyor.

| Düzeltilmiş veri seti | Görsel | Boyut | Büyütülen |
|---|---|---|---|
| Tiny GenImage (224 JPEG96) | 35.000 | 929 MB | 3.631 (BigGAN 2.500 + küçük 1.131 gerçek) |
| Unbiased Tiny GenImage (224 JPEG96) | 23.328 | 594 MB | 2.500 (BigGAN) |

Tüm çıktılar JPEG ve 224×224. Kaggle'da aynı işlem `kaggle/hucre14_2x2_temel_modeller.py` içinde, yerelde `scripts/veri_duzelt.py` ile yapılır:

```bash
python scripts/veri_duzelt.py "<kaynak klasör>" "<hedef klasör>"
```

Özgün klasörlere dokunulmaz; çıktı aynı klasör yapısıyla yeni klasöre yazılır.

## Klasörler
| Klasör | İçerik |
|---|---|
| `kaggle/` | Kaggle not defteri hücreleri (`hucreN_*.py`) |
| `scripts/veri_duzelt.py` | Veri setlerini 224×224 JPEG (kalite 96) formatına getirir |
| `results/SONUCLAR.md` | Tüm deney sonuçları (Accuracy, ROC-AUC, AP, kaynak kullanımı) |

## Hücre sırası
Hücreler aynı kernel oturumunda sırayla çalışmalıdır; oturum yeniden başlarsa önce `hucre0_kurulum.py`.
- `hucre1`–`hucre9`: pilot (veri indeksi, CLIP, ince ayar, 3 seed, özet)
- `hucre10`: metadata kestirme testi (Random Forest)
- `hucre11`: format yanlılığı mini deneyi
- `hucre12`: Deney A — veri miktarı (1k / 2k / 4k)
- `hucre13`: Deney B — üretici çeşitliliği (SD 1.5 + Wukong, 8k)
- `hucre14`: 2×2 deney, düzeltilmiş veri + temel modeller (4k ve 8k)
- `hucre15`: 2×2 deney, düzeltilmiş veri + büyük modeller (4k ve 8k)

## Özet sonuç (düzeltilmiş veri, 5 görülmeyen üretici, Mean Accuracy %, 3 seed)
| Veri | CLIP L/14 → H/14 | ResNet-50 → 101 | ViT-B/16 → L/16 |
|---|---|---|---|
| 4k (SD 1.5) | 86,6 → 89,2 | 83,2 → 83,8 | 80,7 → 81,2 |
| 8k (SD 1.5 + Wukong) | 88,4 → 90,3 | 83,9 → 85,2 | 78,8 → 80,9 |

Wukong 8k eğitiminde görüldüğü için karşılaştırma 5 görülmeyen üreticide (ADM, BigGAN, GLIDE, Midjourney, VQDM) yapılmıştır. CLIP ViT-H/14 ağırlıkları LAION-2B üzerinde eğitilmiştir.
