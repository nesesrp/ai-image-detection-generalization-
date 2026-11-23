# Hücre 15: 2×2 deneyi — 2. adım: düzeltilmiş veri, büyük modeller (4k ve 8k)
#   (2) Düzeltilmiş veri, büyük model    (4) İki katı (düzeltilmiş) veri, büyük model
# Hücre 14 aynı oturumda çalışmış olmalı (run_grid, summarize, GRID, TRAIN_FIX, TEST_FIX).
# Her aile bir boy büyür: ResNet-50 → ResNet-101, ViT-B/16 → ViT-L/16, CLIP ViT-L/14 (OpenAI) → CLIP ViT-H/14 (LAION-2B).
# ViT-L/16 belleğe sığsın diye batch 16 × 2 adım biriktirme = etkin batch 32; diğer ayarlar Base ile aynı.
_CIP = CLIPImageProcessor
class CLIPImageProcessor:  # LAION deposunda işlemci dosyası yoksa OpenAI işlemcisine düş (aynı 224 px ve normalizasyon)
    @staticmethod
    def from_pretrained(name):
        try:
            return _CIP.from_pretrained(name)
        except Exception as e:
            print(f"{name} işlemcisi yüklenemedi ({type(e).__name__}); openai/clip-vit-large-patch14 işlemcisi kullanılıyor")
            return _CIP.from_pretrained("openai/clip-vit-large-patch14")

LARGE = {"CNN": ("ResNet-101", ["resnet101.a1_in1k", "resnet101.a1h_in1k", "resnet101.tv2_in1k"], 32, 1),
         "ViT": ("ViT-L/16", "vit_large_patch16_224.augreg_in21k_ft_in1k", 16, 2),
         "CLIP": ("CLIP ViT-H/14", "laion/CLIP-ViT-H-14-laion2B-s32B-b79K")}
run_grid("Large", LARGE)
summarize()
