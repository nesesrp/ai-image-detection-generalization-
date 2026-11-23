"""Yerel veri setlerini aynı boyut ve formata getirir (rapor Bölüm 3, ön işleme kararı; Kaggle Hücre 14 ile aynı işlem).

Her görsel: RGB'ye çevrilir -> kısa kenarı 224'ten küçükse (BigGAN, 128 px) bicubic ile 224'e büyütülür ->
özgün çözünürlükte 224×224 merkezden kırpılır -> JPEG kalite 96 ile kaydedilir.
Orijinal klasörlere dokunulmaz; çıktı aynı klasör yapısıyla yeni bir klasöre yazılır, dosya adları korunur (uzantı .jpg).

Kullanım: python veri_duzelt.py "<kaynak klasör>" "<hedef klasör>"
"""
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

from PIL import Image

SIZE, QUALITY = 224, 96
EXTS = {".jpg", ".jpeg", ".png"}


def fix_one(args):
    src, dst = args
    if os.path.exists(dst):
        return "skip", None
    try:
        im = Image.open(src).convert("RGB")
        w, h = im.size
        up = min(w, h) < SIZE
        if up:
            s = SIZE / min(w, h)
            im = im.resize((max(SIZE, round(w * s)), max(SIZE, round(h * s))), Image.BICUBIC)
            w, h = im.size
        l, t = (w - SIZE) // 2, (h - SIZE) // 2
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        im.crop((l, t, l + SIZE, t + SIZE)).save(dst, "JPEG", quality=QUALITY)
        return ("upscaled" if up else "ok"), None
    except Exception as e:  # bozuk dosya vb.
        return "error", f"{src}: {type(e).__name__}: {e}"


def main(src_root, dst_root):
    jobs = []
    for d, _, files in os.walk(src_root):
        for f in files:
            if os.path.splitext(f)[1].lower() in EXTS:
                rel = os.path.relpath(os.path.join(d, f), src_root)
                jobs.append((os.path.join(src_root, rel), os.path.join(dst_root, os.path.splitext(rel)[0] + ".jpg")))
    dst_names = [j[1] for j in jobs]
    assert len(dst_names) == len(set(dst_names)), "Aynı adlı iki dosya aynı hedefe yazılacaktı"
    print(f"{src_root}: {len(jobs)} görsel -> {dst_root}")
    stats, errors = Counter(), []
    with ProcessPoolExecutor() as ex:
        for i, (status, err) in enumerate(ex.map(fix_one, jobs, chunksize=64), 1):
            stats[status] += 1
            if err:
                errors.append(err)
            if i % 5000 == 0:
                print(f"  {i}/{len(jobs)}")
    print(f"  bitti: {dict(stats)}")
    for e in errors[:20]:
        print("  HATA", e)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
