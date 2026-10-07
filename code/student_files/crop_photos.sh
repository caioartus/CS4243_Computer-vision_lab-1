#!/usr/bin/env bash
# [AI-CODE]
# Centre-crop personal photos to a square, resize to SIZE x SIZE and strip EXIF.
# Only processes photos that are not already in the clean folder.
# Usage: ./crop_photos.sh [SIZE]   (default 600)
set -euo pipefail

SIZE="${1:-600}"
DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="$DIR/personnal_photos"
DST="$DIR/personnal_photos_clean"
mkdir -p "$DST"

python3 - "$SRC" "$DST" "$SIZE" <<'EOF'
import sys
from pathlib import Path

from PIL import Image, ImageOps

src, dst, size = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
extensions = {".jpg", ".jpeg", ".png", ".heic", ".webp"}

done = skipped = 0
for path in sorted(src.iterdir()):
    if path.suffix.lower() not in extensions:
        continue
    out = dst / path.name
    if out.exists():
        skipped += 1
        continue
    with Image.open(path) as img:
        # Apply the EXIF rotation before EXIF is dropped, so photos stay upright.
        img = ImageOps.exif_transpose(img).convert("RGB")
        side = min(img.size)
        left, top = (img.width - side) // 2, (img.height - side) // 2
        img = img.crop((left, top, left + side, top + side))
        img = img.resize((size, size), Image.Resampling.LANCZOS)
        # Saving without the exif argument writes no EXIF metadata.
        img.save(out, quality=95)
    print(f"cropped {path.name}")
    done += 1

print(f"{done} new photo(s) cropped, {skipped} already in {dst.name}")
EOF
