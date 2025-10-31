#!/usr/bin/env python3
"""generate_thumbs.py

Create small, square, compressed thumbnails from images in the public/images
folder and write them to public/images_small (or a folder you pass).

Usage:
  python generate_thumbs.py --in public/images --out public/images_small --size 320 --format webp

Requires: Pillow (pip install pillow)
"""
from pathlib import Path
from PIL import Image
import argparse
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def process_image(src_path: Path, dst_path: Path, size: int, fmt: str, quality: int, force: bool):
    try:
        if dst_path.exists() and not force:
            logging.debug('Skipping existing %s', dst_path)
            return True

        with Image.open(src_path) as im:
            # Convert to RGB (drops alpha) for JPEG; WebP can keep alpha but we'll flatten to white
            if im.mode not in ("RGB", "RGBA"):
                im = im.convert("RGBA")

            w, h = im.size
            min_side = min(w, h)
            left = (w - min_side) // 2
            top = (h - min_side) // 2
            right = left + min_side
            bottom = top + min_side
            im_cropped = im.crop((left, top, right, bottom))

            # Resize with high-quality resampling
            im_resized = im_cropped.resize((size, size), resample=Image.LANCZOS)

            # Prepare save kwargs
            save_kwargs = {}
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            if fmt.lower() == 'webp':
                out_path = dst_path.with_suffix('.webp')
                save_kwargs['format'] = 'WEBP'
                save_kwargs['quality'] = quality
                # keep alpha if present; otherwise convert to RGB
                if im_resized.mode == 'RGBA':
                    im_to_save = im_resized
                else:
                    im_to_save = im_resized.convert('RGB')
            else:
                out_path = dst_path.with_suffix('.jpg')
                save_kwargs['format'] = 'JPEG'
                save_kwargs['quality'] = quality
                save_kwargs['optimize'] = True
                # JPEG doesn't support alpha; flatten over white
                if im_resized.mode == 'RGBA':
                    background = Image.new('RGB', im_resized.size, (255, 255, 255))
                    background.paste(im_resized, mask=im_resized.split()[3])
                    im_to_save = background
                else:
                    im_to_save = im_resized.convert('RGB')

            im_to_save.save(out_path, **save_kwargs)
            logging.info('Wrote %s', out_path)
            return True
    except Exception as e:
        logging.exception('Failed to process %s: %s', src_path, e)
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(description='Generate square thumbnails for images')
    parser.add_argument('--in', dest='in_dir', default='images', help='Input images directory (relative to this script)')
    parser.add_argument('--out', dest='out_dir', default='images_small', help='Output directory to place thumbnails')
    parser.add_argument('--size', type=int, default=320, help='Output size (pixels, square). Default 320')
    parser.add_argument('--format', choices=['webp', 'jpeg'], default='webp', help='Output image format')
    parser.add_argument('--quality', type=int, default=80, help='Quality for compressed image (0-100)')
    parser.add_argument('--force', action='store_true', help='Overwrite existing thumbnails')
    args = parser.parse_args(argv)

    base = Path(__file__).parent
    in_dir = (base / args.in_dir).resolve()
    out_dir = (base / args.out_dir).resolve()

    if not in_dir.exists():
        logging.error('Input directory does not exist: %s', in_dir)
        return 2

    exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}
    total = 0
    success = 0

    for p in sorted(in_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            logging.debug('Skipping non-image: %s', p.name)
            continue

        dst = out_dir / p.name
        total += 1
        ok = process_image(p, dst, args.size, args.format, args.quality, args.force)
        if ok:
            success += 1

    logging.info('Done: %d/%d thumbnails created', success, total)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
