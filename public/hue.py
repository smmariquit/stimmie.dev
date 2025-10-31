from pathlib import Path
import json
import logging
import numpy as np
from skimage import io, color


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def circular_mean_hue(hues, weights=None):
    """Compute circular mean for hues in [0,1].

    hues: 1D array-like of values in [0,1]
    weights: optional weights (same shape) to compute weighted circular mean
    """
    angles = np.asarray(hues, dtype=np.float64) * (2.0 * np.pi)
    if weights is None:
        sin_mean = np.mean(np.sin(angles))
        cos_mean = np.mean(np.cos(angles))
    else:
        w = np.asarray(weights, dtype=np.float64)
        wsum = np.sum(w)
        if wsum == 0:
            return float('nan')
        sin_mean = np.sum(np.sin(angles) * w) / wsum
        cos_mean = np.sum(np.cos(angles) * w) / wsum
    mean_angle = np.arctan2(sin_mean, cos_mean)
    return float((mean_angle / (2.0 * np.pi)) % 1.0)


def process_images(images_dir: Path, out_json: Path, sample_limit: int = 200000):
    """Process images in `images_dir`, compute mean RGB and mean hue, and save summary to out_json.

    This function is robust to errors: it logs and continues on exceptions.
    For large images we sample pixels up to `sample_limit` to avoid excessive memory use.
    """
    results = {}
    if not images_dir.exists():
        logging.error("Images directory does not exist: %s", images_dir)
        return

    for file_path in sorted(images_dir.iterdir()):
        if not file_path.is_file():
            continue
        try:
            logging.info("Reading %s", file_path.name)
            img = io.imread(file_path)

            # Ensure image has 3 color channels (RGB)
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)

            if img.ndim == 3 and img.shape[-1] == 4:
                # RGBA -> composite over white background
                rgba = img.astype('float32') / 255.0
                rgb = rgba[..., :3]
                alpha = rgba[..., 3:4]
                img_f = rgb * alpha + 1.0 * (1.0 - alpha)
            else:
                rgb = img[..., :3]
                if np.issubdtype(rgb.dtype, np.integer):
                    img_f = rgb.astype('float32') / np.iinfo(rgb.dtype).max
                else:
                    img_f = np.clip(rgb.astype('float32'), 0.0, 1.0)

            # Basic summary: mean RGB
            mean_rgb = img_f.mean(axis=(0, 1)).tolist()

            # To avoid huge memory use, sample pixels for hue computation
            pixels = img_f.reshape(-1, 3)
            n_pixels = pixels.shape[0]
            if n_pixels > sample_limit:
                idx = np.random.default_rng(0).choice(n_pixels, sample_limit, replace=False)
                sample = pixels[idx]
            else:
                sample = pixels

            # rgb2hsv accepts (H,W,3); reshape sample to (N,1,3) to convert many colors cheaply
            try:
                hsv_sample = color.rgb2hsv(sample.reshape(-1, 1, 3))
                hue_vals = hsv_sample[:, 0, 0]
                sat_vals = hsv_sample[:, 0, 1]
            except MemoryError:
                logging.warning("MemoryError converting to HSV for %s — trying smaller sample", file_path.name)
                small_n = min(50000, sample.shape[0])
                idx2 = np.random.default_rng(1).choice(sample.shape[0], small_n, replace=False)
                hsv_sample = color.rgb2hsv(sample[idx2].reshape(-1, 1, 3))
                hue_vals = hsv_sample[:, 0, 0]
                sat_vals = hsv_sample[:, 0, 1]

            mean_hue = circular_mean_hue(hue_vals, weights=(sat_vals + 1e-8))

            results[file_path.stem] = {
                "file": file_path.name,
                "mean_rgb": mean_rgb,
                "mean_hue": mean_hue,
                "pixels_sampled": int(min(n_pixels, sample_limit)),
            }

            logging.info("Processed %s: mean_rgb=%s mean_hue=%.4f", file_path.name, mean_rgb, mean_hue)

        except Exception as exc:
            # Catch everything so processing continues for remaining files
            logging.exception("Failed to process %s: %s", file_path.name, exc)

    # Save results (small JSON summary)
    try:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with out_json.open('w', encoding='utf-8') as fh:
            json.dump(results, fh, indent=2)
        logging.info("Wrote summary to %s", out_json)
    except Exception:
        logging.exception("Failed to write output JSON: %s", out_json)


if __name__ == '__main__':
    base = Path(__file__).parent
    images_dir = base / 'images'
    out_json = base / 'image_data_summary.json'
    process_images(images_dir, out_json)