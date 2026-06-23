/**
 * Heuristic detection for Cloudflare / CAPTCHA / blank error screenshots.
 */

import sharp from "sharp";

export async function looksLikeBlockedScreenshot(buffer) {
  try {
    const { data, info } = await sharp(buffer)
      .resize(120, 63)
      .raw()
      .toBuffer({ resolveWithObject: true });

    const width = info.width;
    const height = info.height;
    const total = width * height;
    let lightPixels = 0;
    let edgePixels = 0;
    const colors = new Set();

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const index = (y * width + x) * 3;
        const red = data[index];
        const green = data[index + 1];
        const blue = data[index + 2];
        const luminance = 0.299 * red + 0.587 * green + 0.114 * blue;

        if (luminance > 230) {
          lightPixels++;
        }

        colors.add(`${red >> 4},${green >> 4},${blue >> 4}`);

        if (x < width - 1) {
          const next = (y * width + x + 1) * 3;
          const delta =
            Math.abs(red - data[next]) +
            Math.abs(green - data[next + 1]) +
            Math.abs(blue - data[next + 2]);
          if (delta > 60) {
            edgePixels++;
          }
        }
      }
    }

    const lightRatio = lightPixels / total;
    const edgeRatio = edgePixels / ((width - 1) * height);
    const colorCount = colors.size;

    // Near-blank interstitial (e.g. CHED Cloudflare).
    if (lightRatio > 0.78 && edgeRatio < 0.025) {
      return true;
    }

    // Sparse light challenge page (e.g. Reddit block).
    if (lightRatio > 0.81 && edgeRatio < 0.09 && colorCount < 125) {
      return true;
    }

    return false;
  } catch {
    return false;
  }
}
