import imageData from "../../public/archive/mosaic/image_data_summary.json";
import {
  archiveMosaicImage,
  archiveMosaicThumb,
} from "@/lib/archive-paths";

/** Mosaic tiles for archived site versions (v1, v2). */
export const archiveMosaicImages = Object.values(imageData).map((it) => ({
  ...it,
  src: archiveMosaicImage(it.file),
  thumbSrc: archiveMosaicThumb(it.file),
}));

export const archiveMosaicImagesByHue = [...archiveMosaicImages].sort(
  (a, b) => a.mean_hue - b.mean_hue,
);
