#!/usr/bin/env node
/**
 * Copy freshie-recommendation cover images into public/freshie-recommendations/
 * for Discord / messenger sharing.
 *
 *   npm run freshie:images
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { FRESHIE_RECOMMENDATIONS } from "../src/data/freshie-recommendations.js";
import {
  getOpportunities,
  resolveOpportunityImage,
} from "../src/data/opportunities.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(__dirname, "..");
const OUT_DIR = path.join(ROOT, "public/freshie-recommendations");

function extensionFor(publicPath) {
  const ext = path.extname(publicPath).toLowerCase();
  return ext || ".png";
}

function main() {
  const items = getOpportunities();
  const byTitle = new Map(items.map((item) => [item.title, item]));
  const manifest = [];

  fs.mkdirSync(OUT_DIR, { recursive: true });

  for (const entry of FRESHIE_RECOMMENDATIONS) {
    const item = byTitle.get(entry.boardTitle);
    if (!item) {
      throw new Error(`Board item not found: ${entry.boardTitle}`);
    }

    const sourcePublicPath = resolveOpportunityImage(item);
    const sourcePath = path.join(ROOT, "public", sourcePublicPath.replace(/^\//, ""));
    if (!fs.existsSync(sourcePath)) {
      throw new Error(`Image missing for ${entry.boardTitle}: ${sourcePath}`);
    }

    const filename = `${entry.slug}${extensionFor(sourcePublicPath)}`;
    const destPath = path.join(OUT_DIR, filename);
    fs.copyFileSync(sourcePath, destPath);

    manifest.push({
      order: Number.parseInt(entry.slug.slice(0, 2), 10),
      slug: entry.slug,
      title: item.title,
      url: item.url,
      image: `/freshie-recommendations/${filename}`,
      sourceImage: sourcePublicPath,
    });
  }

  fs.writeFileSync(
    path.join(OUT_DIR, "manifest.json"),
    `${JSON.stringify(manifest, null, 2)}\n`,
  );

  console.log(`Exported ${manifest.length} images to public/freshie-recommendations/`);
  for (const row of manifest) {
    console.log(`  ${row.image}`);
  }
}

main();
