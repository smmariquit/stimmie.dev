#!/usr/bin/env node
/**
 * Copy community-highlight cover images into public/community-highlights/
 *
 *   npm run community:images
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { COMMUNITY_HIGHLIGHTS } from "../src/data/community-highlights.js";
import {
  getOpportunities,
  resolveOpportunityImage,
} from "../src/data/opportunities.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(__dirname, "..");
const OUT_DIR = path.join(ROOT, "public/community-highlights");

function extensionFor(publicPath) {
  const ext = path.extname(publicPath).toLowerCase();
  return ext || ".png";
}

function resolveBoardItem(byTitle, entry) {
  const lookupTitle = entry.imageBoardTitle ?? entry.boardTitle;
  const item = byTitle.get(lookupTitle);
  if (!item) {
    throw new Error(`Board item not found: ${lookupTitle}`);
  }
  return item;
}

function main() {
  const items = getOpportunities();
  const byTitle = new Map(items.map((item) => [item.title, item]));
  const manifest = [];
  const exportedFiles = new Set();

  fs.mkdirSync(OUT_DIR, { recursive: true });

  for (const entry of COMMUNITY_HIGHLIGHTS) {
    const item = resolveBoardItem(byTitle, entry);
    const sourcePublicPath = resolveOpportunityImage(item);
    const sourcePath = path.join(ROOT, "public", sourcePublicPath.replace(/^\//, ""));
    if (!fs.existsSync(sourcePath)) {
      throw new Error(`Image missing for ${entry.imageBoardTitle ?? entry.boardTitle}: ${sourcePath}`);
    }

    const filename = `${entry.slug}${extensionFor(sourcePublicPath)}`;
    const destPath = path.join(OUT_DIR, filename);
    fs.copyFileSync(sourcePath, destPath);
    exportedFiles.add(filename);

    manifest.push({
      order: Number.parseInt(entry.slug.slice(0, 2), 10),
      slug: entry.slug,
      title: entry.title,
      url: entry.url,
      blurb: entry.blurb,
      image: `/community-highlights/${filename}`,
      sourceImage: sourcePublicPath,
      ...(entry.boardTitle ? { boardTitle: entry.boardTitle } : {}),
      ...(entry.includes ? { includes: entry.includes } : {}),
    });
  }

  for (const file of fs.readdirSync(OUT_DIR)) {
    if (file === "manifest.json") {
      continue;
    }
    if (!exportedFiles.has(file)) {
      fs.unlinkSync(path.join(OUT_DIR, file));
    }
  }

  fs.writeFileSync(
    path.join(OUT_DIR, "manifest.json"),
    `${JSON.stringify(manifest, null, 2)}\n`,
  );

  console.log(`Exported ${manifest.length} images to public/community-highlights/`);
  for (const row of manifest) {
    console.log(`  ${row.image}`);
  }
}

main();
