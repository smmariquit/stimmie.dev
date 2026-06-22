#!/usr/bin/env node
/**
 * Copy assets required by /archive/v1 and /v2 into public/archive/.
 * Safe to re-run (idempotent). Run after changing live-site assets:
 *   npm run archive:snapshot
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const publicDir = path.join(root, "public");

function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true });
}

function copyFile(src, dest) {
  if (!fs.existsSync(src)) {
    console.warn(`  skip missing: ${path.relative(root, src)}`);
    return false;
  }
  ensureDir(path.dirname(dest));
  fs.copyFileSync(src, dest);
  return true;
}

function copyMany(files, destDir, srcResolver = (f) => f) {
  let copied = 0;
  for (const file of files) {
    const src = srcResolver(file);
    const name = path.basename(file);
    if (copyFile(src, path.join(destDir, name))) copied++;
  }
  return copied;
}

// ── Mosaic (shared by v1 + v2 backgrounds) ─────────────────────────────

function snapshotMosaic(manifest) {
  const mosaicDir = path.join(publicDir, "archive", "mosaic");
  const legacyJson = path.join(publicDir, "image_data_summary.json");
  const archiveJson = path.join(mosaicDir, "image_data_summary.json");
  const legacyImages = path.join(publicDir, "images");
  const archiveImages = path.join(mosaicDir, "images");

  ensureDir(mosaicDir);

  if (fs.existsSync(legacyJson) && !fs.existsSync(archiveJson)) {
    fs.renameSync(legacyJson, archiveJson);
    console.log("moved image_data_summary.json → archive/mosaic/");
  } else if (fs.existsSync(legacyJson) && fs.existsSync(archiveJson)) {
    console.log("mosaic json already in archive; leaving legacy copy");
  }

  if (fs.existsSync(legacyImages) && !fs.existsSync(archiveImages)) {
    fs.renameSync(legacyImages, archiveImages);
    console.log("moved images/ → archive/mosaic/images/");
  } else if (fs.existsSync(legacyImages) && fs.existsSync(archiveImages)) {
    console.log("mosaic images already in archive; leaving legacy copy");
  }

  if (!fs.existsSync(archiveJson)) {
    throw new Error("Missing public/archive/mosaic/image_data_summary.json");
  }

  manifest.push(archiveJson);
  const imageFiles = fs.readdirSync(archiveImages);
  for (const f of imageFiles) {
    manifest.push(path.join(archiveImages, f));
  }
  console.log(`mosaic: ${imageFiles.length} images`);
}

// ── v1 frozen assets ───────────────────────────────────────────────────

const V1_TALKS = [
  "talk1.jpg",
  "talk2.jpg",
  "talk3.jpg",
  "talk4.jpg",
  "talk5.jpg",
  "talk6.jpg",
  "talk7.gif",
];
const V1_PROJECTS = [
  "project1.jpg",
  "project2.jpg",
  "project3.jpg",
  "project4.jpg",
  "project5.jpg",
];
const V1_LOGOS = [
  "linkedin.png",
  "instagram.png",
  "github.png",
  "medium.png",
  "myanimelist.png",
  "pizza.png",
  "letterboxd.png",
  "goodreads.png",
  "spotify.png",
  "lastfm.png",
  "strava.png",
  "email.png",
];

function snapshotV1(manifest) {
  const base = path.join(publicDir, "archive", "v1");
  copyMany(V1_TALKS, path.join(base, "talks"), (f) =>
    path.join(publicDir, "talks", f),
  );
  copyMany(V1_PROJECTS, path.join(base, "projects"), (f) =>
    path.join(publicDir, "projects", f),
  );
  copyMany(V1_LOGOS, path.join(base, "logos"), (f) =>
    path.join(publicDir, "logos", f),
  );
  for (const sub of ["talks", "projects", "logos"]) {
    const dir = path.join(base, sub);
    if (fs.existsSync(dir)) {
      for (const f of fs.readdirSync(dir)) {
        manifest.push(path.join(dir, f));
      }
    }
  }
  console.log("v1 assets snapshotted");
}

// ── v2 frozen assets ───────────────────────────────────────────────────

const V2_TALKS = [
  "uprhs-day1-thumbnail.png",
  "uprhs-day2.jpg",
  "talk3.jpg",
  "jpcs-qcu-thumbnail.jpg",
  "talk5.jpg",
  "talk6.jpg",
  "dlsu-eces-agile-edge.jpg",
  "dsg-apps-workshop.jpeg",
  "python-in-research-thumbnail.png",
  "ml-workshop-smcl.jpg",
  "hearthcraft-thumbnail.png",
  "present-insights-better-thumbnail.png",
];
const V2_PROJECTS = [
  "project1.jpg",
  "project2.jpg",
  "project3.jpg",
  "project4.jpg",
  "project5.jpg",
];
const V2_LOGOS = [
  "linkedin.png",
  "email.png",
  "instagram.png",
  "facebook.svg",
  "discord.svg",
  "github.png",
  "leetcode.png",
  "kattis.png",
  "kaggle.svg",
  "letterboxd.png",
  "imdb.png",
  "myanimelist.png",
  "goodreads.png",
  "spotify.png",
  "lastfm.png",
  "medium.png",
  "pizza.png",
  "hackathon.png",
  "freshie.png",
  "strava.png",
  "osu.svg",
];

function snapshotV2(manifest) {
  const base = path.join(publicDir, "archive", "v2");
  copyMany(V2_TALKS, path.join(base, "talks"), (f) =>
    path.join(publicDir, "talks", f),
  );
  copyMany(V2_PROJECTS, path.join(base, "projects"), (f) =>
    path.join(publicDir, "projects", f),
  );
  copyMany(V2_LOGOS, path.join(base, "logos"), (f) =>
    path.join(publicDir, "logos", f),
  );
  for (const sub of ["talks", "projects", "logos"]) {
    const dir = path.join(base, sub);
    if (fs.existsSync(dir)) {
      for (const f of fs.readdirSync(dir)) {
        manifest.push(path.join(dir, f));
      }
    }
  }
  const screenshot = path.join(publicDir, "archive", "v2", "screenshot.png");
  if (fs.existsSync(screenshot)) {
    manifest.push(screenshot);
  }
  console.log("v2 assets snapshotted");
}

// ── Main ───────────────────────────────────────────────────────────────

const manifest = [];
snapshotMosaic(manifest);
snapshotV1(manifest);
snapshotV2(manifest);

const manifestOut = {
  generatedAt: new Date().toISOString(),
  description:
    "Do not delete. Archives (/archive/v1, /v2) depend on these files.",
  files: manifest.map((abs) => path.relative(root, abs).replace(/\\/g, "/")),
};

const manifestPath = path.join(publicDir, "archive", "MANIFEST.json");
ensureDir(path.dirname(manifestPath));
fs.writeFileSync(manifestPath, `${JSON.stringify(manifestOut, null, 2)}\n`);
console.log(`wrote ${manifestOut.files.length} entries to archive/MANIFEST.json`);
