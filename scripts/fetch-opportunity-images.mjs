#!/usr/bin/env node
/**
 * Fetch cover images for /opportunities:
 *   1. og:image / twitter:image from the page
 *   2. Screenshot API fallback (ScreenshotOne or Microlink)
 *   3. Type default placeholder
 *
 *   npm run opportunities:images
 *   npm run opportunities:images -- --force
 *   npm run opportunities:images -- --retry-fallbacks
 *   npm run opportunities:images -- --screenshot-only --retry-fallbacks
 *
 * Env: see .env.example (loads .env.local and .env from repo root)
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  OPPORTUNITY_TYPE_DEFAULT_IMAGES,
  getOpportunityId,
  opportunityIssues,
} from "../src/data/opportunities.js";
import {
  getScreenshotProvider,
  takePageScreenshot,
} from "./lib/opportunity-screenshot.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const outDir = path.join(root, "public", "opportunities");
const manifestPath = path.join(root, "src", "data", "opportunity-images.json");

const USER_AGENT =
  "stimmie.dev-opportunity-fetcher/1.0 (+https://stimmie.dev/opportunities)";
const FETCH_TIMEOUT_MS = 20_000;
const MAX_IMAGE_BYTES = 5 * 1024 * 1024;
const SCREENSHOT_DELAY_MS = 1200;

const args = process.argv.slice(2);
const force = args.includes("--force");
const screenshotOnly = args.includes("--screenshot-only");
const retryFallbacks = args.includes("--retry-fallbacks");
const onlyId = args.find((a) => a.startsWith("--id="))?.split("=")[1];

loadLocalEnv();

function loadLocalEnv() {
  for (const name of [".env.local", ".env"]) {
    const envPath = path.join(root, name);
    if (!fs.existsSync(envPath)) {
      continue;
    }
    for (const line of fs.readFileSync(envPath, "utf8").split("\n")) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith("#")) {
        continue;
      }
      const eq = trimmed.indexOf("=");
      if (eq === -1) {
        continue;
      }
      const key = trimmed.slice(0, eq).trim();
      let value = trimmed.slice(eq + 1).trim();
      if (
        (value.startsWith('"') && value.endsWith('"')) ||
        (value.startsWith("'") && value.endsWith("'"))
      ) {
        value = value.slice(1, -1);
      }
      if (!process.env[key]) {
        process.env[key] = value;
      }
    }
  }
}

function loadManifest() {
  if (!fs.existsSync(manifestPath)) {
    return { generatedAt: null, images: {} };
  }
  return JSON.parse(fs.readFileSync(manifestPath, "utf8"));
}

function saveManifest(manifest) {
  manifest.generatedAt = new Date().toISOString();
  fs.writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
}

function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true });
}

function extensionFromContentType(contentType) {
  if (!contentType) {
    return ".jpg";
  }
  if (contentType.includes("png")) {
    return ".png";
  }
  if (contentType.includes("webp")) {
    return ".webp";
  }
  if (contentType.includes("gif")) {
    return ".gif";
  }
  if (contentType.includes("svg")) {
    return ".svg";
  }
  return ".jpg";
}

function extensionFromUrl(imageUrl) {
  try {
    const ext = path.extname(new URL(imageUrl).pathname).toLowerCase();
    if ([".jpg", ".jpeg", ".png", ".webp", ".gif"].includes(ext)) {
      return ext === ".jpeg" ? ".jpg" : ext;
    }
  } catch {
    // ignore
  }
  return null;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function fetchWithTimeout(url, options = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
  try {
    return await fetch(url, {
      ...options,
      signal: controller.signal,
      headers: {
        "User-Agent": USER_AGENT,
        Accept: "text/html,application/xhtml+xml,image/*,*/*;q=0.8",
        ...options.headers,
      },
      redirect: "follow",
    });
  } finally {
    clearTimeout(timer);
  }
}

function extractOgImage(html, pageUrl) {
  const patterns = [
    /<meta[^>]+property=["']og:image(?::url)?["'][^>]+content=["']([^"']+)["']/gi,
    /<meta[^>]+content=["']([^"']+)["'][^>]+property=["']og:image(?::url)?["']/gi,
    /<meta[^>]+name=["']twitter:image(?::src)?["'][^>]+content=["']([^"']+)["']/gi,
    /<meta[^>]+content=["']([^"']+)["'][^>]+name=["']twitter:image(?::src)?["']/gi,
  ];

  for (const pattern of patterns) {
    pattern.lastIndex = 0;
    const match = pattern.exec(html);
    if (match?.[1]) {
      try {
        return new URL(match[1].trim(), pageUrl).href;
      } catch {
        // try next pattern
      }
    }
  }

  return null;
}

function listOpportunities() {
  const rows = [];
  for (const issue of opportunityIssues) {
    for (const item of issue.items) {
      const id = getOpportunityId(issue.slug, item);
      rows.push({ issue, item, id });
    }
  }
  return rows;
}

function localPathForId(id, ext) {
  return {
    destPath: path.join(outDir, `${id}${ext}`),
    publicPath: `/opportunities/${id}${ext}`,
  };
}

function findExistingImageFile(id) {
  for (const ext of [".webp", ".jpg", ".png", ".jpeg", ".gif"]) {
    const destPath = path.join(outDir, `${id}${ext}`);
    if (fs.existsSync(destPath)) {
      return localPathForId(id, ext);
    }
  }
  return null;
}

function shouldProcessRow(id, manifest) {
  if (onlyId && id !== onlyId) {
    return false;
  }
  if (retryFallbacks) {
    const status = manifest.images[id]?.status;
    return !status || status === "fallback";
  }
  return true;
}

function writeImageFile({ id, issue, item, buffer, contentType, status, extra }) {
  const ext = extensionFromContentType(contentType);
  const { destPath, publicPath } = localPathForId(id, ext);

  const existing = findExistingImageFile(id);
  if (existing && existing.destPath !== destPath && fs.existsSync(existing.destPath)) {
    fs.unlinkSync(existing.destPath);
  }

  if (buffer.byteLength > MAX_IMAGE_BYTES) {
    throw new Error(`image too large (${buffer.byteLength} bytes)`);
  }

  ensureDir(path.dirname(destPath));
  fs.writeFileSync(destPath, buffer);

  return {
    status,
    path: publicPath,
    pageUrl: item.url,
    issueSlug: issue.slug,
    title: item.title,
    contentType,
    bytes: buffer.byteLength,
    ...extra,
  };
}

async function fetchOgImageForPage(pageUrl) {
  const response = await fetchWithTimeout(pageUrl, {
    headers: { Accept: "text/html,application/xhtml+xml" },
  });
  if (!response.ok) {
    throw new Error(`page HTTP ${response.status}`);
  }

  const html = await response.text();
  const imageUrl = extractOgImage(html, response.url || pageUrl);
  if (!imageUrl) {
    throw new Error("no og:image or twitter:image found");
  }

  return imageUrl;
}

async function saveFromOg({ issue, item, id }) {
  const pageUrl = item.imageUrl || item.url;
  const imageUrl = await fetchOgImageForPage(pageUrl);
  const probe = await fetchWithTimeout(imageUrl);

  if (!probe.ok) {
    throw new Error(`image HTTP ${probe.status}`);
  }

  const contentType = probe.headers.get("content-type") ?? "";
  if (!contentType.startsWith("image/")) {
    throw new Error(`not an image (${contentType || "unknown type"})`);
  }

  const buffer = Buffer.from(await probe.arrayBuffer());
  const manifestEntry = writeImageFile({
    id,
    issue,
    item,
    buffer,
    contentType,
    status: "fetched",
    extra: { imageUrl, source: "og" },
  });

  return { id, status: "fetched", detail: imageUrl, manifestEntry };
}

async function saveFromScreenshot({ issue, item, id }) {
  await sleep(SCREENSHOT_DELAY_MS);
  const pageUrl = item.imageUrl || item.url;
  const shot = await takePageScreenshot(pageUrl);
  const manifestEntry = writeImageFile({
    id,
    issue,
    item,
    buffer: shot.buffer,
    contentType: shot.contentType,
    status: "screenshot",
    extra: {
      screenshotProvider: shot.provider,
      screenshotUrl: shot.sourceUrl,
      source: "screenshot",
    },
  });

  return {
    id,
    status: "screenshot",
    detail: `${shot.provider} → ${manifestEntry.path}`,
    manifestEntry,
  };
}

function saveTypeFallback({ issue, item, id, errors }) {
  const fallback =
    OPPORTUNITY_TYPE_DEFAULT_IMAGES[item.type] ??
    OPPORTUNITY_TYPE_DEFAULT_IMAGES.program;

  return {
    status: "fallback",
    path: fallback,
    pageUrl: item.url,
    issueSlug: issue.slug,
    title: item.title,
    error: errors.join(" | "),
  };
}

async function processOpportunity(row, manifest) {
  const { issue, item, id } = row;

  if (!shouldProcessRow(id, manifest)) {
    return null;
  }

  if (item.image) {
    manifest.images[id] = {
      status: "override",
      path: item.image,
      pageUrl: item.url,
      issueSlug: issue.slug,
      title: item.title,
    };
    return { id, status: "override", detail: "manual image field" };
  }

  if (!force) {
    const existing = findExistingImageFile(id);
    const prior = manifest.images[id];
    if (
      existing &&
      prior &&
      (prior.status === "fetched" || prior.status === "screenshot")
    ) {
      manifest.images[id] = {
        ...prior,
        path: existing.publicPath,
        cached: true,
      };
      return { id, status: "cached", detail: existing.publicPath };
    }
  }

  const errors = [];

  if (!screenshotOnly) {
    try {
      const result = await saveFromOg({ issue, item, id });
      manifest.images[id] = result.manifestEntry;
      return result;
    } catch (error) {
      errors.push(
        `og: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }

  try {
    const result = await saveFromScreenshot({ issue, item, id });
    manifest.images[id] = result.manifestEntry;
    return result;
  } catch (error) {
    errors.push(
      `screenshot: ${error instanceof Error ? error.message : String(error)}`,
    );
  }

  manifest.images[id] = saveTypeFallback({ issue, item, id, errors });
  return { id, status: "fallback", detail: manifest.images[id].error };
}

async function main() {
  ensureDir(outDir);
  const manifest = loadManifest();
  const rows = listOpportunities();
  const provider = getScreenshotProvider();

  console.log(`Processing ${rows.length} opportunities…`);
  console.log(
    `  mode: ${screenshotOnly ? "screenshot-only" : "og → screenshot → default"}`,
  );
  console.log(`  screenshot provider: ${provider}`);
  if (retryFallbacks) {
    console.log("  retrying prior fallbacks only");
  }
  console.log();

  const summary = {
    fetched: 0,
    screenshot: 0,
    cached: 0,
    fallback: 0,
    override: 0,
  };

  for (const row of rows) {
    const result = await processOpportunity(row, manifest);
    if (!result) {
      continue;
    }

    summary[result.status] = (summary[result.status] ?? 0) + 1;
    const label = result.status.padEnd(10);
    console.log(
      `${label} ${result.id}${result.detail ? ` — ${result.detail}` : ""}`,
    );
  }

  saveManifest(manifest);

  console.log("\nDone.");
  console.log(
    `  og: ${summary.fetched ?? 0}  screenshot: ${summary.screenshot ?? 0}  cached: ${summary.cached ?? 0}  fallback: ${summary.fallback ?? 0}  override: ${summary.override ?? 0}`,
  );
  console.log(`  manifest: ${path.relative(root, manifestPath)}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
