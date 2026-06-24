#!/usr/bin/env node
/**
 * Scan a Proton Mail export for opportunity-like emails.
 *
 *   node scripts/scan-proton-export.mjs /path/to/mail_EXPORT_DIR
 *   node scripts/scan-proton-export.mjs /path/to/mail_EXPORT_DIR --days 45
 */

import fs from "node:fs";
import path from "node:path";

const exportDir = process.argv[2];
const daysArg = process.argv.find((a) => a.startsWith("--days="));
const maxDays = daysArg ? Number(daysArg.split("=")[1]) : 45;

if (!exportDir || !fs.existsSync(exportDir)) {
  console.error(
    "Usage: node scripts/scan-proton-export.mjs /path/to/mail_EXPORT_DIR [--days=45]",
  );
  process.exit(1);
}

const SUBJECT_SIGNALS =
  /\b(hackathon|internship|scholarship|fellowship|grant|cohort|deadline|register|registration|RSVP|call for|apply now|applications? open|luma|devpost|IEEE|UNESCO|Outreachy|GSoC|meetup|workshop|conference|competition|challenge|OJT|volunteer|opportunity|CFP|submission)\b/i;

const SUBJECT_BLOCK =
  /\b(unsubscribe|password|receipt|invoice|verification code|confirm your email|newsletter digest|your order|shipping|promo code only)\b/i;

const URL_RE = /https?:\/\/[^\s<>"')\]]+/gi;

function protonTimeToDate(seconds) {
  return new Date(seconds * 1000);
}

function stripHtml(html) {
  return html
    .replace(/<style[\s\S]*?<\/style>/gi, " ")
    .replace(/<script[\s\S]*?<\/script>/gi, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function decodeQuotedPrintable(text) {
  return text
    .replace(/=\r?\n/g, "")
    .replace(/=([0-9A-Fa-f]{2})/g, (_, hex) =>
      String.fromCharCode(Number.parseInt(hex, 16)),
    );
}

function extractUrls(text) {
  const decoded = decodeQuotedPrintable(text);
  const urls = [...decoded.matchAll(URL_RE)].map((m) =>
    m[0].replace(/[.,;:!?)]+$/, ""),
  );
  return [...new Set(urls)];
}

function scoreUrl(url) {
  const lower = url.toLowerCase();
  if (/unsubscribe|mailto:|proton\.me|protonmail|facebook\.com\/tr|google\.com\/url\?/.test(lower)) {
    return -10;
  }
  if (/luma\.com|devpost\.com|computer\.org|ieee\.org|outreachy|eventbrite|meetup\.com|linkedin\.com\/jobs\/view|greenhouse|lever\.co|myworkdayjobs|\.gov\.ph|internship|hackathon|apply|register|scholarship/.test(lower)) {
    return 10;
  }
  if (/bit\.ly|lnkd\.in|t\.co/.test(lower)) {
    return 3;
  }
  return 1;
}

function pickBestUrl(urls) {
  return [...urls].sort((a, b) => scoreUrl(b) - scoreUrl(a))[0] ?? null;
}

const cutoff = new Date();
cutoff.setDate(cutoff.getDate() - maxDays);

const files = fs.readdirSync(exportDir).filter((f) => f.endsWith(".metadata.json"));
const hits = [];

for (const metaFile of files) {
  const raw = JSON.parse(fs.readFileSync(path.join(exportDir, metaFile), "utf8"));
  const payload = raw.Payload ?? raw;
  const subject = payload.Subject ?? "";
  const sender = payload.Sender?.Address ?? "";
  const timeSec = payload.Time ?? 0;
  const received = protonTimeToDate(timeSec);

  if (received < cutoff) {
    continue;
  }
  if (!SUBJECT_SIGNALS.test(subject)) {
    continue;
  }
  if (SUBJECT_BLOCK.test(subject) && !/hackathon|internship|scholarship|IEEE|luma|devpost/i.test(subject)) {
    continue;
  }

  const emlFile = metaFile.replace(/\.metadata\.json$/, ".eml");
  const emlPath = path.join(exportDir, emlFile);
  let body = "";
  if (fs.existsSync(emlPath)) {
    body = fs.readFileSync(emlPath, "utf8");
    if (body.length > 500_000) {
      body = body.slice(0, 500_000);
    }
  }

  const urls = extractUrls(body);
  const bestUrl = pickBestUrl(urls);

  hits.push({
    subject,
    sender,
    received: received.toISOString().slice(0, 10),
    bestUrl,
    urlCount: urls.length,
    topUrls: urls.sort((a, b) => scoreUrl(b) - scoreUrl(a)).slice(0, 5),
  });
}

hits.sort((a, b) => b.received.localeCompare(a.received));

console.log(JSON.stringify({ scanned: files.length, cutoff: cutoff.toISOString().slice(0, 10), hits: hits.length, items: hits }, null, 2));
