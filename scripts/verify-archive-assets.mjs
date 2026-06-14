#!/usr/bin/env node
/**
 * Fail if any file listed in public/archive/MANIFEST.json is missing.
 *   npm run archive:verify
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const manifestPath = path.join(root, "public/archive/MANIFEST.json");

if (!fs.existsSync(manifestPath)) {
  console.error("Missing public/archive/MANIFEST.json — run npm run archive:snapshot");
  process.exit(1);
}

const { files } = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
const missing = files.filter((rel) => !fs.existsSync(path.join(root, rel)));

if (missing.length > 0) {
  console.error(`Archive verify failed: ${missing.length} missing file(s):`);
  for (const f of missing.slice(0, 20)) console.error(`  - ${f}`);
  if (missing.length > 20) console.error(`  … and ${missing.length - 20} more`);
  process.exit(1);
}

console.log(`Archive verify OK (${files.length} files)`);
