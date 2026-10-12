#!/usr/bin/env node
/**
 * Weekly Discord @everyone ping with new stimmie.dev/opportunities entries.
 *
 *   npm run opportunities:discord-digest
 *   npm run opportunities:discord-digest -- --dry-run
 *   npm run opportunities:discord-digest -- --force
 *
 * Meant to run every day via Heroku Scheduler; it no-ops on any day but
 * Monday (Asia/Manila) unless --force is passed. Diffs the live
 * opportunityBoardItems array against the version on GitHub from ~36 hours
 * ago (comfortably before this week's weekly-research commit, after last
 * week's) so it only announces what's actually new.
 *
 * Env (see .env.example):
 *   DISCORD_WEBHOOK_URL
 *   OPPORTUNITIES_DISCORD_REPO (default smmariquit/stimmie.dev)
 *   OPPORTUNITIES_DISCORD_BRANCH (default main)
 *
 * IMPORTANT: GitHub's commits API defaults to the repo's default branch when
 * no `sha` is given. This repo's default branch is `staging` (dependency
 * bumps only), not `main` (where the opportunities content actually lands),
 * so every lookup below passes `sha=branch` explicitly. Don't drop it.
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { opportunityBoardItems } from "../src/data/opportunity-board-items.js";
import { OPPORTUNITY_TYPES } from "../src/data/opportunities.js";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const args = new Set(process.argv.slice(2));
const dryRun = args.has("--dry-run");
const force = args.has("--force");

const MANILA_TZ = "Asia/Manila";
const BOARD_URL = "https://stimmie.dev/opportunities";
const DATA_PATH = "src/data/opportunity-board-items.js";
const PREVIOUS_SNAPSHOT_HOURS = 36;
const MAX_LISTED_ITEMS = 12;
const EMBED_DESCRIPTION_LIMIT = 4096;

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

function isManilaMonday(date = new Date()) {
  const weekday = new Intl.DateTimeFormat("en-US", {
    timeZone: MANILA_TZ,
    weekday: "short",
  }).format(date);
  return weekday === "Mon";
}

function githubHeaders() {
  const token = process.env.GITHUB_TOKEN ?? process.env.GH_TOKEN;
  return {
    "User-Agent": "stimmie-opportunities-discord-digest",
    Accept: "application/vnd.github+json",
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
}

/** Latest commit on `branch` touching DATA_PATH at or before `cutoff`, via the GitHub API. */
async function fetchPreviousCommitSha(repo, branch, cutoffIso) {
  const url = `https://api.github.com/repos/${repo}/commits?sha=${encodeURIComponent(branch)}&path=${encodeURIComponent(DATA_PATH)}&until=${cutoffIso}&per_page=1`;
  const res = await fetch(url, { headers: githubHeaders() });
  if (!res.ok) {
    throw new Error(`GitHub commits lookup failed: ${res.status} ${await res.text()}`);
  }
  const commits = await res.json();
  return commits[0]?.sha ?? null;
}

/** Fetches opportunityBoardItems as it existed at `sha`. Returns [] if there's no earlier history. */
async function fetchItemsAtCommit(repo, sha) {
  if (!sha) {
    return [];
  }

  const url = `https://api.github.com/repos/${repo}/contents/${DATA_PATH}?ref=${sha}`;
  const res = await fetch(url, { headers: githubHeaders() });
  if (!res.ok) {
    throw new Error(`GitHub content fetch failed: ${res.status} ${await res.text()}`);
  }

  const { content } = await res.json();
  const source = Buffer.from(content, "base64").toString("utf-8");

  const tmpPath = path.join(root, `.tmp-opportunity-board-items-${sha}.mjs`);
  fs.writeFileSync(tmpPath, source);
  try {
    const mod = await import(pathToFileURL(tmpPath).href);
    return mod.opportunityBoardItems;
  } finally {
    fs.unlinkSync(tmpPath);
  }
}

function formatDeadline(item) {
  const entry = item.dates?.[0];
  if (!entry) {
    return "";
  }
  return ` — ${entry.label} ${entry.date}`;
}

function formatEntry(item) {
  const typeLabel = OPPORTUNITY_TYPES[item.type]?.label ?? item.type;
  return `**[${item.title}](${item.url})** · ${typeLabel} · ${item.org}${formatDeadline(item)}\n${item.blurb}`;
}

function buildDigest(newItems) {
  const shown = newItems.slice(0, MAX_LISTED_ITEMS);
  const overflow = newItems.length - shown.length;

  let description = shown.map(formatEntry).join("\n\n");
  if (overflow > 0) {
    description += `\n\n...and ${overflow} more on the board.`;
  }
  if (description.length > EMBED_DESCRIPTION_LIMIT) {
    description = `${description.slice(0, EMBED_DESCRIPTION_LIMIT - 1)}…`;
  }

  const title = `${newItems.length} new opportunit${newItems.length === 1 ? "y" : "ies"} this week`;

  return { title, description };
}

async function postToDiscord({ title, description }) {
  const webhookUrl = process.env.DISCORD_WEBHOOK_URL;
  if (!webhookUrl) {
    throw new Error("DISCORD_WEBHOOK_URL is not set");
  }

  const res = await fetch(webhookUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      content: "@everyone",
      username: "Stimmie Opportunities",
      allowed_mentions: { parse: ["everyone"] },
      embeds: [
        {
          title,
          url: BOARD_URL,
          description,
          color: 0xff6b35,
          footer: { text: "stimmie.dev/opportunities" },
        },
      ],
    }),
  });

  if (!res.ok) {
    throw new Error(`Discord webhook failed: ${res.status} ${await res.text()}`);
  }
}

async function main() {
  if (!force && !isManilaMonday()) {
    console.log("Not Monday in Asia/Manila, skipping (pass --force to override).");
    return;
  }

  const repo = process.env.OPPORTUNITIES_DISCORD_REPO ?? "smmariquit/stimmie.dev";
  const branch = process.env.OPPORTUNITIES_DISCORD_BRANCH ?? "main";
  const cutoff = new Date(
    Date.now() - PREVIOUS_SNAPSHOT_HOURS * 60 * 60 * 1000,
  ).toISOString();

  const previousSha = await fetchPreviousCommitSha(repo, branch, cutoff);
  const previousItems = await fetchItemsAtCommit(repo, previousSha);
  const previousUrls = new Set(previousItems.map((item) => item.url));

  const newItems = opportunityBoardItems.filter(
    (item) => !previousUrls.has(item.url),
  );

  console.log(
    `Previous snapshot: ${previousSha ?? "(none, first run)"}, ${previousItems.length} items. Current: ${opportunityBoardItems.length} items. New: ${newItems.length}.`,
  );

  if (!newItems.length) {
    console.log("No new opportunities since last week, skipping post.");
    return;
  }

  const digest = buildDigest(newItems);

  if (dryRun) {
    console.log("DRY RUN — would post to Discord:");
    console.log(digest.title);
    console.log(digest.description);
    return;
  }

  await postToDiscord(digest);
  console.log(`Posted ${newItems.length} new opportunities to Discord.`);
}

main().catch((error) => {
  console.error(error.message ?? error);
  process.exit(1);
});
