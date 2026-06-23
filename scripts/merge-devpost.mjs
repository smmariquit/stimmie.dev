#!/usr/bin/env node
/**
 * Merge open online Devpost hackathons into opportunity-board-items.js
 *
 *   node scripts/merge-devpost.mjs
 *   node scripts/merge-devpost.mjs --dry-run
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { slugifyOpportunityTitle } from "../src/data/opportunities.js";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const outPath = path.join(root, "src/data/opportunity-board-items.js");
const dryRun = process.argv.includes("--dry-run");
const writeHarvest = process.argv.includes("--harvest");

const MONTHS = {
  jan: 0,
  feb: 1,
  mar: 2,
  apr: 3,
  may: 4,
  jun: 5,
  jul: 6,
  aug: 7,
  sep: 8,
  oct: 9,
  nov: 10,
  dec: 11,
};

const TITLE_BLOCKLIST =
  /\b(india high school|iit india|africa deep tech|rice university urban)\b/i;

function normUrl(url) {
  try {
    const parsed = new URL(url);
    const pathname = `${parsed.pathname}${parsed.search}`.replace(/\/$/, "");
    return `${parsed.hostname}${pathname}`.toLowerCase();
  } catch {
    return url.toLowerCase();
  }
}

function pad2(value) {
  return String(value).padStart(2, "0");
}

function toIsoDate(year, monthIndex, day) {
  return `${year}-${pad2(monthIndex + 1)}-${pad2(day)}`;
}

function parseDatePart(part, year, monthHint) {
  const trimmed = part.trim();
  const withMonth = trimmed.match(/^([A-Za-z]+)\s+(\d{1,2})$/);
  if (withMonth) {
    const monthIndex = MONTHS[withMonth[1].slice(0, 3).toLowerCase()];
    if (monthIndex === undefined) return null;
    return toIsoDate(Number(year), monthIndex, Number(withMonth[2]));
  }

  const dayOnly = trimmed.match(/^(\d{1,2})$/);
  if (dayOnly && monthHint) {
    const monthIndex = MONTHS[monthHint.slice(0, 3).toLowerCase()];
    if (monthIndex === undefined) return null;
    return toIsoDate(Number(year), monthIndex, Number(dayOnly[1]));
  }

  return null;
}

function parseSubmissionPeriod(period) {
  const match = String(period || "").match(/^(.+?)\s*-\s*(.+),\s*(\d{4})$/);
  if (!match) return null;

  const [, startPart, endPart, year] = match;
  const startDate = parseDatePart(startPart, year);
  const endDate = parseDatePart(endPart, year, startPart);

  if (!endDate) return null;

  const dates = [
    {
      label: "Submission deadline",
      date: endDate,
    },
  ];

  if (startDate && startDate !== endDate) {
    dates.unshift({
      label: "Hackathon window",
      date: startDate,
      endDate,
    });
  }

  return dates;
}

function stripHtml(value) {
  return String(value || "")
    .replace(/<[^>]+>/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function devpostToItem(hackathon) {
  const themes = (hackathon.themes || []).map((theme) => theme.name);
  const beginnerFriendly = themes.some((name) => name === "Beginner Friendly");
  const prize = stripHtml(hackathon.prize_amount || "");
  const themeText =
    themes.slice(0, 3).join(", ") || "Open-ended build";
  const timeLeft = hackathon.time_left_to_submission
    ? ` ${hackathon.time_left_to_submission}.`
    : "";

  const item = {
    title: hackathon.title.trim(),
    type: "hackathon",
    url: hackathon.url,
    org: hackathon.organization_name || "Devpost",
    location: hackathon.displayed_location?.location || "Online",
    blurb: `${themeText} hackathon on Devpost.${prize ? ` Prizes: ${prize}.` : ""}${timeLeft}`.trim(),
  };

  const dates = parseSubmissionPeriod(hackathon.submission_period_dates);
  if (dates) {
    item.dates = dates;
  }

  if (beginnerFriendly) {
    item.beginnerFriendly = true;
  }

  if (hackathon.thumbnail_url) {
    item.imageUrl = hackathon.url;
  }

  return item;
}

function mergeItems(existing, incoming) {
  const seenUrls = new Set(existing.map((item) => normUrl(item.url)));
  const seenTitles = new Set(
    existing.map((item) => slugifyOpportunityTitle(item.title)),
  );

  const merged = [...existing];
  let added = 0;

  for (const item of incoming) {
    const urlKey = normUrl(item.url);
    const titleKey = slugifyOpportunityTitle(item.title);

    if (seenUrls.has(urlKey) || seenTitles.has(titleKey)) {
      continue;
    }

    seenUrls.add(urlKey);
    seenTitles.add(titleKey);
    merged.push(item);
    added += 1;
  }

  return { merged, added };
}

function formatItem(item, indent = "    ") {
  const lines = [`${indent}{`];
  lines.push(`${indent}  title: ${JSON.stringify(item.title)},`);
  lines.push(`${indent}  type: ${JSON.stringify(item.type)},`);
  lines.push(`${indent}  url: ${JSON.stringify(item.url)},`);

  if (item.imageUrl) {
    lines.push(`${indent}  imageUrl: ${JSON.stringify(item.imageUrl)},`);
  }
  if (item.image) {
    lines.push(`${indent}  image: ${JSON.stringify(item.image)},`);
  }
  if (item.imageAlt) {
    lines.push(`${indent}  imageAlt: ${JSON.stringify(item.imageAlt)},`);
  }

  lines.push(`${indent}  org: ${JSON.stringify(item.org)},`);
  lines.push(`${indent}  location: ${JSON.stringify(item.location)},`);

  if (item.dates?.length) {
    lines.push(
      `${indent}  dates: ${JSON.stringify(item.dates, null, 2).replace(/\n/g, `\n${indent}  `)},`,
    );
  }

  if (typeof item.beginnerFriendly === "boolean") {
    lines.push(`${indent}  beginnerFriendly: ${item.beginnerFriendly},`);
  }

  lines.push(`${indent}  blurb: ${JSON.stringify(item.blurb)},`);
  lines.push(`${indent}},`);
  return lines.join("\n");
}

function itemToHarvestEntry(item) {
  const entry = {
    title: item.title,
    type: item.type,
    url: item.url,
    org: item.org,
    location: item.location,
    blurb: item.blurb,
    source_platform: "Devpost",
    source_url: item.url,
    confidence: "High",
  };

  if (item.imageUrl) {
    entry.image_url = item.imageUrl;
  }
  if (item.dates?.length) {
    entry.dates = item.dates;
  }
  if (typeof item.beginnerFriendly === "boolean") {
    entry.beginner_friendly = item.beginnerFriendly;
  }

  return entry;
}

function writeHarvestFile(candidates, { added, merged, existingCount }) {
  const today = new Date().toISOString().slice(0, 10);
  const harvestPath = path.join(
    root,
    `research/opportunities/responses/${today}-devpost-harvest.md`,
  );

  const sections = candidates.map((item) => {
    const json = JSON.stringify(itemToHarvestEntry(item), null, 2);
    return `### ${item.title}\n\n\`\`\`json\n${json}\n\`\`\``;
  });

  const content = `# Devpost harvest — ${today}

- **Date researched:** ${today}
- **Tool:** \`npm run opportunities:devpost\` (Devpost API)
- **Prompt:** [../prompts/03-devpost-harvest.md](../prompts/03-devpost-harvest.md)
- **Status:** published
- **Issue slug:** devpost-${today}

---

## Coverage report

| Metric | Count |
| ------ | -----: |
| API candidates (open, online) | ${candidates.length} |
| New merged this run | ${added} |
| Board total after merge | ${merged.length} |
| Prior board count | ${existingCount} |

Filter: open + online + not invite-only; blocklist for region-locked titles; skip closed submission deadlines.

---

## Hackathons

${sections.join("\n\n")}
`;

  fs.writeFileSync(harvestPath, content);
  return harvestPath;
}

function writeItemsFile(items) {
  const body = items.map((item) => formatItem(item)).join("\n");
  const content = `// Opportunity board items. Merged via scripts/merge-harvest.mjs and scripts/merge-devpost.mjs
// Regenerate harvest: node scripts/merge-harvest.mjs
// Regenerate Devpost: node scripts/merge-devpost.mjs

export const opportunityBoardItems = [
${body}
];
`;

  fs.writeFileSync(outPath, content);
}

function isStillOpen(item) {
  const deadline = item.dates?.find((entry) =>
    /deadline|closes|submission/i.test(entry.label),
  )?.date;
  if (!deadline) return true;

  const today = new Date().toISOString().slice(0, 10);
  return deadline >= today;
}

async function fetchDevpostHackathons() {
  const params = new URLSearchParams();
  params.append("status[]", "upcoming");
  params.append("status[]", "open");
  params.append("challenge_type[]", "online");
  params.set("per_page", "100");

  const response = await fetch(
    `https://devpost.com/api/hackathons?${params.toString()}`,
    { headers: { Accept: "application/json" } },
  );

  if (!response.ok) {
    throw new Error(`Devpost API ${response.status}`);
  }

  const data = await response.json();
  return data.hackathons || [];
}

function upgradeExistingDevpostUrls(items) {
  const redditDevpost = "https://redditgameswithahook.devpost.com/";
  return items.map((item) => {
    if (
      slugifyOpportunityTitle(item.title) ===
        slugifyOpportunityTitle("Reddit's Games with a Hook Hackathon") &&
      item.url.includes("reddit.com")
    ) {
      return {
        ...item,
        url: redditDevpost,
        imageUrl: redditDevpost,
        blurb:
          "Build daily games for Reddit communities with Devvit, React, and Phaser. $40k prize pool on Devpost.",
      };
    }
    return item;
  });
}

const { opportunityBoardItems } = await import(
  `file://${outPath}?t=${Date.now()}`
);

const hackathons = await fetchDevpostHackathons();
const candidates = hackathons
  .filter((hackathon) => hackathon.open_state === "open")
  .filter((hackathon) => hackathon.displayed_location?.location === "Online")
  .filter((hackathon) => !hackathon.invite_only)
  .filter((hackathon) => !TITLE_BLOCKLIST.test(hackathon.title))
  .map(devpostToItem)
  .filter(isStillOpen);

const upgraded = upgradeExistingDevpostUrls(opportunityBoardItems);
const { merged, added } = mergeItems(upgraded, candidates);

if (dryRun) {
  console.log(
    `Would add ${added} Devpost hackathons (${candidates.length} candidates, ${opportunityBoardItems.length} existing → ${merged.length} total)`,
  );
  for (const item of merged.slice(-added)) {
    console.log(`  + ${item.title}: ${item.url}`);
  }
} else {
  writeItemsFile(merged);
  console.log(
    `Merged ${added} Devpost hackathons (${candidates.length} candidates, ${opportunityBoardItems.length} existing → ${merged.length} total)`,
  );
  console.log(`Wrote ${path.relative(root, outPath)}`);

  if (writeHarvest) {
    const harvestPath = writeHarvestFile(candidates, {
      added,
      merged,
      existingCount: opportunityBoardItems.length,
    });
    console.log(`Wrote ${path.relative(root, harvestPath)}`);
  }
}
