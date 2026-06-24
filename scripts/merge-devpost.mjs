#!/usr/bin/env node
/**
 * Merge open online Devpost hackathons into opportunity-board-items.js
 *
 *   node scripts/merge-devpost.mjs
 *   node scripts/merge-devpost.mjs --dry-run
 *   node scripts/merge-devpost.mjs --skip-roles   # API only, no description scan
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
    .replace(/&nbsp;/gi, " ")
    .replace(/&amp;/gi, "&")
    .replace(/&quot;/gi, '"')
    .replace(/&#39;/gi, "'")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">")
    .replace(/\s+/g, " ")
    .trim();
}

const DESC_START = /id="challenge-description">/i;
const DESC_END =
  /<article\s+id="(?:judges|prizes)"|<div\s+class="info-with-icons"/i;

const ROLE_PATTERNS = [
  /call\s+for\s+[\w\s-]{0,50}(?:judges?|mentors?|speakers?|panelists?|volunteers?|coaches?|ambassadors?|facilitators?|moderators?)/i,
  /(?:looking|seeking)\s+for\s+[\w\s-]{0,50}(?:judges?|mentors?|speakers?|panelists?|volunteers?|coaches?|ambassadors?|facilitators?)/i,
  /(?:judges?|mentors?|speakers?|panelists?|volunteers?|coaches?)\s+(?:wanted|needed|applications?\s+(?:are\s+)?open)/i,
  /apply\s+(?:here|now|to\s+be|to\s+join)\s+.{0,80}(?:judge|mentor|speaker|volunteer|panelist)/i,
  /interested\s+in\s+(?:being|serving\s+as|joining\s+as)\s+(?:a\s+)?(?:judge|mentor|speaker|volunteer|panelist)/i,
  /(?:sign\s*up|register)\s+(?:here\s+)?(?:to\s+)?(?:be\s+)?(?:a\s+)?(?:judge|mentor|speaker|volunteer)/i,
  /(?:judge|mentor|speaker|volunteer)\s+(?:signup|sign-up|application|interest)\s+(?:form|link)/i,
  /we\s+(?:are\s+)?(?:recruiting|accepting)\s+[\w\s]{0,30}(?:judges?|mentors?|speakers?|volunteers?)/i,
];

const ROLE_WORDS = [
  "judge",
  "mentor",
  "speaker",
  "volunteer",
  "panelist",
  "coach",
  "ambassador",
  "facilitator",
  "moderator",
];

const FORM_URL_PATTERN =
  /https?:\/\/(?:forms\.gle\/[A-Za-z0-9_-]+|docs\.google\.com\/forms\/[^\s"'<>]+|typeform\.com\/to\/[^\s"'<>]+|tally\.so\/[^\s"'<>]+)/gi;

function extractChallengeDescription(pageHtml) {
  const match = pageHtml.match(DESC_START);
  if (!match) {
    return "";
  }

  const startIndex = match.index + match[0].length;
  const rest = pageHtml.slice(startIndex);
  const end = rest.search(DESC_END);
  const chunk = end === -1 ? rest.slice(0, 120_000) : rest.slice(0, end);
  return stripHtml(chunk);
}

function extractFormUrls(...sources) {
  const urls = new Set();
  for (const source of sources) {
    for (const match of String(source || "").matchAll(FORM_URL_PATTERN)) {
      urls.add(match[0].replace(/[.,;:!?)]+$/, ""));
    }
  }
  return [...urls];
}

function extractContactEmails(pageHtml) {
  const emails = new Set();
  for (const match of String(pageHtml || "").matchAll(/mailto:([^"'<>?\s]+)/gi)) {
    const raw = stripHtml(match[1]).split(/[<>"']/)[0].trim();
    if (/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(raw) && !/devpost/i.test(raw)) {
      emails.add(raw);
    }
  }
  return [...emails];
}

function roleLabelFromText(text) {
  const lower = text.toLowerCase();
  const labels = [
    ["judge", "Judge"],
    ["mentor", "Mentor"],
    ["speaker", "Speaker"],
    ["panelist", "Panelist"],
    ["volunteer", "Volunteer"],
    ["coach", "Coach"],
    ["ambassador", "Ambassador"],
    ["facilitator", "Facilitator"],
    ["moderator", "Moderator"],
  ];

  for (const [word, label] of labels) {
    if (lower.includes(word)) {
      return label;
    }
  }

  return "Contributor";
}

function detectRoleOpportunities(description, pageHtml) {
  if (!description || description.length < 80) {
    return [];
  }

  const snippets = [];
  for (const pattern of ROLE_PATTERNS) {
    const flags = pattern.flags.includes("g") ? pattern.flags : `${pattern.flags}g`;
    for (const match of description.matchAll(new RegExp(pattern.source, flags))) {
      snippets.push(match[0].trim());
    }
  }

  const lower = description.toLowerCase();
  const forms = extractFormUrls(description, pageHtml);
  const emails = extractContactEmails(pageHtml);

  if (snippets.length === 0) {
    for (const role of ROLE_WORDS) {
      if (!lower.includes(role)) {
        continue;
      }

      const contextual =
        new RegExp(
          `${role}.{0,200}(?:apply|form|sign up|register|email us|reach out|mailto)`,
          "i",
        ).test(lower) ||
        (forms.length > 0 &&
          new RegExp(`(?:become|join as|serve as).{0,80}${role}`, "i").test(
            lower,
          ));

      if (contextual) {
        snippets.push(`${role} opportunity mentioned in hackathon description`);
      }
    }
  }

  const uniqueSnippets = [...new Set(snippets)];
  if (uniqueSnippets.length === 0) {
    return [];
  }

  const byLabel = new Map();
  for (const snippet of uniqueSnippets) {
    const label = roleLabelFromText(snippet);
    if (!byLabel.has(label)) {
      byLabel.set(label, { label, snippet, forms, emails });
    }
  }

  return [...byLabel.values()];
}

function normalizeRecruitmentEmail(email) {
  return email.replace(/\.cim$/i, ".com");
}

function pickRoleApplyUrl(hackathonUrl, { forms, emails }) {
  if (forms[0]) {
    return forms[0];
  }
  if (emails[0]) {
    return `mailto:${normalizeRecruitmentEmail(emails[0])}`;
  }
  return `${hackathonUrl}#challenge-description`;
}

function roleOpportunityToItem(hackathon, hackathonItem, role) {
  const rolePlural = role.label === "Coach" ? "Coaches" : `${role.label}s`;
  const applyUrl = pickRoleApplyUrl(hackathon.url, role);
  const item = {
    title: `Call for ${rolePlural} — ${hackathon.title}`,
    type: "event",
    url: applyUrl,
    imageUrl: hackathon.url,
    org: hackathon.organization_name || hackathonItem.org,
    location: hackathonItem.location,
    blurb: `Devpost hackathon recruiting ${rolePlural.toLowerCase()}. ${stripHtml(role.snippet).slice(0, 120)}.`,
  };

  if (hackathonItem.dates?.length) {
    item.dates = hackathonItem.dates.map((entry) => ({
      ...entry,
      label: entry.label.replace(/^Submission/, "Hackathon submission"),
    }));
  }

  return item;
}

async function fetchHackathonPage(url) {
  const response = await fetch(url, {
    headers: { "User-Agent": "stimmie.dev-opportunities-bot/1.0" },
  });

  if (!response.ok) {
    throw new Error(`Devpost page ${response.status} for ${url}`);
  }

  return response.text();
}

async function mapInBatches(hackathons, batchSize = 6) {
  const results = new Map();
  for (let index = 0; index < hackathons.length; index += batchSize) {
    const batch = hackathons.slice(index, index + batchSize);
    const entries = await Promise.all(
      batch.map(async (item) => {
        try {
          const pageHtml = await fetchHackathonPage(item.url);
          return [item.url, pageHtml];
        } catch (error) {
          console.warn(`Skip description scan: ${item.url} (${error.message})`);
          return [item.url, ""];
        }
      }),
    );

    for (const [url, pageHtml] of entries) {
      results.set(url, pageHtml);
    }
  }

  return results;
}

function buildDevpostCandidates(hackathons, pageHtmlByUrl) {
  const hackathonItems = [];
  const roleItems = [];

  for (const hackathon of hackathons) {
    const item = devpostToItem(hackathon);
    if (!isStillOpen(item)) {
      continue;
    }

    hackathonItems.push(item);

    const pageHtml = pageHtmlByUrl.get(hackathon.url) || "";
    const description = extractChallengeDescription(pageHtml);
    const roles = detectRoleOpportunities(description, pageHtml);

    for (const role of roles) {
      roleItems.push(roleOpportunityToItem(hackathon, item, role));
    }
  }

  return { hackathonItems, roleItems };
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

function isRoleCallItem(item) {
  return item.type === "event" && /^Call for /i.test(item.title);
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
    const roleCall = isRoleCallItem(item);

    if (seenTitles.has(titleKey)) {
      continue;
    }

    if (!roleCall && seenUrls.has(urlKey)) {
      continue;
    }

    seenTitles.add(titleKey);
    if (!roleCall) {
      seenUrls.add(urlKey);
    }
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

function writeHarvestFile(
  hackathonCandidates,
  roleCandidates,
  { added, addedRoles, merged, existingCount },
) {
  const today = new Date().toISOString().slice(0, 10);
  const harvestPath = path.join(
    root,
    `research/opportunities/responses/${today}-devpost-harvest.md`,
  );

  const hackathonSections = hackathonCandidates.map((item) => {
    const json = JSON.stringify(itemToHarvestEntry(item), null, 2);
    return `### ${item.title}\n\n\`\`\`json\n${json}\n\`\`\``;
  });

  const roleSections = roleCandidates.map((item) => {
    const json = JSON.stringify(itemToHarvestEntry(item), null, 2);
    return `### ${item.title}\n\n\`\`\`json\n${json}\n\`\`\``;
  });

  const content = `# Devpost harvest — ${today}

- **Date researched:** ${today}
- **Tool:** \`npm run opportunities:devpost\` (Devpost API + description scan)
- **Prompt:** [../prompts/03-devpost-harvest.md](../prompts/03-devpost-harvest.md)
- **Status:** published
- **Issue slug:** devpost-${today}

---

## Coverage report

| Metric | Count |
| ------ | -----: |
| API candidates (open, online) | ${hackathonCandidates.length} |
| Role opportunities (judges, mentors, speakers, volunteers) | ${roleCandidates.length} |
| New hackathons merged this run | ${added} |
| New role listings merged this run | ${addedRoles} |
| Board total after merge | ${merged.length} |
| Prior board count | ${existingCount} |

Filter: open + online + not invite-only; blocklist for region-locked titles; skip closed submission deadlines. Role scan parses each Devpost \`#challenge-description\` for judge, mentor, speaker, and volunteer calls.

---

## Hackathons

${hackathonSections.join("\n\n")}

---

## Role opportunities

Calls for judges, mentors, speakers, or volunteers found in Devpost hackathon descriptions. \`url\` points to the apply form, email, or description anchor when no direct link exists.

${roleSections.length ? roleSections.join("\n\n") : "_No role opportunities detected this run._"}
`;

  fs.writeFileSync(harvestPath, content);
  return harvestPath;
}

function writeItemsFile(items) {
  const body = items.map((item) => formatItem(item)).join("\n");
  const content = `// Opportunity board items. Merged via scripts/merge-harvest.mjs and scripts/merge-devpost.mjs
// Regenerate harvest: node scripts/merge-harvest.mjs
// Regenerate Devpost: node scripts/merge-devpost.mjs
// Proton inbox scan: node scripts/scan-proton-export.mjs /path/to/mail_EXPORT_DIR

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

const skipRoles = process.argv.includes("--skip-roles");

const { opportunityBoardItems } = await import(
  `file://${outPath}?t=${Date.now()}`
);

const hackathons = await fetchDevpostHackathons();
const filteredHackathons = hackathons
  .filter((hackathon) => hackathon.open_state === "open")
  .filter((hackathon) => hackathon.displayed_location?.location === "Online")
  .filter((hackathon) => !hackathon.invite_only)
  .filter((hackathon) => !TITLE_BLOCKLIST.test(hackathon.title));

let pageHtmlByUrl = new Map();
if (!skipRoles) {
  console.log(
    `Scanning ${filteredHackathons.length} Devpost descriptions for role opportunities…`,
  );
  pageHtmlByUrl = await mapInBatches(filteredHackathons);
}

const { hackathonItems, roleItems } = buildDevpostCandidates(
  filteredHackathons,
  pageHtmlByUrl,
);

const upgraded = upgradeExistingDevpostUrls(opportunityBoardItems);
const { merged: afterHackathons, added } = mergeItems(upgraded, hackathonItems);
const { merged, added: addedRoles } = mergeItems(afterHackathons, roleItems);

if (dryRun) {
  console.log(
    `Would add ${added} Devpost hackathons and ${addedRoles} role listings (${hackathonItems.length} hackathons, ${roleItems.length} roles detected, ${opportunityBoardItems.length} existing → ${merged.length} total)`,
  );
  const newHackathons = afterHackathons.slice(upgraded.length);
  const newRoles = merged.slice(afterHackathons.length);
  for (const item of [...newHackathons, ...newRoles]) {
    console.log(`  + [${item.type}] ${item.title}: ${item.url}`);
  }
} else {
  writeItemsFile(merged);
  console.log(
    `Merged ${added} Devpost hackathons and ${addedRoles} role listings (${hackathonItems.length} hackathons, ${roleItems.length} roles detected, ${opportunityBoardItems.length} existing → ${merged.length} total)`,
  );
  console.log(`Wrote ${path.relative(root, outPath)}`);

  if (writeHarvest) {
    const harvestPath = writeHarvestFile(hackathonItems, roleItems, {
      added,
      addedRoles,
      merged,
      existingCount: opportunityBoardItems.length,
    });
    console.log(`Wrote ${path.relative(root, harvestPath)}`);
  }
}
