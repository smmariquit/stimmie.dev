#!/usr/bin/env node
/**
 * Email nag when opportunitiesBoard.lastUpdated is older than N days.
 *
 *   npm run opportunities:stale-nag
 *   npm run opportunities:stale-nag -- --dry-run
 *   npm run opportunities:stale-nag -- --force
 *
 * Env (see .env.example):
 *   RESEND_API_KEY
 *   OPPORTUNITIES_NAG_TO
 *   OPPORTUNITIES_NAG_FROM
 *   OPPORTUNITIES_STALE_DAYS (default 14)
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { getOpportunitiesBoard } from "../src/data/opportunities.js";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const args = new Set(process.argv.slice(2));
const dryRun = args.has("--dry-run");
const force = args.has("--force");

const MANILA_TZ = "Asia/Manila";
const BOARD_URL = "https://stimmie.dev/opportunities";
const EDIT_HINT =
  "src/data/opportunities.js → opportunitiesBoard.lastUpdated (YYYY-MM-DD)";

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

function todayManilaDateKey() {
  return new Intl.DateTimeFormat("en-CA", { timeZone: MANILA_TZ }).format(
    new Date(),
  );
}

function daysBetween(startKey, endKey) {
  const start = new Date(`${startKey}T12:00:00Z`).getTime();
  const end = new Date(`${endKey}T12:00:00Z`).getTime();
  return Math.round((end - start) / 86_400_000);
}

function formatHumanDate(dateKey) {
  return new Date(`${dateKey}T12:00:00Z`).toLocaleDateString("en-US", {
    month: "long",
    day: "numeric",
    year: "numeric",
    timeZone: MANILA_TZ,
  });
}

function buildCopy({ lastUpdated, daysStale, listingCount, staleAfter }) {
  const overdue = Math.max(0, daysStale - staleAfter);
  const subject =
    overdue >= 14
      ? `🚨 Opportunities board is ${daysStale} days stale. Update it.`
      : overdue >= 7
        ? `⚠️ Opportunities board: ${daysStale} days without an update`
        : `Opportunities board is ${daysStale} days old`;

  const intro =
    overdue >= 14
      ? "Okay. The opportunities page is rotting on the vine and people can tell."
      : overdue >= 7
        ? "Friendly ping that turned less friendly: your board is getting old."
        : "Your opportunities roundup is past the freshness window you set.";

  const text = [
    intro,
    "",
    `Last updated: ${formatHumanDate(lastUpdated)} (${lastUpdated})`,
    `Today (Manila): ${todayManilaDateKey()}`,
    `Days since update: ${daysStale}`,
    `Your threshold: ${staleAfter} days`,
    `Live listings on board: ${listingCount}`,
    "",
    `Live page: ${BOARD_URL}`,
    `Bump date in: ${EDIT_HINT}`,
    "",
    "Quick refresh checklist:",
    "  1. Harvest / merge new entries",
    "  2. npm run opportunities:devpost",
    "  3. npm run opportunities:images",
    "  4. Bump lastUpdated and push to main",
    "",
    "This email repeats daily until you update lastUpdated.",
    "",
    "— opportunities stale nag (stimmie.dev)",
  ].join("\n");

  const html = `
    <div style="font-family:system-ui,sans-serif;line-height:1.55;color:#111;max-width:36rem">
      <p style="font-size:1.05rem;font-weight:700;margin:0 0 0.75rem">${intro}</p>
      <table style="border-collapse:collapse;margin:0 0 1rem;font-size:0.95rem">
        <tr><td style="padding:0.2rem 0.75rem 0.2rem 0;color:#555">Last updated</td><td><strong>${formatHumanDate(lastUpdated)}</strong> (${lastUpdated})</td></tr>
        <tr><td style="padding:0.2rem 0.75rem 0.2rem 0;color:#555">Today (Manila)</td><td>${todayManilaDateKey()}</td></tr>
        <tr><td style="padding:0.2rem 0.75rem 0.2rem 0;color:#555">Days stale</td><td><strong>${daysStale}</strong> (threshold: ${staleAfter})</td></tr>
        <tr><td style="padding:0.2rem 0.75rem 0.2rem 0;color:#555">Listings</td><td>${listingCount}</td></tr>
      </table>
      <p style="margin:0 0 0.75rem">
        <a href="${BOARD_URL}">Open the live board</a>
      </p>
      <p style="margin:0 0 0.75rem;font-size:0.9rem;color:#444">
        Bump <code>${EDIT_HINT}</code>, then push.
      </p>
      <ol style="margin:0 0 1rem;padding-left:1.2rem;font-size:0.9rem">
        <li>Harvest / merge new entries</li>
        <li><code>npm run opportunities:devpost</code></li>
        <li><code>npm run opportunities:images</code></li>
        <li>Push to main</li>
      </ol>
      <p style="margin:0;font-size:0.85rem;color:#666">
        This nags you daily until <code>lastUpdated</code> is current.
      </p>
    </div>
  `.trim();

  return { subject, text, html };
}

async function sendResendEmail({ to, from, subject, text, html }) {
  const apiKey = process.env.RESEND_API_KEY;
  if (!apiKey) {
    throw new Error("RESEND_API_KEY is not set");
  }

  const response = await fetch("https://api.resend.com/emails", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      from,
      to: [to],
      subject,
      text,
      html,
    }),
  });

  const body = await response.text();
  if (!response.ok) {
    throw new Error(`Resend ${response.status}: ${body}`);
  }

  return body;
}

async function main() {
  const board = getOpportunitiesBoard();
  const lastUpdated = board.lastUpdated;
  const today = todayManilaDateKey();
  const daysStale = daysBetween(lastUpdated, today);
  const staleAfter = Number(process.env.OPPORTUNITIES_STALE_DAYS ?? 14);
  const listingCount = board.items.length;

  if (!lastUpdated || !/^\d{4}-\d{2}-\d{2}$/.test(lastUpdated)) {
    console.error(`Invalid opportunitiesBoard.lastUpdated: ${lastUpdated}`);
    process.exit(1);
  }

  const isStale = daysStale >= staleAfter;

  console.log(
    `Opportunities board: lastUpdated=${lastUpdated}, today=${today}, days=${daysStale}, threshold=${staleAfter}, stale=${isStale}`,
  );

  if (!isStale && !force) {
    console.log("Board is fresh. No email sent.");
    return;
  }

  const to = process.env.OPPORTUNITIES_NAG_TO;
  const from =
    process.env.OPPORTUNITIES_NAG_FROM ??
    "stimmie.dev opportunities <onboarding@resend.dev>";

  const copy = buildCopy({
    lastUpdated,
    daysStale,
    listingCount,
    staleAfter,
  });

  if (dryRun) {
    console.log("DRY RUN — would send email:");
    console.log(`  to: ${to ?? "(missing OPPORTUNITIES_NAG_TO)"}`);
    console.log(`  from: ${from}`);
    console.log(`  subject: ${copy.subject}`);
    console.log(copy.text);
    return;
  }

  if (!to) {
    console.error("OPPORTUNITIES_NAG_TO is not set");
    process.exit(1);
  }

  const result = await sendResendEmail({
    to,
    from,
    subject: copy.subject,
    text: copy.text,
    html: copy.html,
  });

  console.log(`Nag email sent to ${to}`);
  console.log(result);
}

main().catch((error) => {
  console.error(error.message ?? error);
  process.exit(1);
});
