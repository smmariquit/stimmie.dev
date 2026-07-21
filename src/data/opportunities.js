// src/data/opportunities.js
//
// Curated board for /opportunities — one living page, not issues.
// Bump `lastUpdated` when you add, remove, or materially edit entries.
//
// Research pipeline: research/opportunities/
// Optional `image` overrides everything. Optional `imageUrl` for image fetch only.

import { opportunityBoardItems } from "./opportunity-board-items.js";
import opportunityImagesManifest from "./opportunity-images.json" with { type: "json" };

const MANILA_TZ = "Asia/Manila";
/** Matches existing cover-image filenames and manifest keys. */
const OPPORTUNITY_IMAGE_PREFIX = "q3-2026";

function loadOpportunityImages() {
  return opportunityImagesManifest;
}

export const OPPORTUNITY_TYPE_DEFAULT_IMAGES = {
  hackathon: "/opportunities/defaults/hackathon.svg",
  "game-jam": "/opportunities/defaults/game-jam.svg",
  internship: "/opportunities/defaults/internship.svg",
  event: "/opportunities/defaults/event.svg",
  certificate: "/opportunities/defaults/certificate.svg",
  program: "/opportunities/defaults/program.svg",
};

/** Shown when no cover image exists or the file fails to load. */
export const OPPORTUNITY_UNAVAILABLE_IMAGE =
  "/opportunities/defaults/unavailable.svg";

/** Shared cover for UPOU MODeL courses (model.upou.edu.ph). */
export const UPOU_MODEL_IMAGE = "/opportunities/shared/upou-model.png";

function isUpouModelOpportunity(item) {
  return /model\.upou\.edu\.ph/i.test(item.url ?? "");
}

export const OPPORTUNITY_TYPES = {
  hackathon: { label: "Hackathon", badge: "neo-badge-hackathon" },
  "game-jam": { label: "Game jam", badge: "neo-badge-game-jam" },
  internship: { label: "Internship", badge: "neo-badge-internship" },
  event: { label: "Event", badge: "neo-badge-event" },
  certificate: { label: "Certificate", badge: "neo-badge-certificate" },
  program: { label: "Program", badge: "neo-badge-program" },
};

/** Display order for grouped issue pages. */
export const OPPORTUNITY_TYPE_ORDER = [
  "hackathon",
  "game-jam",
  "internship",
  "program",
  "event",
  "certificate",
];

export const opportunitiesBoard = {
  lastUpdated: "2026-07-21",
  intro:
    "A living roundup of hackathons, game jams, internships, scholarships, events, and programs worth a look if you're in the Philippines (or online). Deadlines are Manila time unless noted.",
  items: opportunityBoardItems,
};


export function getOpportunitiesBoard() {
  return opportunitiesBoard;
}

export function getOpportunities() {
  return opportunitiesBoard.items;
}

/** @deprecated Use getOpportunities — kept for scripts during transition. */
export const opportunityIssues = [
  {
    slug: OPPORTUNITY_IMAGE_PREFIX,
    items: opportunitiesBoard.items,
  },
];

export function getIssueBySlug(slug) {
  if (slug === OPPORTUNITY_IMAGE_PREFIX) {
    return opportunityIssues[0];
  }
  return undefined;
}

export function slugifyOpportunityTitle(title) {
  return title
    .toLowerCase()
    .normalize("NFD")
    .replace(/\p{M}/gu, "")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 48);
}

/** Stable id for image filenames and manifest keys. */
export function getOpportunityId(item) {
  if (item.id) {
    return item.id;
  }
  return `${OPPORTUNITY_IMAGE_PREFIX}-${slugifyOpportunityTitle(item.title)}`;
}

/** Manual `image` wins, then UPOU MODeL shared art, then fetched og:image, then type default. */
export function resolveOpportunityImage(item) {
  if (item.image) {
    return item.image;
  }

  if (isUpouModelOpportunity(item)) {
    return UPOU_MODEL_IMAGE;
  }

  const id = getOpportunityId(item);
  const entry = loadOpportunityImages().images[id];
  if (
    (entry?.status === "fetched" ||
      entry?.status === "screenshot" ||
      entry?.status === "favicon") &&
    entry.path
  ) {
    return entry.path;
  }

  return OPPORTUNITY_UNAVAILABLE_IMAGE;
}

export function getOpportunityImagePresentation(item) {
  if (item.image || isUpouModelOpportunity(item)) {
    return { className: "" };
  }

  const id = getOpportunityId(item);
  const entry = loadOpportunityImages().images[id];

  if (entry?.status === "favicon") {
    return { className: "neo-opportunity-favicon-image" };
  }

  const resolved = resolveOpportunityImage(item);
  if (resolved === OPPORTUNITY_UNAVAILABLE_IMAGE) {
    return { className: "neo-opportunity-placeholder-image" };
  }

  const src = entry?.path ?? "";
  if (src.includes("/opportunities/defaults/")) {
    return { className: "neo-opportunity-default-image" };
  }

  return { className: "" };
}

const BEGINNER_BLURB_SIGNALS = [
  /beginner[- ]friendly/i,
  /\bbeginners\b/i,
  /no prior(?: \w+)? experience required/i,
  /no experience required/i,
  /\bon-?ramp\b/i,
  /starter for learners/i,
  /good for .+ beginners/i,
  /self-paced free courses/i,
  /lighter entry point/i,
  /without needing a formal/i,
];

const NOT_BEGINNER_BLURB = [
  /already have some/i,
  /(?:stronger )?next-step option/i,
  /CPD program/i,
  /telco pro/i,
];

const BEGINNER_TITLE_SIGNALS = [
  /^Introduction to /i,
  /^Basics of /i,
  /^Learner Support in /i,
  /\bEssentials\b/,
  /\bIT Support Professional Certificate\b/,
];

/** Explicit `beginnerFriendly` wins; otherwise infer from title/blurb. */
export function isOpportunityBeginnerFriendly(item) {
  if (item.beginnerFriendly === true) {
    return true;
  }
  if (item.beginnerFriendly === false) {
    return false;
  }

  const title = item.title ?? "";
  const blurb = item.blurb ?? "";

  if (/\bAdvanced\b/i.test(title)) {
    return false;
  }

  if (
    NOT_BEGINNER_BLURB.some((pattern) => pattern.test(blurb)) &&
    !/no prior(?: \w+)? experience required/i.test(blurb)
  ) {
    return false;
  }

  if (BEGINNER_BLURB_SIGNALS.some((pattern) => pattern.test(blurb))) {
    return true;
  }

  if (BEGINNER_TITLE_SIGNALS.some((pattern) => pattern.test(title))) {
    return true;
  }

  if (
    item.type === "certificate" &&
    /Professional Certificate/.test(title) &&
    !/IT Automation|Advanced/.test(title) &&
    /entry|foundation|students and early-career|career-shifters|side hustles|first-job|junior professionals|design-entry|structured training/i.test(
      blurb,
    )
  ) {
    return true;
  }

  if (item.type === "certificate" && /MOOC|MODeL/i.test(`${title} ${blurb}`)) {
    return true;
  }

  return false;
}

const AI_TEXT_SIGNALS = [
  /\bAI\b/,
  /artificial intelligence/i,
  /machine learning/i,
  /\bML\b/,
  /generative AI/i,
  /\bGenAI\b/i,
  /\bLLM/i,
  /large language model/i,
  /\bGPT\b/i,
  /\bGemini\b/i,
  /\bClaude\b/i,
  /\bBedrock\b/i,
  /OpenAI/i,
  /agentic/i,
  /AI agent/i,
  /deep learning/i,
  /neural network/i,
  /\bNLP\b/i,
  /natural language processing/i,
  /computer vision/i,
  /prompt engineering/i,
  /\bRAG\b/,
  /data science/i,
  /data analytics/i,
  /SAP Analytics/i,
  /MLOps/i,
  /Copilot/i,
  /diffusion model/i,
  /transformer/i,
  /fine-?tun(e|ing)/i,
];

// MIL hackathon title contains no AI tech signal; UNESCO one is MIL-focused.
const AI_NEGATIVE_SIGNALS = [
  /media and information literacy/i,
  /\bMIL\b(?!\s*(engineer|ops))/i,
];

/** Explicit `aiRelated` wins; otherwise infer from title, org, and blurb. */
export function isOpportunityAiRelated(item) {
  if (item.aiRelated === true) {
    return true;
  }
  if (item.aiRelated === false) {
    return false;
  }

  const text = `${item.title ?? ""} ${item.org ?? ""} ${item.blurb ?? ""}`;

  if (AI_NEGATIVE_SIGNALS.some((pattern) => pattern.test(text))) {
    if (!/\bAI\b|artificial intelligence|machine learning|generative|LLM|GPT|Gemini|Claude|agentic/i.test(text)) {
      return false;
    }
  }

  return AI_TEXT_SIGNALS.some((pattern) => pattern.test(text));
}

export function isOpportunityGameJam(item) {
  return item.type === "game-jam";
}

export function filterOpportunities(
  items,
  {
    aiOnly = false,
    hideAi = false,
    gameJamOnly = false,
    type = "all",
    query = "",
  } = {},
) {
  let filtered = items;

  const typeFilter = gameJamOnly ? "game-jam" : type;

  if (typeFilter !== "all") {
    filtered = filtered.filter((item) => item.type === typeFilter);
  }

  if (aiOnly) {
    filtered = filtered.filter(isOpportunityAiRelated);
  } else if (hideAi) {
    filtered = filtered.filter((item) => !isOpportunityAiRelated(item));
  }

  const normalizedQuery = query.trim().toLowerCase();
  if (normalizedQuery) {
    const terms = normalizedQuery.split(/\s+/).filter(Boolean);
    filtered = filtered.filter((item) => {
      const haystack = getOpportunitySearchText(item);
      return terms.every((term) => haystack.includes(term));
    });
  }

  return filtered;
}

function getOpportunitySearchText(item) {
  const typeLabel = OPPORTUNITY_TYPES[item.type]?.label ?? item.type ?? "";
  let host = "";

  try {
    host = new URL(item.url).hostname.replace(/^www\./, "");
  } catch {
    host = "";
  }

  return [item.title, item.org, item.blurb, item.location, typeLabel, host]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
}

const SEARCH_QUICK_TERMS = [
  "hackathon",
  "internship",
  "online",
  "beginner-friendly",
  "AI",
  "Google",
  "scholarship",
];

/**
 * @returns {{ text: string, kind: "title"|"org"|"location"|"type"|"quick" }[]}
 */
export function getOpportunitySearchSuggestions(items, query = "", limit = 8) {
  const normalized = query.trim().toLowerCase();

  if (!normalized) {
    const orgCounts = new Map();
    for (const item of items) {
      if (item.org) {
        orgCounts.set(item.org, (orgCounts.get(item.org) ?? 0) + 1);
      }
    }

    const topOrgs = [...orgCounts.entries()]
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
      .slice(0, 4)
      .map(([org]) => org);

    const quick = [...new Set([...topOrgs, ...SEARCH_QUICK_TERMS])];
    return quick.slice(0, limit).map((text) => ({ text, kind: "quick" }));
  }

  const ranked = [];
  const seen = new Set();

  function add(text, kind, priority) {
    const trimmed = text?.trim();
    if (!trimmed) {
      return;
    }

    const key = trimmed.toLowerCase();
    if (seen.has(key) || !key.includes(normalized)) {
      return;
    }

    seen.add(key);
    const startsWith = key.startsWith(normalized) ? 0 : 1;
    ranked.push({
      text: trimmed,
      kind,
      score: startsWith * 10 + priority,
    });
  }

  for (const item of items) {
    add(item.title, "title", 0);
    add(item.org, "org", 1);
    add(item.location, "location", 2);
    add(OPPORTUNITY_TYPES[item.type]?.label ?? "", "type", 3);
  }

  return ranked
    .sort(
      (a, b) =>
        a.score - b.score ||
        a.text.length - b.text.length ||
        a.text.localeCompare(b.text),
    )
    .slice(0, limit)
    .map(({ text, kind }) => ({ text, kind }));
}

export function formatBoardUpdated(date) {
  return new Date(date).toLocaleDateString("en-US", {
    month: "long",
    day: "numeric",
    year: "numeric",
    timeZone: MANILA_TZ,
  });
}

export function groupIssueItemsByType(items) {
  const groups = new Map();
  for (const item of items) {
    if (!groups.has(item.type)) {
      groups.set(item.type, []);
    }
    groups.get(item.type).push(item);
  }

  return OPPORTUNITY_TYPE_ORDER.filter((type) => groups.has(type)).map((type) => ({
    type,
    items: groups.get(type),
  }));
}

export function getOpportunityType(type) {
  return OPPORTUNITY_TYPES[type] ?? {
    label: type,
    badge: "neo-badge-program",
  };
}

const OPPORTUNITY_FORMATS = {
  online: { kind: "online", label: "Online", shortLabel: "Online" },
  hybrid: { kind: "hybrid", label: "Hybrid", shortLabel: "Hybrid" },
  onsite: { kind: "onsite", label: "In person", shortLabel: "F2F" },
};

/** Online / hybrid / in-person from free-text location field. */
export function getOpportunityFormat(location) {
  if (!location?.trim()) {
    return null;
  }

  const loc = location.trim();
  const lower = loc.toLowerCase();

  if (/\bhybrid\b/.test(lower)) {
    return OPPORTUNITY_FORMATS.hybrid;
  }

  const isOnlineish = /\b(online|remote|livestream|virtual)\b/.test(lower);
  if (!isOnlineish) {
    return OPPORTUNITY_FORMATS.onsite;
  }

  const onlineSlash = /^online\s*\/\s*(.+)$/i.exec(loc);
  if (onlineSlash) {
    const place = onlineSlash[1].trim().toLowerCase();
    if (place === "philippines" || place === "ph") {
      return OPPORTUNITY_FORMATS.online;
    }
    return OPPORTUNITY_FORMATS.hybrid;
  }

  return OPPORTUNITY_FORMATS.online;
}

/** Physical place label — omits redundant "Online" when format pill covers it. */
export function getOpportunityPlaceLabel(location) {
  if (!location?.trim()) {
    return null;
  }

  const loc = location.trim();
  if (/^(online|remote)$/i.test(loc)) {
    return null;
  }

  const stripped = loc
    .replace(/^online\s*\/\s*/i, "")
    .replace(/^hybrid\s*\/\s*/i, "")
    .replace(/\s*\/\s*hybrid\s*$/i, "")
    .replace(/\bhybrid\b\s*/gi, "")
    .replace(/^remote\s*\/\s*/i, "")
    .trim();

  if (!stripped || /^(online|remote)$/i.test(stripped)) {
    return null;
  }

  return stripped;
}

/** Nearest upcoming date, or first listed if all are past. */
export function getPrimaryOpportunityDate(dates) {
  if (!dates?.length) {
    return null;
  }
  const upcoming = dates.find((entry) => !isDatePast(entry.date));
  return upcoming ?? dates[0];
}

export function formatOpportunityDate(date, endDate) {
  const dateOpts = {
    month: "short",
    day: "numeric",
    year: "numeric",
    timeZone: MANILA_TZ,
  };
  const timeOpts = {
    ...dateOpts,
    hour: "numeric",
    minute: "2-digit",
    timeZone: MANILA_TZ,
  };

  const showTime = date.length > 10;
  const start = showTime
    ? new Date(date).toLocaleString("en-US", timeOpts)
    : new Date(date).toLocaleDateString("en-US", dateOpts);

  if (!endDate) {
    return start;
  }

  const end = new Date(endDate).toLocaleDateString("en-US", dateOpts);

  if (showTime) {
    const startDateOnly = new Date(date).toLocaleDateString("en-US", dateOpts);
    return `${startDateOnly} – ${end}`;
  }

  return `${start} – ${end}`;
}

export function isDatePast(date) {
  return new Date(date).getTime() < Date.now();
}

const DEADLINE_DATE_LABEL_RE =
  /deadline|closes?|close|due|application|registration|submission|cfp|proposals?|apply|inquiries|rsvp/i;

/** Prefer registration/apply deadlines over event dates when both exist. */
export function getOpportunityDeadlineEntry(dates) {
  if (!dates?.length) {
    return null;
  }

  const deadlineLike = dates.find((entry) =>
    DEADLINE_DATE_LABEL_RE.test(entry.label ?? ""),
  );
  if (deadlineLike) {
    return deadlineLike;
  }

  return getPrimaryOpportunityDate(dates);
}

/** End of deadline calendar day in Asia/Manila (23:59:59+08:00). */
export function getManilaDeadlineEndMs(isoDate) {
  const day = isoDate.slice(0, 10);
  return new Date(`${day}T23:59:59+08:00`).getTime();
}

function formatCountdownClock(ms) {
  const totalSeconds = Math.max(0, Math.floor(ms / 1000));
  const days = Math.floor(totalSeconds / 86400);
  if (days >= 2) {
    return `${days} days left`;
  }
  if (days === 1) {
    return "1 day left";
  }

  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  if (hours > 0) {
    return `${hours}h ${minutes}m left`;
  }
  if (minutes > 0) {
    return `${minutes}m left`;
  }
  return "Under 1m left";
}

/**
 * Snapshot for deadline UI. `nowMs` is injectable for tests and live ticks.
 * @returns {{ status: "none"|"past"|"today"|"soon"|"future", label: string, clock: string, hint: string }}
 */
export function getOpportunityDeadlineSnapshot(isoDate, nowMs = Date.now()) {
  if (!isoDate) {
    return {
      status: "none",
      label: "",
      clock: "",
      hint: "",
    };
  }

  const endMs = getManilaDeadlineEndMs(isoDate);
  const diffMs = endMs - nowMs;
  const deadlineKey = isoDate.slice(0, 10);
  const todayKey = getTodayManilaDateKey();

  if (diffMs <= 0) {
    return {
      status: "past",
      label: "Deadline passed",
      clock: "",
      hint: "Registration may already be closed. Verify on the official site.",
    };
  }

  const clock = formatCountdownClock(diffMs);

  if (deadlineKey === todayKey) {
    return {
      status: "today",
      label: "Closes today",
      clock,
      hint: "Apply now. The listing may close before midnight if slots fill up.",
    };
  }

  const daysLeft = Math.ceil(diffMs / 86400000);
  if (daysLeft <= 3) {
    return {
      status: "soon",
      label: "Closing soon",
      clock: daysLeft >= 2 ? clock : clock,
      hint: "",
    };
  }

  return {
    status: "future",
    label: "Time left",
    clock,
    hint: "",
  };
}

function toManilaDateKey(iso) {
  return new Intl.DateTimeFormat("en-CA", { timeZone: MANILA_TZ }).format(
    new Date(iso),
  );
}

function formatCalendarMonthLabel(monthKey) {
  const [year, month] = monthKey.split("-").map(Number);
  return new Date(Date.UTC(year, month - 1, 1)).toLocaleDateString("en-US", {
    month: "long",
    year: "numeric",
    timeZone: "UTC",
  });
}

export function getTodayManilaDateKey() {
  return new Intl.DateTimeFormat("en-CA", { timeZone: MANILA_TZ }).format(
    new Date(),
  );
}

export function formatCalendarDayHeading(dateKey) {
  const [year, month, day] = dateKey.split("-").map(Number);
  return new Date(Date.UTC(year, month - 1, day)).toLocaleDateString("en-US", {
    weekday: "long",
    month: "short",
    day: "numeric",
    timeZone: "UTC",
  });
}

export function groupEventsByDateKey(events) {
  const map = new Map();
  for (const event of events) {
    if (!map.has(event.dateKey)) {
      map.set(event.dateKey, []);
    }
    map.get(event.dateKey).push(event);
  }

  return [...map.entries()].sort(([a], [b]) => a.localeCompare(b));
}

/** Flatten item dates into sorted calendar events (Manila time). */
export function buildOpportunityCalendarEvents(items) {
  const events = [];

  for (const item of items) {
    if (!item.dates?.length) {
      continue;
    }

    for (const entry of item.dates) {
      if (!entry.date) {
        continue;
      }

      const dateKey = toManilaDateKey(entry.date);
      events.push({
        id: `${getOpportunityId(item)}-${entry.label}-${dateKey}`,
        dateKey,
        endDateKey: entry.endDate ? toManilaDateKey(entry.endDate) : null,
        isoDate: entry.date,
        endDate: entry.endDate ?? null,
        label: entry.label,
        title: item.title,
        url: item.url,
        type: item.type,
      });
    }
  }

  return events.sort(
    (a, b) =>
      a.dateKey.localeCompare(b.dateKey) || a.title.localeCompare(b.title),
  );
}

export function groupCalendarEventsByMonth(events) {
  const monthMap = new Map();
  for (const event of events) {
    const monthKey = event.dateKey.slice(0, 7);
    if (!monthMap.has(monthKey)) {
      monthMap.set(monthKey, []);
    }
    monthMap.get(monthKey).push(event);
  }

  return [...monthMap.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([monthKey, monthEvents]) => ({
      monthKey,
      label: formatCalendarMonthLabel(monthKey),
      events: monthEvents,
    }));
}

/** Month metadata + grid layout data for the opportunity calendar. */
export function buildCalendarMonths(events) {
  return groupCalendarEventsByMonth(events).map(
    ({ monthKey, label, events: monthEvents }) => {
      const [year, month] = monthKey.split("-").map(Number);
      return {
        monthKey,
        label,
        year,
        monthIndex: month - 1,
        events: monthEvents,
        eventsByDay: groupEventsByDay(monthEvents),
      };
    },
  );
}

/** @deprecated Use buildCalendarMonths */
export function buildIssueCalendarMonths(items) {
  return buildCalendarMonths(buildOpportunityCalendarEvents(items));
}

function groupEventsByDay(events) {
  const map = new Map();
  for (const event of events) {
    if (!map.has(event.dateKey)) {
      map.set(event.dateKey, []);
    }
    map.get(event.dateKey).push(event);
  }
  return map;
}

export function getMonthGridCells(year, monthIndex) {
  const firstWeekday = new Date(Date.UTC(year, monthIndex, 1)).getUTCDay();
  const daysInMonth = new Date(Date.UTC(year, monthIndex + 1, 0)).getUTCDate();
  const cells = [];

  for (let i = 0; i < firstWeekday; i++) {
    cells.push(null);
  }

  for (let day = 1; day <= daysInMonth; day++) {
    const month = String(monthIndex + 1).padStart(2, "0");
    const dayStr = String(day).padStart(2, "0");
    cells.push(`${year}-${month}-${dayStr}`);
  }

  return cells;
}

const CALENDAR_WEEKDAYS = ["Su", "Mo", "Tu", "We", "Th", "Fr", "Sa"];
export { CALENDAR_WEEKDAYS };
