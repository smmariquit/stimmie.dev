// src/data/opportunities.js
//
// Curated board for /opportunities — one living page, not issues.
// Bump `lastUpdated` when you add, remove, or materially edit entries.
//
// Research pipeline: research/opportunities/
// Optional `image` overrides everything. Optional `imageUrl` for image fetch only.

import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { opportunityBoardItems } from "./opportunity-board-items.js";

const MANILA_TZ = "Asia/Manila";
/** Matches existing cover-image filenames and manifest keys. */
const OPPORTUNITY_IMAGE_PREFIX = "q3-2026";
const manifestPath = join(
  dirname(fileURLToPath(import.meta.url)),
  "opportunity-images.json",
);

function loadOpportunityImages() {
  if (!existsSync(manifestPath)) {
    return { images: {} };
  }
  return JSON.parse(readFileSync(manifestPath, "utf8"));
}

export const OPPORTUNITY_TYPE_DEFAULT_IMAGES = {
  hackathon: "/opportunities/defaults/hackathon.png",
  internship: "/opportunities/defaults/internship.png",
  event: "/opportunities/defaults/event.png",
  certificate: "/opportunities/defaults/certificate.svg",
  program: "/opportunities/defaults/program.jpg",
};

export const OPPORTUNITY_TYPES = {
  hackathon: { label: "Hackathon", badge: "neo-badge-hackathon" },
  internship: { label: "Internship", badge: "neo-badge-internship" },
  event: { label: "Event", badge: "neo-badge-event" },
  certificate: { label: "Certificate", badge: "neo-badge-certificate" },
  program: { label: "Program", badge: "neo-badge-program" },
};

/** Display order for grouped issue pages. */
export const OPPORTUNITY_TYPE_ORDER = [
  "hackathon",
  "internship",
  "program",
  "event",
  "certificate",
];

export const opportunitiesBoard = {
  lastUpdated: "2026-06-24",
  intro:
    "A living roundup of hackathons, internships, scholarships, events, and programs worth a look if you're in the Philippines (or online). Deadlines are Manila time unless noted.",
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

/** Manual `image` wins, then fetched og:image, then type default. */
export function resolveOpportunityImage(item) {
  if (item.image) {
    return item.image;
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

  return (
    OPPORTUNITY_TYPE_DEFAULT_IMAGES[item.type] ??
    OPPORTUNITY_TYPE_DEFAULT_IMAGES.program
  );
}

export function getOpportunityImagePresentation(item) {
  if (item.image) {
    return { className: "" };
  }

  const id = getOpportunityId(item);
  const entry = loadOpportunityImages().images[id];

  if (entry?.status === "favicon") {
    return { className: "neo-opportunity-favicon-image" };
  }

  const src = entry?.path ?? "";
  if (src.includes("/opportunities/defaults/")) {
    return { className: "neo-opportunity-default-image" };
  }

  return { className: "" };
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

/** @deprecated Use buildOpportunityCalendarEvents */
export function buildIssueCalendarMonths(items) {
  const events = buildOpportunityCalendarEvents(items);
  return groupCalendarEventsByMonth(events).map(({ monthKey, label, events: monthEvents }) => {
    const [year, month] = monthKey.split("-").map(Number);
    return {
      monthKey,
      label,
      year,
      monthIndex: month - 1,
      events: monthEvents,
      eventsByDay: groupEventsByDay(monthEvents),
    };
  });
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
