// src/lib/visitorCount.js

// Free, no-auth hosted hit counter (https://abacus.jasoncameron.dev).
// `/hit` increments and returns the new value; `/get` reads without incrementing.

const NAMESPACE = "stimmie-dev";
const KEY = "site-visits";
const BASE = "https://abacus.jasoncameron.dev";
const STORAGE_KEY = "neo-counted-date";

let sharedPromise = null;

function getTodayKey() {
  const now = new Date();
  const month = String(now.getMonth() + 1).padStart(2, "0");
  const day = String(now.getDate()).padStart(2, "0");
  return `${now.getFullYear()}-${month}-${day}`;
}

function alreadyCountedToday() {
  try {
    return localStorage.getItem(STORAGE_KEY) === getTodayKey();
  } catch {
    return false;
  }
}

function markCountedToday() {
  try {
    localStorage.setItem(STORAGE_KEY, getTodayKey());
  } catch {
    /* private mode / blocked storage */
  }
}

async function requestCount(increment) {
  const action = increment ? "hit" : "get";
  const res = await fetch(`${BASE}/${action}/${NAMESPACE}/${KEY}`, {
    cache: "no-store",
  });
  if (!res.ok) throw new Error("bad status");
  const data = await res.json();
  const value = data?.value ?? data?.count;
  if (typeof value !== "number") throw new Error("bad payload");
  return value;
}

/**
 * One increment per browser per calendar day; refreshes the same day only read.
 * Dedupes concurrent calls when sidebar + footer both mount the counter.
 */
export function fetchVisitorCount() {
  if (!sharedPromise) {
    sharedPromise = (async () => {
      const increment = !alreadyCountedToday();
      const value = await requestCount(increment);
      if (increment) markCountedToday();
      return value;
    })().catch((err) => {
      sharedPromise = null;
      throw err;
    });
  }
  return sharedPromise;
}
