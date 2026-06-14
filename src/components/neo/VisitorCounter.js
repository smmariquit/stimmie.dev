"use client";

import { useEffect, useState } from "react";

// Free, no-auth hosted hit counter (https://abacus.jasoncameron.dev).
// `/hit` increments and returns the new value; `/get` reads without incrementing.
const NAMESPACE = "stimmie-dev";
const KEY = "site-visits";
const BASE = "https://abacus.jasoncameron.dev";
const SESSION_FLAG = "neo-counted";

function pad(value, length) {
  return String(value).padStart(length, "0");
}

export default function VisitorCounter({
  digits = 6,
  label = "you are visitor #",
}) {
  const [count, setCount] = useState(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let active = true;

    // Count a visit once per browser session; otherwise just read the total.
    let alreadyCounted = false;
    try {
      alreadyCounted = sessionStorage.getItem(SESSION_FLAG) === "1";
    } catch {
      // sessionStorage may be unavailable (private mode); fall through to /hit.
    }

    const endpoint = `${BASE}/${alreadyCounted ? "get" : "hit"}/${NAMESPACE}/${KEY}`;

    fetch(endpoint, { cache: "no-store" })
      .then((res) =>
        res.ok ? res.json() : Promise.reject(new Error("bad status")),
      )
      .then((data) => {
        if (!active) return;
        const value = data?.value ?? data?.count;
        if (typeof value === "number") {
          setCount(value);
          try {
            sessionStorage.setItem(SESSION_FLAG, "1");
          } catch {
            /* ignore */
          }
        } else {
          setFailed(true);
        }
      })
      .catch(() => {
        if (active) setFailed(true);
      });

    return () => {
      active = false;
    };
  }, []);

  let display;
  if (count != null) display = pad(count, digits);
  else if (failed) display = "?".repeat(digits);
  else display = "0".repeat(digits);

  const announce =
    count != null
      ? `You are visitor number ${count.toLocaleString()}.`
      : failed
        ? "Visitor count unavailable."
        : "Loading visitor count.";

  return (
    <div>
      <p
        className="text-xl mb-2 text-center"
        style={{ fontFamily: "var(--neo-pixel)", color: "#1a1a1a" }}
      >
        {label}
      </p>
      <p
        className="neo-counter"
        role="status"
        aria-live="polite"
        aria-label={announce}
      >
        {display.split("").map((char, i) => (
          <span key={i} className="neo-counter-digit" aria-hidden="true">
            {char}
          </span>
        ))}
      </p>
    </div>
  );
}
