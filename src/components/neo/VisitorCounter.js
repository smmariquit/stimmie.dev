"use client";

import { fetchVisitorCount } from "@/lib/visitorCount";
import { useEffect, useState } from "react";

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

    fetchVisitorCount()
      .then((value) => {
        if (active) setCount(value);
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
      <p className="neo-counter-label">{label}</p>
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
