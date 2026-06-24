"use client";

import { useEffect, useMemo, useState } from "react";
import {
  getOpportunityDeadlineEntry,
  getOpportunityDeadlineSnapshot,
} from "@/data/opportunities";

export default function OpportunityDeadlineTicker({ dates }) {
  const deadline = getOpportunityDeadlineEntry(dates);
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    const intervalId = window.setInterval(() => setNow(Date.now()), 60_000);
    return () => window.clearInterval(intervalId);
  }, []);

  const snapshot = useMemo(
    () => getOpportunityDeadlineSnapshot(deadline?.date, now),
    [deadline, now],
  );

  if (!deadline || snapshot.status === "none") {
    return null;
  }

  return (
    <div
      className={`neo-opportunity-deadline-ticker neo-opportunity-deadline-ticker--${snapshot.status}`}
      role="status"
      aria-live="polite"
      aria-atomic="true"
    >
      <p className="neo-opportunity-deadline-ticker-row m-0 mt-1.5">
        <span className="neo-opportunity-deadline-ticker-label">
          {snapshot.label}
          {deadline.label ? (
            <span className="neo-opportunity-deadline-ticker-context">
              {" "}
              ({deadline.label})
            </span>
          ) : null}
        </span>
        {snapshot.clock ? (
          <span className="neo-opportunity-deadline-ticker-clock">
            {snapshot.clock}
          </span>
        ) : null}
      </p>
      {snapshot.hint ? (
        <p className="neo-opportunity-deadline-ticker-hint m-0 mt-1">
          {snapshot.hint}
        </p>
      ) : null}
    </div>
  );
}
