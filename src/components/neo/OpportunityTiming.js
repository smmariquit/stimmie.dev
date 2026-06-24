"use client";

import { useEffect, useMemo, useState } from "react";
import {
  formatOpportunityDate,
  getOpportunityDeadlineEntry,
  getOpportunityDeadlineSnapshot,
  getPrimaryOpportunityDate,
  isDatePast,
} from "@/data/opportunities";

function TimingSep() {
  return (
    <span className="neo-opportunity-timing-sep" aria-hidden="true">
      ·
    </span>
  );
}

function SecondaryDateRow({ entry }) {
  const past = isDatePast(entry.date);

  return (
    <p
      className={`neo-opportunity-timing neo-opportunity-timing--secondary${past ? " neo-opportunity-timing--past" : ""}`}
    >
      <span className="neo-opportunity-timing-label">{entry.label}</span>
      <TimingSep />
      <span className="neo-opportunity-timing-date">
        {formatOpportunityDate(entry.date, entry.endDate)}
      </span>
    </p>
  );
}

export default function OpportunityTiming({ dates }) {
  const deadline = getOpportunityDeadlineEntry(dates);
  const primary = getPrimaryOpportunityDate(dates);
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    const intervalId = window.setInterval(() => setNow(Date.now()), 60_000);
    return () => window.clearInterval(intervalId);
  }, []);

  const snapshot = useMemo(
    () =>
      deadline ? getOpportunityDeadlineSnapshot(deadline.date, now) : null,
    [deadline, now],
  );

  const showDeadline =
    deadline && snapshot && snapshot.status !== "none";
  const sameAsPrimary =
    primary &&
    deadline &&
    primary.date === deadline.date &&
    primary.label === deadline.label;
  const showPrimary =
    primary &&
    !sameAsPrimary &&
    (!deadline || primary.date !== deadline.date);

  if (!showDeadline && !showPrimary) {
    return null;
  }

  const deadlineDate = deadline
    ? formatOpportunityDate(deadline.date, deadline.endDate)
    : "";

  return (
    <div
      className="neo-opportunity-card-timing"
      role="group"
      aria-label="Important dates"
    >
      {showDeadline ? (
        <p
          className={`neo-opportunity-timing neo-opportunity-timing--${snapshot.status}`}
          role="status"
          aria-live="polite"
          aria-atomic="true"
        >
          {snapshot.status === "past" ? (
            <>
              <span className="neo-opportunity-timing-label">
                {deadline.label}
              </span>
              <TimingSep />
              <span className="neo-opportunity-timing-date">
                {deadlineDate}
              </span>
              <TimingSep />
              <span className="neo-opportunity-timing-status">Passed</span>
            </>
          ) : snapshot.status === "today" || snapshot.status === "soon" ? (
            <>
              <span className="neo-opportunity-timing-status">
                {snapshot.label}
              </span>
              <TimingSep />
              <span className="neo-opportunity-timing-label">
                {deadline.label}
              </span>
              {snapshot.clock ? (
                <>
                  <TimingSep />
                  <span className="neo-opportunity-timing-countdown">
                    {snapshot.clock}
                  </span>
                </>
              ) : null}
            </>
          ) : (
            <>
              <span className="neo-opportunity-timing-label">
                {deadline.label}
              </span>
              <TimingSep />
              <span className="neo-opportunity-timing-date">
                {deadlineDate}
              </span>
              {snapshot.clock ? (
                <>
                  <TimingSep />
                  <span className="neo-opportunity-timing-countdown">
                    {snapshot.clock}
                  </span>
                </>
              ) : null}
            </>
          )}
        </p>
      ) : null}

      {showPrimary ? <SecondaryDateRow entry={primary} /> : null}

      {showDeadline && snapshot.hint ? (
        <p className="neo-opportunity-timing-hint m-0">{snapshot.hint}</p>
      ) : null}
    </div>
  );
}
