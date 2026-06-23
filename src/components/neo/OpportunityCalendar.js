import {
  CALENDAR_WEEKDAYS,
  buildIssueCalendarMonths,
  formatOpportunityDate,
  getMonthGridCells,
  getOpportunityType,
  isDatePast,
} from "@/data/opportunities";

function CalendarMonth({ month }) {
  const cells = getMonthGridCells(month.year, month.monthIndex);
  const todayKey = new Intl.DateTimeFormat("en-CA", {
    timeZone: "Asia/Manila",
  }).format(new Date());

  return (
    <div className="neo-cal-month">
      <h3 className="neo-cal-month-title">{month.label}</h3>

      <div className="neo-cal-grid" role="grid" aria-label={month.label}>
        {CALENDAR_WEEKDAYS.map((day) => (
          <div
            key={day}
            className="neo-cal-weekday"
            role="columnheader"
            aria-hidden="true"
          >
            {day}
          </div>
        ))}

        {cells.map((dateKey, index) => {
          if (!dateKey) {
            return (
              <div
                key={`pad-${month.monthKey}-${index}`}
                className="neo-cal-day neo-cal-day--empty"
                aria-hidden="true"
              />
            );
          }

          const dayEvents = month.eventsByDay.get(dateKey) ?? [];
          const dayNum = Number(dateKey.slice(8, 10));
          const isToday = dateKey === todayKey;
          const isPast =
            dayEvents.length > 0 &&
            dayEvents.every((event) => isDatePast(event.isoDate));

          return (
            <div
              key={dateKey}
              className={`neo-cal-day${dayEvents.length ? " neo-cal-day--has-events" : ""}${isToday ? " neo-cal-day--today" : ""}${isPast ? " neo-cal-day--past" : ""}`}
              role="gridcell"
            >
              <span className="neo-cal-day-num">{dayNum}</span>
              {dayEvents.length > 0 ? (
                <div className="neo-cal-dots" aria-hidden="true">
                  {dayEvents.slice(0, 3).map((event) => (
                    <span
                      key={event.id}
                      className={`neo-cal-dot neo-cal-dot--${event.type}`}
                    />
                  ))}
                  {dayEvents.length > 3 ? (
                    <span className="neo-cal-dot-more">
                      +{dayEvents.length - 3}
                    </span>
                  ) : null}
                </div>
              ) : null}
            </div>
          );
        })}
      </div>

      <ul className="neo-cal-agenda list-none p-0 m-0">
        {month.events.map((event) => {
          const type = getOpportunityType(event.type);
          const past = isDatePast(event.isoDate);
          const dateText = formatOpportunityDate(event.isoDate, event.endDate);

          return (
            <li
              key={event.id}
              className={`neo-cal-agenda-item${past ? " neo-cal-agenda-item--past" : ""}`}
            >
              <time
                className="neo-cal-agenda-date"
                dateTime={event.dateKey}
              >
                {dateText}
              </time>
              <a
                href={event.url}
                target="_blank"
                rel="noopener noreferrer"
                className="neo-cal-agenda-link group"
              >
                <span className={`neo-badge ${type.badge}`}>{type.label}</span>
                <span className="neo-cal-agenda-title">{event.title}</span>
                <span className="neo-cal-agenda-label">{event.label}</span>
              </a>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

export default function OpportunityCalendar({ issueSlug, items }) {
  const months = buildIssueCalendarMonths(items, issueSlug);

  if (!months.length) {
    return null;
  }

  const typesInIssue = [
    ...new Set(months.flatMap((month) => month.events.map((event) => event.type))),
  ].sort(
    (a, b) =>
      ["hackathon", "internship", "program", "event", "certificate"].indexOf(
        a,
      ) -
      ["hackathon", "internship", "program", "event", "certificate"].indexOf(b),
  );

  return (
    <section
      className="neo-opportunity-calendar"
      aria-labelledby="opportunity-calendar-heading"
    >
      <h2 id="opportunity-calendar-heading" className="neo-cal-heading">
        Calendar
      </h2>
      <p className="neo-cal-intro neo-muted">
        Key dates in Manila time (UTC+8). Click an entry to open the official
        page.
      </p>

      <div className="neo-cal-legend" aria-label="Opportunity types">
        {typesInIssue.map((type) => {
          const info = getOpportunityType(type);
          return (
            <span key={type} className="neo-cal-legend-item">
              <span
                className={`neo-cal-dot neo-cal-dot--${type}`}
                aria-hidden="true"
              />
              {info.label}
            </span>
          );
        })}
      </div>

      <div className="neo-cal-months">
        {months.map((month) => (
          <CalendarMonth key={month.monthKey} month={month} />
        ))}
      </div>
    </section>
  );
}
