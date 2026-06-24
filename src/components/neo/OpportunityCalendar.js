import {
  buildCalendarMonths,
  buildOpportunityCalendarEvents,
  CALENDAR_WEEKDAYS,
  formatCalendarDayHeading,
  getMonthGridCells,
  getOpportunityType,
  getTodayManilaDateKey,
  groupEventsByDateKey,
  isDatePast,
} from "@/data/opportunities";

const GRID_CHIP_LIMIT = 2;

function truncateTitle(title, max = 42) {
  if (title.length <= max) {
    return title;
  }
  return `${title.slice(0, max - 1)}…`;
}

function CalendarEventChip({ event }) {
  const type = getOpportunityType(event.type);
  const past = isDatePast(event.isoDate);

  return (
    <a
      href={event.url}
      target="_blank"
      rel="noopener noreferrer"
      className={`neo-cal-chip neo-cal-chip--${event.type}${past ? " neo-cal-chip--past" : ""}`}
      title={`${event.title}: ${event.label}`}
    >
      <span className={`neo-cal-chip-type ${type.badge}`}>{type.label}</span>
      <span className="neo-cal-chip-title">{truncateTitle(event.title)}</span>
    </a>
  );
}

function CalendarGrid({ month, todayKey }) {
  const cells = getMonthGridCells(month.year, month.monthIndex);

  return (
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
        const visibleEvents = dayEvents.slice(0, GRID_CHIP_LIMIT);
        const hiddenCount = dayEvents.length - visibleEvents.length;

        return (
          <div
            key={dateKey}
            className={`neo-cal-day${dayEvents.length ? " neo-cal-day--has-events" : ""}${isToday ? " neo-cal-day--today" : ""}${isPast ? " neo-cal-day--past" : ""}`}
            role="gridcell"
            aria-label={
              dayEvents.length
                ? `${formatCalendarDayHeading(dateKey)}: ${dayEvents.map((event) => event.title).join(", ")}`
                : formatCalendarDayHeading(dateKey)
            }
          >
            <span className="neo-cal-day-num">{dayNum}</span>
            {dayEvents.length > 0 ? (
              <div className="neo-cal-chips">
                {visibleEvents.map((event) => (
                  <CalendarEventChip key={event.id} event={event} />
                ))}
                {hiddenCount > 0 ? (
                  <a
                    href={`#cal-day-${dateKey}`}
                    className="neo-cal-chip-more"
                  >
                    +{hiddenCount} more
                  </a>
                ) : null}
              </div>
            ) : null}
          </div>
        );
      })}
    </div>
  );
}

function CalendarEventRow({ event }) {
  const type = getOpportunityType(event.type);
  const past = isDatePast(event.isoDate);

  return (
    <li
      className={`neo-cal-event${past ? " neo-cal-event--past" : ""}`}
    >
      <div className="neo-cal-event-body">
        <a
          href={event.url}
          target="_blank"
          rel="noopener noreferrer"
          className="neo-cal-event-link group"
        >
          <span className={`neo-badge ${type.badge}`}>{type.label}</span>
          <span className="neo-cal-event-title">{event.title}</span>
        </a>
        <p className="neo-cal-event-label m-0">{event.label}</p>
      </div>
    </li>
  );
}

function DayGroup({ dateKey, events }) {
  const past = events.every((event) => isDatePast(event.isoDate));

  return (
    <div
      id={`cal-day-${dateKey}`}
      className={`neo-cal-day-group${past ? " neo-cal-day-group--past" : ""}`}
    >
      <h4 className="neo-cal-day-heading">
        <time dateTime={dateKey}>{formatCalendarDayHeading(dateKey)}</time>
        <span className="neo-cal-day-count">
          {events.length}{" "}
          {events.length === 1 ? "deadline" : "deadlines"}
        </span>
      </h4>
      <ul className="neo-cal-event-list list-none p-0 m-0">
        {events.map((event) => (
          <CalendarEventRow key={event.id} event={event} />
        ))}
      </ul>
    </div>
  );
}

function MonthSection({ month, todayKey }) {
  const dayGroups = groupEventsByDateKey(month.events);

  return (
    <section
      className="neo-cal-month"
      aria-labelledby={`cal-month-${month.monthKey}`}
    >
      <h3 id={`cal-month-${month.monthKey}`} className="neo-cal-month-title">
        {month.label}
        <span className="neo-cal-month-count">{month.events.length}</span>
      </h3>

      <CalendarGrid month={month} todayKey={todayKey} />

      <div className="neo-cal-agenda-label-row">
        <h4 className="neo-cal-agenda-subtitle">All dates this month</h4>
      </div>
      <div className="neo-cal-day-groups">
        {dayGroups.map(([dateKey, events]) => (
          <DayGroup key={dateKey} dateKey={dateKey} events={events} />
        ))}
      </div>
    </section>
  );
}

export default function OpportunityCalendar({ items }) {
  const allEvents = buildOpportunityCalendarEvents(items);
  if (!allEvents.length) {
    return null;
  }

  const todayKey = getTodayManilaDateKey();
  const upcoming = allEvents
    .filter((event) => event.dateKey >= todayKey)
    .slice(0, 8);
  const months = buildCalendarMonths(allEvents);

  return (
    <details className="neo-opportunity-calendar neo-cal-disclosure" open>
      <summary className="neo-cal-disclosure-summary">
        <span className="neo-cal-disclosure-title-wrap">
          <h2 id="opportunity-calendar-heading" className="neo-cal-heading m-0">
            Dates &amp; deadlines
          </h2>
          <span className="neo-cal-disclosure-meta">
            {allEvents.length} dated {allEvents.length === 1 ? "entry" : "entries"}
          </span>
        </span>
        <span className="neo-cal-disclosure-toggle" aria-hidden="true" />
      </summary>

      <div className="neo-cal-disclosure-body">
      <p className="neo-cal-intro neo-muted">
        Month view plus a full list below (Manila time, UTC+8). Chips in the
        grid show titles. Click any entry to open the official page and
        confirm the deadline there.
      </p>

      {upcoming.length > 0 ? (
        <section
          className="neo-cal-upcoming"
          aria-labelledby="opportunity-calendar-upcoming"
        >
          <h3
            id="opportunity-calendar-upcoming"
            className="neo-cal-upcoming-title"
          >
            Coming up next
          </h3>
          <ul className="neo-cal-upcoming-list list-none p-0 m-0">
            {upcoming.map((event) => {
              const type = getOpportunityType(event.type);
              const past = isDatePast(event.isoDate);

              return (
                <li
                  key={event.id}
                  className={`neo-cal-upcoming-item${past ? " neo-cal-upcoming-item--past" : ""}`}
                >
                  <time
                    className="neo-cal-upcoming-when"
                    dateTime={event.dateKey}
                  >
                    {formatCalendarDayHeading(event.dateKey)}
                  </time>
                  <a
                    href={event.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="neo-cal-upcoming-link group"
                  >
                    <span className={`neo-badge ${type.badge}`}>
                      {type.label}
                    </span>
                    <span className="neo-cal-upcoming-title-text">
                      {event.title}
                    </span>
                  </a>
                  <span className="neo-cal-upcoming-label">{event.label}</span>
                </li>
              );
            })}
          </ul>
        </section>
      ) : null}

      <div className="neo-cal-months">
        {months.map((month) => (
          <MonthSection key={month.monthKey} month={month} todayKey={todayKey} />
        ))}
      </div>
      </div>
    </details>
  );
}
