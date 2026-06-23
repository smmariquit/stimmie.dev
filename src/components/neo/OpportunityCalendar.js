import {
  buildOpportunityCalendarEvents,
  formatOpportunityDate,
  getOpportunityType,
  groupCalendarEventsByMonth,
  isDatePast,
} from "@/data/opportunities";

function CalendarEventRow({ event }) {
  const type = getOpportunityType(event.type);
  const past = isDatePast(event.isoDate);
  const dateText = formatOpportunityDate(event.isoDate, event.endDate);

  return (
    <li
      className={`neo-cal-event${past ? " neo-cal-event--past" : ""}`}
    >
      <time className="neo-cal-event-when" dateTime={event.dateKey}>
        {dateText}
      </time>
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

export default function OpportunityCalendar({ items }) {
  const allEvents = buildOpportunityCalendarEvents(items);
  if (!allEvents.length) {
    return null;
  }

  const months = groupCalendarEventsByMonth(allEvents);

  return (
    <section
      className="neo-opportunity-calendar"
      aria-labelledby="opportunity-calendar-heading"
    >
      <h2 id="opportunity-calendar-heading" className="neo-cal-heading">
        Dates &amp; deadlines
      </h2>
      <p className="neo-cal-intro neo-muted">
        Every labeled date from the listings above, in Manila time (UTC+8).
        Click a title to open the official page — then double-check the deadline
        there.
      </p>

      <div className="neo-cal-months">
        {months.map((month) => (
          <div key={month.monthKey} className="neo-cal-month">
            <h3 className="neo-cal-month-title">{month.label}</h3>
            <ul className="neo-cal-event-list list-none p-0 m-0">
              {month.events.map((event) => (
                <CalendarEventRow key={event.id} event={event} />
              ))}
            </ul>
          </div>
        ))}
      </div>
    </section>
  );
}
