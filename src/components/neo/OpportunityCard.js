import Image from "next/image";
import {
  formatOpportunityDate,
  getOpportunityType,
  isDatePast,
} from "@/data/opportunities";

function OpportunityDates({ dates }) {
  if (!dates?.length) {
    return null;
  }

  return (
    <ul className="neo-opportunity-dates list-none p-0 m-0 mt-2">
      {dates.map((entry) => {
        const past = isDatePast(entry.date);
        const value = formatOpportunityDate(entry.date, entry.endDate);

        return (
          <li
            key={`${entry.label}-${entry.date}`}
            className={past ? "neo-opportunity-date-past" : undefined}
          >
            <span className="neo-opportunity-date-label">{entry.label}</span>
            <span className="neo-opportunity-date-value">{value}</span>
          </li>
        );
      })}
    </ul>
  );
}

export default function OpportunityCard({ item }) {
  const type = getOpportunityType(item.type);
  const cardClass = item.image ? "neo-media-card group block h-full" : "neo-link-card group block";

  return (
    <a
      href={item.url}
      target="_blank"
      rel="noopener noreferrer"
      className={cardClass}
      aria-label={`Open opportunity: ${item.title}`}
    >
      <article>
        {item.image ? (
          <Image
            src={item.image}
            alt={item.imageAlt || item.title}
            width={800}
            height={450}
            quality={90}
            sizes="(max-width: 640px) 100vw, 360px"
            className="neo-thumb-lg w-full aspect-video object-cover"
          />
        ) : null}

        <div className={item.image ? "mt-2" : undefined}>
          <p className="m-0 flex flex-wrap items-center gap-2">
            <span className={`neo-badge ${type.badge}`}>{type.label}</span>
          </p>

          <p className="m-0 mt-1.5 font-bold text-xl group-hover:text-[#cc0066]">
            {item.title}
          </p>

          {(item.org || item.location) && (
            <p
              className="m-0 mt-1 text-base neo-muted"
              style={{ fontFamily: "var(--neo-ui)" }}
            >
              {[item.org, item.location].filter(Boolean).join(" · ")}
            </p>
          )}

          <OpportunityDates dates={item.dates} />

          {item.blurb ? (
            <p className="m-0 mt-2 text-base leading-relaxed">{item.blurb}</p>
          ) : null}
        </div>
      </article>
    </a>
  );
}
