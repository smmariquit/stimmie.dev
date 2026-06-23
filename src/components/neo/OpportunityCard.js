import Image from "next/image";
import {
  formatOpportunityDate,
  getOpportunityType,
  getPrimaryOpportunityDate,
  isDatePast,
  resolveOpportunityImage,
} from "@/data/opportunities";

function OpportunityPrimaryDate({ dates }) {
  const primary = getPrimaryOpportunityDate(dates);
  if (!primary) {
    return null;
  }

  const past = isDatePast(primary.date);
  const value = formatOpportunityDate(primary.date, primary.endDate);

  return (
    <p
      className={`neo-opportunity-primary-date m-0 mt-1.5${past ? " neo-opportunity-date-past" : ""}`}
    >
      <span className="neo-opportunity-date-label">{primary.label}</span>
      <span className="neo-opportunity-date-value">{value}</span>
    </p>
  );
}

export default function OpportunityCard({ item, issueSlug }) {
  const type = getOpportunityType(item.type);
  const imageSrc = resolveOpportunityImage(issueSlug, item);
  const isDefaultImage = imageSrc.includes("/opportunities/defaults/");

  return (
    <a
      href={item.url}
      target="_blank"
      rel="noopener noreferrer"
      className={`neo-media-card neo-opportunity-card neo-opportunity-card--${item.type} group block h-full`}
      aria-label={`Open opportunity: ${item.title}`}
    >
      <article>
        <div className="neo-opportunity-thumb relative">
          <Image
            src={imageSrc}
            alt={item.imageAlt || item.title}
            width={800}
            height={450}
            quality={90}
            sizes="(max-width: 640px) 100vw, 360px"
            className={`neo-thumb-lg w-full aspect-video object-cover${isDefaultImage ? " neo-opportunity-default-image" : ""}`}
          />
          <span
            className={`neo-badge neo-opportunity-badge ${type.badge}`}
            aria-hidden="true"
          >
            {type.label}
          </span>
        </div>

        <div className="mt-2">
          <p className="m-0 font-bold text-lg leading-snug group-hover:text-[#cc0066]">
            {item.title}
          </p>

          {(item.org || item.location) && (
            <p
              className="m-0 mt-1 text-sm neo-muted neo-opportunity-meta"
              style={{ fontFamily: "var(--neo-ui)" }}
            >
              {[item.org, item.location].filter(Boolean).join(" · ")}
            </p>
          )}

          <OpportunityPrimaryDate dates={item.dates} />

          {item.blurb ? (
            <p className="neo-opportunity-blurb m-0 mt-1.5 text-sm leading-relaxed">
              {item.blurb}
            </p>
          ) : null}
        </div>
      </article>
    </a>
  );
}
