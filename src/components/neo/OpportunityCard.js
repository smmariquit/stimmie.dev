import Image from "next/image";
import OpportunityDeadlineTicker from "@/components/neo/OpportunityDeadlineTicker";
import {
  formatOpportunityDate,
  getOpportunityFormat,
  getOpportunityImagePresentation,
  getOpportunityPlaceLabel,
  getOpportunityType,
  getPrimaryOpportunityDate,
  isDatePast,
  isOpportunityBeginnerFriendly,
  isOpportunityAiRelated,
  resolveOpportunityImage,
} from "@/data/opportunities";

function FormatPill({ format, className = "" }) {
  if (!format) {
    return null;
  }

  return (
    <span
      className={`neo-format-pill neo-format-${format.kind} ${className}`.trim()}
      title={format.label}
    >
      <span className="neo-format-dot" aria-hidden="true" />
      {format.shortLabel}
    </span>
  );
}

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

export default function OpportunityCard({ item }) {
  const type = getOpportunityType(item.type);
  const format = getOpportunityFormat(item.location);
  const place = getOpportunityPlaceLabel(item.location);
  const imageSrc = resolveOpportunityImage(item);
  const imagePresentation = getOpportunityImagePresentation(item);
  const beginnerFriendly = isOpportunityBeginnerFriendly(item);
  const aiRelated = isOpportunityAiRelated(item);

  return (
    <a
      href={item.url}
      target="_blank"
      rel="noopener noreferrer"
      className={`neo-media-card neo-opportunity-card neo-opportunity-card--${item.type} group block h-full`}
      aria-label={`Open opportunity: ${item.title}${format ? ` (${format.label})` : ""}${aiRelated ? ", AI-related" : ""}${beginnerFriendly ? ", beginner-friendly" : ""}`}
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
            className={`neo-thumb-lg w-full aspect-video object-cover${imagePresentation.className ? ` ${imagePresentation.className}` : ""}`}
          />
          <span
            className={`neo-badge neo-opportunity-badge ${type.badge}`}
            aria-hidden="true"
          >
            {type.label}
          </span>
          {format ? (
            <FormatPill
              format={format}
              className="neo-opportunity-format-badge"
            />
          ) : null}
        </div>

        <div className="mt-2">
          <p className="m-0 font-bold text-lg leading-snug group-hover:text-[#cc0066]">
            {item.title}
          </p>

          {item.org ? (
            <p className="m-0 mt-1 neo-muted neo-opportunity-org">{item.org}</p>
          ) : null}

          {(format || place || beginnerFriendly || aiRelated) && (
            <div className="neo-opportunity-location-row m-0 mt-1.5">
              {place ? (
                <span className="neo-location-pill" title={place}>
                  <span className="neo-location-icon" aria-hidden="true">
                    ⌖
                  </span>
                  <span className="neo-location-text">{place}</span>
                </span>
              ) : format?.kind === "online" ? (
                <span className="neo-location-pill neo-location-pill--online">
                  <span className="neo-location-icon" aria-hidden="true">
                    ⊕
                  </span>
                  <span className="neo-location-text">Worldwide</span>
                </span>
              ) : null}
              {aiRelated ? (
                <span className="neo-ai-pill" title="AI-related">
                  AI
                </span>
              ) : null}
              {beginnerFriendly ? (
                <span className="neo-beginner-pill" title="Beginner-friendly">
                  Beginner-friendly
                </span>
              ) : null}
            </div>
          )}

          <OpportunityPrimaryDate dates={item.dates} />

          <OpportunityDeadlineTicker dates={item.dates} />

          {item.blurb ? (
            <p className="neo-opportunity-blurb m-0 mt-1.5 leading-relaxed">
              {item.blurb}
            </p>
          ) : null}
        </div>
      </article>
    </a>
  );
}
