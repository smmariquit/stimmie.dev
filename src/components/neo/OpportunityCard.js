import OpportunityCoverImage from "@/components/neo/OpportunityCoverImage";
import OpportunityTiming from "@/components/neo/OpportunityTiming";
import {
  getOpportunityFormat,
  getOpportunityImagePresentation,
  getOpportunityPlaceLabel,
  getOpportunityType,
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

export default function OpportunityCard({ item }) {
  const type = getOpportunityType(item.type);
  const format = getOpportunityFormat(item.location);
  const place = getOpportunityPlaceLabel(item.location);
  const imageSrc = resolveOpportunityImage(item);
  const imagePresentation = getOpportunityImagePresentation(item);
  const beginnerFriendly = isOpportunityBeginnerFriendly(item);
  const aiRelated = isOpportunityAiRelated(item);
  const hasMeta = format || place || beginnerFriendly || aiRelated;

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
          <OpportunityCoverImage
            src={imageSrc}
            alt={item.imageAlt || item.title}
            className={imagePresentation.className}
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

        <div className="neo-opportunity-card-body">
          <header className="neo-opportunity-card-head">
            <h3 className="neo-opportunity-card-title m-0">{item.title}</h3>
            {item.org ? (
              <p className="neo-opportunity-org m-0">{item.org}</p>
            ) : null}
          </header>

          {hasMeta ? (
            <div className="neo-opportunity-location-row">
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
          ) : null}

          <OpportunityTiming dates={item.dates} />

          {item.blurb ? (
            <p className="neo-opportunity-blurb m-0">{item.blurb}</p>
          ) : null}
        </div>
      </article>
    </a>
  );
}
