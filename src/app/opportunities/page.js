import OpportunityCalendar from "@/components/neo/OpportunityCalendar";
import OpportunityCard from "@/components/neo/OpportunityCard";
import OpportunityDisclaimer from "@/components/neo/OpportunityDisclaimer";
import PageShell from "@/components/neo/PageShell";
import {
  formatBoardUpdated,
  getOpportunities,
  getOpportunitiesBoard,
  getOpportunityType,
  groupIssueItemsByType,
} from "@/data/opportunities";

export const metadata = {
  title: "Opportunities",
  description:
    "A curated, incomplete roundup of hackathons, internships, events, and programs for students and early-career builders in the Philippines.",
};

export default function OpportunitiesPage() {
  const board = getOpportunitiesBoard();
  const items = getOpportunities();
  const sections = groupIssueItemsByType(items);

  return (
    <PageShell
      title="~ opportunities ~"
      intro={board.intro}
      current="/opportunities"
      maxWidth="64rem"
    >
      <p
        className="neo-opportunity-updated m-0 mb-4 text-sm neo-muted"
        style={{ fontFamily: "var(--neo-ui)" }}
      >
        <strong>Last updated:</strong>{" "}
        <time dateTime={board.lastUpdated}>
          {formatBoardUpdated(board.lastUpdated)}
        </time>
        {" · "}
        {items.length} listings
      </p>

      <OpportunityDisclaimer />

      <div className="neo-opportunity-sections mt-6">
        {sections.map(({ type, items: typeItems }) => {
          const typeInfo = getOpportunityType(type);

          return (
            <section
              key={type}
              className={`neo-opportunity-section neo-opportunity-section--${type}`}
              aria-labelledby={`opportunities-${type}-heading`}
            >
              <h2
                id={`opportunities-${type}-heading`}
                className="neo-opportunity-section-heading"
              >
                <span className={`neo-badge ${typeInfo.badge}`}>
                  {typeInfo.label}
                </span>
                <span className="neo-opportunity-section-count">
                  {typeItems.length}
                </span>
              </h2>

              <ul
                className="grid grid-cols-1 sm:grid-cols-2 gap-5 list-none p-0 m-0"
                aria-label={`${typeInfo.label} opportunities`}
              >
                {typeItems.map((item) => (
                  <li key={item.title}>
                    <OpportunityCard item={item} />
                  </li>
                ))}
              </ul>
            </section>
          );
        })}
      </div>

      <OpportunityCalendar items={items} />
    </PageShell>
  );
}
