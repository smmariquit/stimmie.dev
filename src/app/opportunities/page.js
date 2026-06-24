import OpportunityBoard from "@/components/neo/OpportunityBoard";
import OpportunityDisclaimer from "@/components/neo/OpportunityDisclaimer";
import PageShell from "@/components/neo/PageShell";
import {
  formatBoardUpdated,
  getOpportunities,
  getOpportunitiesBoard,
} from "@/data/opportunities";

export const metadata = {
  title: "Opportunities",
  description:
    "A curated, incomplete roundup of hackathons, internships, events, and programs for students and early-career builders in the Philippines.",
};

export default function OpportunitiesPage() {
  const board = getOpportunitiesBoard();
  const items = getOpportunities();

  return (
    <PageShell
      title="~ opportunities ~"
      intro={board.intro}
      current="/opportunities"
      maxWidth="64rem"
    >
      <div className="neo-opportunity-page">
        <p className="neo-opportunity-updated m-0 mb-4 neo-muted">
          <strong>Last updated:</strong>{" "}
          <time dateTime={board.lastUpdated}>
            {formatBoardUpdated(board.lastUpdated)}
          </time>
          {" · "}
          {items.length} listings
        </p>

        <OpportunityDisclaimer />

        <OpportunityBoard items={items} />
      </div>
    </PageShell>
  );
}
