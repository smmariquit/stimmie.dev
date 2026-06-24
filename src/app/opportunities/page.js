import OpportunityBoard from "@/components/neo/OpportunityBoard";
import OpportunityDisclaimer from "@/components/neo/OpportunityDisclaimer";
import PageShell from "@/components/neo/PageShell";
import {
  formatBoardUpdated,
  getOpportunities,
  getOpportunitiesBoard,
  OPPORTUNITY_TYPE_ORDER,
  OPPORTUNITY_TYPES,
} from "@/data/opportunities";

const OPPORTUNITIES_SHARE_TITLE = "Opportunities";

export function generateMetadata() {
  const board = getOpportunitiesBoard();
  const items = getOpportunities();
  const updated = formatBoardUpdated(board.lastUpdated);
  const rows = OPPORTUNITY_TYPE_ORDER.map((type) => {
    const count = items.filter((item) => item.type === type).length;
    return count ? `${OPPORTUNITY_TYPES[type].label} ${count}` : null;
  }).filter(Boolean);
  const description = `${items.length} live listings: ${rows.join(", ")}. Last updated ${updated}. Personal roundup — verify deadlines on official sites.`;

  return {
    title: "Opportunities",
    description,
    alternates: {
      canonical: "/opportunities",
    },
    openGraph: {
      title: OPPORTUNITIES_SHARE_TITLE,
      description,
      url: "/opportunities",
      siteName: "Stimmie",
      locale: "en_US",
      type: "website",
    },
    twitter: {
      card: "summary_large_image",
      title: OPPORTUNITIES_SHARE_TITLE,
      description,
    },
  };
}

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
