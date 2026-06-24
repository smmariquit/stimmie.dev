import OpportunityBoard from "@/components/neo/OpportunityBoard";
import OpportunityDisclaimer from "@/components/neo/OpportunityDisclaimer";
import PageShell from "@/components/neo/PageShell";
import {
  formatBoardUpdated,
  getOpportunities,
  getOpportunitiesBoard,
} from "@/data/opportunities";

const OPPORTUNITIES_OG_IMAGE = "/opportunities/og.png";

export function generateMetadata() {
  const board = getOpportunitiesBoard();
  const items = getOpportunities();
  const updated = formatBoardUpdated(board.lastUpdated);
  const description = `${board.intro} ${items.length} listings · last updated ${updated}.`;
  const shareTitle = "~ opportunities ~";

  return {
    title: "Opportunities",
    description,
    alternates: {
      canonical: "/opportunities",
    },
    openGraph: {
      title: shareTitle,
      description,
      url: "/opportunities",
      siteName: "Stimmie",
      locale: "en_US",
      type: "website",
      images: [
        {
          url: OPPORTUNITIES_OG_IMAGE,
          width: 1200,
          height: 630,
          alt: "Stimmie opportunities board for hackathons, internships, events, and programs in the Philippines",
        },
      ],
    },
    twitter: {
      card: "summary_large_image",
      title: shareTitle,
      description,
      images: [OPPORTUNITIES_OG_IMAGE],
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
