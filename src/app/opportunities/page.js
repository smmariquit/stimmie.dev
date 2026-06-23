import Link from "next/link";
import PageShell from "@/components/neo/PageShell";
import { getSortedIssues } from "@/data/opportunities";

export const metadata = {
  title: "Opportunities",
  description:
    "A periodic roundup of hackathons, internships, events, and programs worth applying to.",
};

function formatPublished(date) {
  return new Date(date).toLocaleDateString("en-US", {
    month: "long",
    day: "numeric",
    year: "numeric",
    timeZone: "Asia/Manila",
  });
}

function issueHeading(issue) {
  return issue.title
    ? `Issue #${issue.issueNumber} — ${issue.title}`
    : `Issue #${issue.issueNumber}`;
}

export default function OpportunitiesPage() {
  const issues = getSortedIssues();

  return (
    <PageShell
      title="~ opportunities ~"
      intro="A newsletter-style roundup of hackathons, internships, events, certificate programs, and other things worth your time. New issues drop when I find good stuff."
      current="/opportunities"
      maxWidth="52rem"
    >
      <ul
        className="flex flex-col gap-4 list-none p-0 m-0"
        aria-label="Opportunity newsletter issues"
      >
        {issues.map((issue) => (
          <li key={issue.slug}>
            <Link
              href={`/opportunities/${issue.slug}`}
              className="neo-link-card group"
            >
              <span
                className="text-base neo-muted font-mono block"
                aria-hidden="true"
              >
                {formatPublished(issue.published)}
              </span>
              <span className="block font-bold text-xl mt-0.5 group-hover:text-[#cc0066]">
                {issueHeading(issue)}
              </span>
              {issue.intro ? (
                <span className="block mt-1">{issue.intro}</span>
              ) : null}
              <span
                className="block mt-2 text-base neo-muted"
                style={{ fontFamily: "var(--neo-ui)" }}
              >
                {issue.items.length} opportunit
                {issue.items.length === 1 ? "y" : "ies"}
              </span>
            </Link>
          </li>
        ))}
      </ul>
    </PageShell>
  );
}
