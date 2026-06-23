import Link from "next/link";
import { notFound } from "next/navigation";
import OpportunityCard from "@/components/neo/OpportunityCard";
import PageShell from "@/components/neo/PageShell";
import {
  getIssueBySlug,
  opportunityIssues,
} from "@/data/opportunities";

export function generateStaticParams() {
  return opportunityIssues.map((issue) => ({ slug: issue.slug }));
}

function issueHeading(issue) {
  return issue.title
    ? `Issue #${issue.issueNumber} — ${issue.title}`
    : `Issue #${issue.issueNumber}`;
}

export async function generateMetadata({ params }) {
  const { slug } = await params;
  const issue = getIssueBySlug(slug);
  if (!issue) {
    return { title: "Issue not found" };
  }

  const title = issueHeading(issue);
  return {
    title,
    description: issue.intro,
    openGraph: {
      title,
      description: issue.intro,
      type: "article",
    },
  };
}

export default async function OpportunityIssuePage({ params }) {
  const { slug } = await params;
  const issue = getIssueBySlug(slug);
  if (!issue) {
    notFound();
  }

  return (
    <PageShell
      title={`~ ${issueHeading(issue)} ~`}
      intro={issue.intro}
      current="/opportunities"
      maxWidth="64rem"
    >
      <p className="mb-4 text-base">
        <Link href="/opportunities">◄ back to all issues</Link>
      </p>

      <ul
        className="grid grid-cols-1 sm:grid-cols-2 gap-5 list-none p-0 m-0"
        aria-label={`Opportunities in ${issueHeading(issue)}`}
      >
        {issue.items.map((item) => (
          <li key={`${issue.slug}-${item.title}`}>
            <OpportunityCard item={item} />
          </li>
        ))}
      </ul>
    </PageShell>
  );
}
