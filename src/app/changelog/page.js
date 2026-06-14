// src/app/changelog/page.js

import PageShell from "@/components/neo/PageShell";
import { getChangelogEntries, getSiteVersion } from "@/lib/changelog";

export const metadata = {
  title: "Changelog",
  description: "Release history and version notes for stimmie.dev",
};

// Renders inline markdown: [text](url) links, **bold**, and `code`. Returns an
// array of strings and React elements. Links render as short hyperlinks (just
// the link text), so long GitHub commit/compare URLs don't bloat the page.
function renderInline(text) {
  const nodes = [];
  const pattern =
    /\[([^\]]+)\]\(([^)]+)\)|\*\*(.+?)\*\*|`([^`]+)`/g;
  let lastIndex = 0;
  let match;
  let key = 0;
  while ((match = pattern.exec(text)) !== null) {
    if (match.index > lastIndex) {
      nodes.push(text.slice(lastIndex, match.index));
    }
    if (match[2] !== undefined) {
      nodes.push(
        <a
          key={key++}
          href={match[2]}
          target="_blank"
          rel="noopener noreferrer"
        >
          {match[1]}
        </a>,
      );
    } else if (match[3] !== undefined) {
      nodes.push(<strong key={key++}>{match[3]}</strong>);
    } else if (match[4] !== undefined) {
      nodes.push(
        <code key={key++} className="font-mono">
          {match[4]}
        </code>,
      );
    }
    lastIndex = pattern.lastIndex;
  }
  if (lastIndex < text.length) nodes.push(text.slice(lastIndex));
  return nodes;
}

function ChangelogBody({ text }) {
  return (
    <div className="neo-changelog-body text-base leading-relaxed">
      {text.split("\n").map((line, i) => {
        if (line.startsWith("### ")) {
          return (
            <h3
              key={i}
              className="font-bold neo-accent-blue mt-4 mb-1"
              style={{ fontFamily: "var(--neo-ui)" }}
            >
              {renderInline(line.replace(/^### /, ""))}
            </h3>
          );
        }
        if (line.startsWith("* ") || line.startsWith("- ")) {
          return (
            <p key={i} className="ml-4 my-1">
              • {renderInline(line.replace(/^[*-] /, ""))}
            </p>
          );
        }
        if (line.trim() === "") return <br key={i} />;
        return <p key={i}>{renderInline(line)}</p>;
      })}
    </div>
  );
}

export default function ChangelogPage() {
  const version = getSiteVersion();
  const entries = getChangelogEntries();

  return (
    <PageShell
      title="~ changelog ~"
      intro={`Current version: v${version}`}
      current="/changelog"
      maxWidth="52rem"
    >
      <div className="space-y-6">
        {entries.map((entry) => (
          <section
            key={entry.heading}
            className="pb-6"
            style={{ borderBottom: "2px dashed #d6008f" }}
          >
            <h2 className="neo-section-title">
              {renderInline(entry.heading)}
            </h2>
            <ChangelogBody text={entry.body} />
          </section>
        ))}
      </div>
    </PageShell>
  );
}
