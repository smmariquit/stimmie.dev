"use client";

import { useCallback, useState } from "react";

export default function SectionHeading({ id, title }) {
  const [copied, setCopied] = useState(false);

  const copyLink = useCallback(async () => {
    const url = `${window.location.origin}${window.location.pathname}#${id}`;
    try {
      await navigator.clipboard.writeText(url);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      window.prompt("Copy this link:", url);
    }
  }, [id]);

  return (
    <div className="neo-section-header">
      <h2 className="neo-section-title">
        <a href={`#${id}`} className="neo-section-title-link">
          {title}
        </a>
      </h2>
      <div className="neo-section-link-tools">
        <code className="neo-section-slug">#{id}</code>
        <button
          type="button"
          className={`neo-section-copy${copied ? " neo-section-copy--copied" : ""}`}
          onClick={copyLink}
          aria-label={copied ? "Section link copied" : `Copy link to ${title}`}
        >
          {copied ? (
            <svg viewBox="0 0 16 16" aria-hidden="true">
              <path
                d="M3.5 8.5 6.5 11.5 12.5 4.5"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          ) : (
            <svg viewBox="0 0 16 16" aria-hidden="true">
              <path
                d="M6.25 9.25a2.25 2.25 0 0 0 3.18 0l1.82-1.82a2.25 2.25 0 1 0-3.18-3.18L7.5 5.32"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.25"
                strokeLinecap="round"
              />
              <path
                d="M9.75 6.75a2.25 2.25 0 0 0-3.18 0L4.75 8.57a2.25 2.25 0 1 0 3.18 3.18l.57-.57"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.25"
                strokeLinecap="round"
              />
            </svg>
          )}
        </button>
      </div>
    </div>
  );
}
