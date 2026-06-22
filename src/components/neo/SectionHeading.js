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
          className="neo-section-copy"
          onClick={copyLink}
          aria-label={copied ? "Section link copied" : `Copy link to ${title}`}
        >
          {copied ? "copied!" : "copy"}
        </button>
      </div>
    </div>
  );
}
