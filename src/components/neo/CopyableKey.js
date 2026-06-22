"use client";

import { useCallback, useState } from "react";

export default function CopyableKey({ text, download, downloadName }) {
  const [copied, setCopied] = useState(false);

  const copy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(text.trim());
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      window.prompt("Copy this key:", text.trim());
    }
  }, [text]);

  return (
    <div className="neo-key-wrap">
      <div className="neo-key-toolbar">
        <button
          type="button"
          className={`neo-section-copy neo-key-copy${copied ? " neo-section-copy--copied" : ""}`}
          onClick={copy}
          aria-label={copied ? "Key copied" : "Copy key"}
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
              <rect
                x="5.25"
                y="5.25"
                width="7.5"
                height="7.5"
                rx="1"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.25"
              />
              <path
                d="M4.25 10.75V4.75a1 1 0 0 1 1-1h6"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.25"
                strokeLinecap="round"
              />
            </svg>
          )}
        </button>
        {download ? (
          <a
            href={download}
            download={downloadName}
            className="neo-key-download"
          >
            download
          </a>
        ) : null}
      </div>
      <pre className="neo-key-block">
        <code>{text.trim()}</code>
      </pre>
    </div>
  );
}
