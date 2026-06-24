"use client";

import { useEffect, useId, useMemo, useRef, useState } from "react";
import { getOpportunitySearchSuggestions } from "@/data/opportunities";

const SUGGESTION_KIND_LABEL = {
  title: "Listing",
  org: "Org",
  location: "Location",
  type: "Type",
  quick: "Suggestion",
};

export default function OpportunitySearch({ items, value, onChange }) {
  const inputId = useId();
  const listboxId = useId();
  const rootRef = useRef(null);
  const [open, setOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(-1);

  const suggestions = useMemo(
    () => getOpportunitySearchSuggestions(items, value),
    [items, value],
  );

  const showSuggestions = open && suggestions.length > 0;

  useEffect(() => {
    setActiveIndex(showSuggestions ? 0 : -1);
  }, [value, showSuggestions, suggestions.length]);

  useEffect(() => {
    function handlePointerDown(event) {
      if (!rootRef.current?.contains(event.target)) {
        setOpen(false);
      }
    }

    document.addEventListener("pointerdown", handlePointerDown);
    return () => document.removeEventListener("pointerdown", handlePointerDown);
  }, []);

  function selectSuggestion(text) {
    onChange(text);
    setOpen(false);
  }

  function handleKeyDown(event) {
    if (!showSuggestions) {
      if (event.key === "ArrowDown" && suggestions.length > 0) {
        event.preventDefault();
        setOpen(true);
      }
      return;
    }

    if (event.key === "ArrowDown") {
      event.preventDefault();
      setActiveIndex((current) => (current + 1) % suggestions.length);
      return;
    }

    if (event.key === "ArrowUp") {
      event.preventDefault();
      setActiveIndex(
        (current) => (current - 1 + suggestions.length) % suggestions.length,
      );
      return;
    }

    if (event.key === "Enter" && activeIndex >= 0) {
      event.preventDefault();
      selectSuggestion(suggestions[activeIndex].text);
      return;
    }

    if (event.key === "Escape") {
      event.preventDefault();
      setOpen(false);
    }
  }

  const hasValue = value.trim().length > 0;

  return (
    <div
      ref={rootRef}
      className={`neo-opportunity-search-wrap${showSuggestions ? " neo-opportunity-search-wrap--open" : ""}`}
    >
      <label className="neo-opportunity-search" htmlFor={inputId}>
        <span className="neo-opportunity-search-label">Search</span>
        <span className="neo-opportunity-search-field">
          <input
            id={inputId}
            type="search"
            value={value}
            onChange={(event) => {
              onChange(event.target.value);
              setOpen(true);
            }}
            onFocus={() => setOpen(true)}
            onKeyDown={handleKeyDown}
            placeholder="Title, org, location, type…"
            className="neo-opportunity-search-input"
            autoComplete="off"
            spellCheck={false}
            role="combobox"
            aria-autocomplete="list"
            aria-expanded={showSuggestions}
            aria-controls={listboxId}
            aria-activedescendant={
              showSuggestions && activeIndex >= 0
                ? `${listboxId}-option-${activeIndex}`
                : undefined
            }
          />
          {hasValue ? (
            <button
              type="button"
              className="neo-opportunity-search-clear"
              onClick={() => {
                onChange("");
                setOpen(true);
              }}
              aria-label="Clear search"
            >
              ×
            </button>
          ) : null}
        </span>
      </label>

      {showSuggestions ? (
        <ul
          id={listboxId}
          className="neo-opportunity-search-suggestions list-none p-0 m-0"
          role="listbox"
          aria-label="Search suggestions"
        >
          {suggestions.map((suggestion, index) => (
            <li
              key={`${suggestion.kind}:${suggestion.text}`}
              id={`${listboxId}-option-${index}`}
              role="option"
              aria-selected={index === activeIndex}
            >
              <button
                type="button"
                className={`neo-opportunity-search-suggestion${index === activeIndex ? " neo-opportunity-search-suggestion--active" : ""}`}
                onMouseDown={(event) => event.preventDefault()}
                onClick={() => selectSuggestion(suggestion.text)}
              >
                <span className="neo-opportunity-search-suggestion-text">
                  {suggestion.text}
                </span>
                <span className="neo-opportunity-search-suggestion-kind">
                  {SUGGESTION_KIND_LABEL[suggestion.kind]}
                </span>
              </button>
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}
