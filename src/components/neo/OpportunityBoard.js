"use client";

import { useMemo, useState } from "react";
import OpportunityCalendar from "@/components/neo/OpportunityCalendar";
import OpportunityCard from "@/components/neo/OpportunityCard";
import {
  filterOpportunities,
  getOpportunityType,
  groupIssueItemsByType,
  isOpportunityAiRelated,
  isOpportunityGameJam,
} from "@/data/opportunities";

export default function OpportunityBoard({ items }) {
  const [filter, setFilter] = useState("all");
  const [query, setQuery] = useState("");

  const filteredItems = useMemo(
    () =>
      filterOpportunities(items, {
        aiOnly: filter === "ai",
        gameJamOnly: filter === "game-jam",
        query,
      }),
    [items, filter, query],
  );
  const sections = useMemo(
    () => groupIssueItemsByType(filteredItems),
    [filteredItems],
  );
  const aiCount = useMemo(
    () => items.filter(isOpportunityAiRelated).length,
    [items],
  );

  const gameJamCount = useMemo(
    () => items.filter(isOpportunityGameJam).length,
    [items],
  );

  const hasSearch = query.trim().length > 0;
  const hasActiveFilters = filter !== "all" || hasSearch;

  function clearFilters() {
    setFilter("all");
    setQuery("");
  }

  function emptyMessage() {
    if (hasSearch && filter === "ai") {
      return "No AI-related listings match your search.";
    }
    if (hasSearch && filter === "game-jam") {
      return "No game jams match your search.";
    }
    if (hasSearch) {
      return "No listings match your search.";
    }
    if (filter === "game-jam") {
      return "No game jams match this filter right now.";
    }
    return "No AI-related listings match this filter right now.";
  }

  return (
    <>
      <div className="neo-opportunity-toolbar">
        <label className="neo-opportunity-search" htmlFor="opportunity-search">
          <span className="neo-opportunity-search-label">Search</span>
          <span className="neo-opportunity-search-field">
            <input
              id="opportunity-search"
              type="search"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Title, org, location, type…"
              className="neo-opportunity-search-input"
              autoComplete="off"
              spellCheck={false}
            />
            {hasSearch ? (
              <button
                type="button"
                className="neo-opportunity-search-clear"
                onClick={() => setQuery("")}
                aria-label="Clear search"
              >
                ×
              </button>
            ) : null}
          </span>
        </label>

        <div
          className="neo-opportunity-filters"
          role="group"
          aria-label="Filter opportunities"
        >
          <button
            type="button"
            className={`neo-opportunity-filter${filter === "all" ? " neo-opportunity-filter--active" : ""}`}
            aria-pressed={filter === "all"}
            onClick={() => setFilter("all")}
          >
            All
            <span className="neo-opportunity-filter-count">{items.length}</span>
          </button>
          <button
            type="button"
            className={`neo-opportunity-filter neo-opportunity-filter--game-jam${filter === "game-jam" ? " neo-opportunity-filter--active" : ""}`}
            aria-pressed={filter === "game-jam"}
            onClick={() => setFilter("game-jam")}
          >
            Game jams
            <span className="neo-opportunity-filter-count">{gameJamCount}</span>
          </button>
          <button
            type="button"
            className={`neo-opportunity-filter neo-opportunity-filter--ai${filter === "ai" ? " neo-opportunity-filter--active" : ""}`}
            aria-pressed={filter === "ai"}
            onClick={() => setFilter("ai")}
          >
            AI-related
            <span className="neo-opportunity-filter-count">{aiCount}</span>
          </button>
        </div>
      </div>

      {hasActiveFilters ? (
        <p
          className="neo-opportunity-results m-0 mt-3 text-sm neo-muted"
          style={{ fontFamily: "var(--neo-ui)" }}
          aria-live="polite"
        >
          Showing <strong>{filteredItems.length}</strong> of {items.length}{" "}
          listings
        </p>
      ) : null}

      {filteredItems.length === 0 ? (
        <p className="neo-opportunity-empty neo-muted m-0 mt-4">
          {emptyMessage()}{" "}
          <button
            type="button"
            className="neo-opportunity-filter-link"
            onClick={clearFilters}
          >
            Clear filters
          </button>
          .
        </p>
      ) : (
        <div className="neo-opportunity-sections mt-6">
          {sections.map(({ type, items: typeItems }) => {
            const typeInfo = getOpportunityType(type);

            return (
              <section
                key={type}
                className={`neo-opportunity-section neo-opportunity-section--${type}`}
                aria-labelledby={`opportunities-${type}-heading`}
              >
                <h2
                  id={`opportunities-${type}-heading`}
                  className="neo-opportunity-section-heading"
                >
                  <span className={`neo-badge ${typeInfo.badge}`}>
                    {typeInfo.label}
                  </span>
                  <span className="neo-opportunity-section-count">
                    {typeItems.length}
                  </span>
                </h2>

                <ul
                  className="grid grid-cols-1 sm:grid-cols-2 gap-5 list-none p-0 m-0"
                  aria-label={`${typeInfo.label} opportunities`}
                >
                  {typeItems.map((item) => (
                    <li key={item.title}>
                      <OpportunityCard item={item} />
                    </li>
                  ))}
                </ul>
              </section>
            );
          })}
        </div>
      )}

      <OpportunityCalendar items={filteredItems} />
    </>
  );
}
