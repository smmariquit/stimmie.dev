"use client";

import { useMemo, useState } from "react";
import OpportunityCalendar from "@/components/neo/OpportunityCalendar";
import OpportunityCard from "@/components/neo/OpportunityCard";
import {
  filterOpportunities,
  getOpportunityType,
  groupIssueItemsByType,
  isOpportunityAiRelated,
  OPPORTUNITY_TYPE_ORDER,
  OPPORTUNITY_TYPES,
} from "@/data/opportunities";

export default function OpportunityBoard({ items }) {
  const [typeFilter, setTypeFilter] = useState("all");
  const [aiOnly, setAiOnly] = useState(false);
  const [query, setQuery] = useState("");

  const typeCounts = useMemo(() => {
    const counts = new Map();
    for (const item of items) {
      counts.set(item.type, (counts.get(item.type) ?? 0) + 1);
    }
    return counts;
  }, [items]);

  const filteredItems = useMemo(
    () =>
      filterOpportunities(items, {
        type: typeFilter,
        aiOnly,
        query,
      }),
    [items, typeFilter, aiOnly, query],
  );
  const sections = useMemo(
    () => groupIssueItemsByType(filteredItems),
    [filteredItems],
  );
  const aiCount = useMemo(
    () => items.filter(isOpportunityAiRelated).length,
    [items],
  );

  const hasSearch = query.trim().length > 0;
  const hasActiveFilters = typeFilter !== "all" || aiOnly || hasSearch;

  function clearFilters() {
    setTypeFilter("all");
    setAiOnly(false);
    setQuery("");
  }

  function emptyMessage() {
    const typeLabel =
      typeFilter === "all"
        ? null
        : (OPPORTUNITY_TYPES[typeFilter]?.label ?? typeFilter);

    if (hasSearch && typeLabel && aiOnly) {
      return `No ${typeLabel.toLowerCase()} AI-related listings match your search.`;
    }
    if (hasSearch && typeLabel) {
      return `No ${typeLabel.toLowerCase()} listings match your search.`;
    }
    if (hasSearch && aiOnly) {
      return "No AI-related listings match your search.";
    }
    if (hasSearch) {
      return "No listings match your search.";
    }
    if (typeLabel && aiOnly) {
      return `No ${typeLabel.toLowerCase()} AI-related listings right now.`;
    }
    if (typeLabel) {
      return `No ${typeLabel.toLowerCase()} listings right now.`;
    }
    if (aiOnly) {
      return "No AI-related listings match this filter right now.";
    }
    return "No listings match this filter right now.";
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
      </div>

      <div className="neo-opportunity-filter-groups">
        <div className="neo-opportunity-filter-group">
          <p className="neo-opportunity-filter-group-label">Type</p>
          <div
            className="neo-opportunity-filters"
            role="group"
            aria-label="Filter by opportunity type"
          >
            <button
              type="button"
              className={`neo-opportunity-filter${typeFilter === "all" ? " neo-opportunity-filter--active" : ""}`}
              aria-pressed={typeFilter === "all"}
              onClick={() => setTypeFilter("all")}
            >
              All
              <span className="neo-opportunity-filter-count">{items.length}</span>
            </button>
            {OPPORTUNITY_TYPE_ORDER.map((type) => {
              const count = typeCounts.get(type) ?? 0;
              if (!count) {
                return null;
              }

              const typeInfo = OPPORTUNITY_TYPES[type];
              return (
                <button
                  key={type}
                  type="button"
                  className={`neo-opportunity-filter neo-opportunity-filter--${type}${typeFilter === type ? " neo-opportunity-filter--active" : ""}`}
                  aria-pressed={typeFilter === type}
                  onClick={() => setTypeFilter(type)}
                >
                  {typeInfo.label}
                  <span className="neo-opportunity-filter-count">{count}</span>
                </button>
              );
            })}
          </div>
        </div>

        <div className="neo-opportunity-filter-group">
          <p className="neo-opportunity-filter-group-label">Tags</p>
          <div
            className="neo-opportunity-filters"
            role="group"
            aria-label="Filter by tags"
          >
            <button
              type="button"
              className={`neo-opportunity-filter neo-opportunity-filter--ai${aiOnly ? " neo-opportunity-filter--active" : ""}`}
              aria-pressed={aiOnly}
              onClick={() => setAiOnly((value) => !value)}
            >
              AI-related
              <span className="neo-opportunity-filter-count">{aiCount}</span>
            </button>
          </div>
        </div>
      </div>

      {hasActiveFilters ? (
        <p
          className="neo-opportunity-results m-0 mt-3 neo-muted"
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
