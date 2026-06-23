# Opportunities research

Feeds **[stimmie.dev/opportunities](https://stimmie.dev/opportunities)** — curated issues for students and early-career builders in the Philippines.

## Two-phase pipeline (recommended)

| Phase | Prompt | Output file |
| ----- | ------ | ----------- |
| **1 — Source map** | [prompts/01-source-map.md](./prompts/01-source-map.md) | `responses/YYYY-MM-DD-source-map.md` |
| **2 — Harvest** | [prompts/02-harvest.md](./prompts/02-harvest.md) | `responses/YYYY-MM-DD-<issue-slug>-harvest.md` |
| **3 — Verify** | (manual / agent) | deduped JSON → `src/data/opportunities.js` |

Run phase 1 first. Attach the source-map response (or its “top 15 sources” section) when running phase 2.

**Legacy:** [prompts/00-single-pass.md](./prompts/00-single-pass.md) — one-shot prompt used for early issues; prefer the two-phase flow for Issue #3+.

## After harvest → site

1. Add a new issue object in `src/data/opportunities.js` (new `slug`, bump `issueNumber`).
2. Map each JSON entry to an `items[]` object (`title`, `type`, `url`, `org`, `location`, `dates`, `blurb`).
   - Add `imageUrl` when the apply page blocks bots but a marketing/OG page exists.
3. Run `npm run opportunities:images` for card images.
4. Move the response markdown to `archive/` and note the issue slug in the file header.

## Published from research

| Issue | Slug | Research source |
| ----- | ---- | --------------- |
| #2 Q3 2026 Ecosystem Report | `q3-2026` | pasted report (pre-pipeline) — add to `archive/` when backfilled |

## Image automation

Card images: `npm run opportunities:images` (OG meta → screenshot API → type default). See `.env.example` for screenshot keys.
