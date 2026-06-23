# Opportunities research

Feeds **[stimmie.dev/opportunities](https://stimmie.dev/opportunities)** — one living board for students and early-career builders in the Philippines.

## Two-phase pipeline (recommended)

| Phase | Prompt | Output file |
| ----- | ------ | ----------- |
| **1 — Source map** | [prompts/01-source-map.md](./prompts/01-source-map.md) | `responses/YYYY-MM-DD-source-map.md` |
| **2 — Harvest** | [prompts/02-harvest.md](./prompts/02-harvest.md) | `responses/YYYY-MM-DD-harvest.md` |
| **2b — Devpost** | [prompts/03-devpost-harvest.md](./prompts/03-devpost-harvest.md) | `responses/YYYY-MM-DD-devpost-harvest.md` |
| **3 — Verify** | (manual / agent) | deduped JSON → `opportunitiesBoard.items` |

Run phase 1 first. Attach the source-map response (or its “top 15 sources” section) when running phase 2.

## After harvest → site

1. Merge harvest: `npm run opportunities:merge` (full) or `npm run opportunities:devpost` (Devpost only).
2. Bump `opportunitiesBoard.lastUpdated` in `src/data/opportunities.js`.
3. Run `npm run opportunities:images` for card images.
4. Move the response markdown to `archive/` when done.

## Image automation

Card images: `npm run opportunities:images` (OG meta → screenshot API → type default). See `.env.example` for screenshot keys.
