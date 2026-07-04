# Opportunities research

Feeds **[stimmie.dev/opportunities](https://stimmie.dev/opportunities)**: one living board for students and early-career builders in the Philippines.

## Two-phase pipeline (recommended)

| Phase | Prompt | Output file |
| ----- | ------ | ----------- |
| **1: Source map** | [prompts/01-source-map.md](./prompts/01-source-map.md) | `responses/YYYY-MM-DD-source-map.md` |
| **2: Harvest** | [prompts/02-harvest.md](./prompts/02-harvest.md) | `responses/YYYY-MM-DD-harvest.md` |
| **2b: Devpost** | [prompts/03-devpost-harvest.md](./prompts/03-devpost-harvest.md) | `responses/YYYY-MM-DD-devpost-harvest.md` |
| **2c: Email** | [prompts/04-email-inbox-harvest.md](./prompts/04-email-inbox-harvest.md) | `responses/YYYY-MM-DD-email-harvest.md` |
| **2d: Game jams** | [prompts/05-game-jam-harvest.md](./prompts/05-game-jam-harvest.md) | `responses/YYYY-MM-DD-game-jam-harvest.md` |
| **2e: Contributors & sponsors** | [prompts/06-contributor-calls-harvest.md](./prompts/06-contributor-calls-harvest.md) (or **06a–06d** in parallel) | `responses/YYYY-MM-DD-contributor-calls-harvest.md` |
| **2f: Humanities & sociocivic** | [prompts/07-humanities-sociocivic-harvest.md](./prompts/07-humanities-sociocivic-harvest.md) | `responses/YYYY-MM-DD-humanities-sociocivic-harvest.md` |
| **2g: Scholarships** | [prompts/08-scholarships-harvest.md](./prompts/08-scholarships-harvest.md) | `responses/YYYY-MM-DD-scholarships-harvest.md` |
| **2h: Campus / university events** | [prompts/09-campus-university-events-harvest.md](./prompts/09-campus-university-events-harvest.md) | `responses/YYYY-MM-DD-campus-events-harvest.md` |
| **2i: Exchange programs** | [prompts/10-exchange-programs-harvest.md](./prompts/10-exchange-programs-harvest.md) | `responses/YYYY-MM-DD-exchange-programs-harvest.md` |
| **3: Verify** | (manual / agent) | deduped JSON → `opportunitiesBoard.items` |

Run phase 1 first. Attach the source-map response (or its “top 15 sources” section) when running phase 2.

### Contributor calls (Prompt 6): modular run

For best results, run sub-prompts **in parallel** and merge one response file:

| Sub-prompt | Focus |
| ---------- | ----- |
| [06a](./prompts/06a-call-for-speakers-harvest.md) | CFP / speakers |
| [06b](./prompts/06b-call-for-judges-mentors-harvest.md) | Judges & mentors |
| [06c](./prompts/06c-sponsorship-calls-harvest.md) | Sponsor calls & funding programs |
| [06d](./prompts/06d-volunteer-facilitator-harvest.md) | Volunteers & facilitators |

Shared JSON schema and excludes live in [06](./prompts/06-contributor-calls-harvest.md). Devpost judge/mentor scan: `npm run opportunities:devpost`.

## After harvest → site

1. Merge harvest: `npm run opportunities:merge` (full) or `npm run opportunities:devpost` (Devpost only).
2. Bump `opportunitiesBoard.lastUpdated` in `src/data/opportunities.js`.
3. Run `npm run opportunities:images` for card images.
4. Move the response markdown to `archive/` when done.

## Image automation

Card images: `npm run opportunities:images` (OG meta → screenshot API → type default). See `.env.example` for screenshot keys.
