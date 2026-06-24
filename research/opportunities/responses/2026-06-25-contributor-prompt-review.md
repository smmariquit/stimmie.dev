# Contributor prompt review — 2026-06-25

Meta-analysis of [06-contributor-calls-harvest.md](../prompts/06-contributor-calls-harvest.md): prompt type, best practices, and modular refactor.

## Findings

- **Type:** Multi-step research / instruction-following prompt for Deep Research or Cursor browse.
- **Strengths:** Clear categories, hard excludes, canonical URL rule, JSON schema, Devpost automation hook.
- **Weaknesses:** High cognitive load in one run; output fields buried in tables; citations implied but not mandatory.

## Recommendation (implemented)

**Modularize** into focused sub-prompts (speakers, judges/mentors, sponsors, volunteers) run in parallel, then merge. Each sub-prompt:

- States deadline rule upfront (45 days, future only)
- Lists required output fields in a table
- Names category-specific sources and search queries
- Sets a minimum target count per run

**Orchestrator:** Prompt 6 now links 06a–06d, keeps shared schema/excludes once, and documents a 3–4 day implementation checklist.

## Sub-prompt mapping

| Report alternative | Site file |
| ------------------ | --------- |
| Speakers only | `06a-call-for-speakers-harvest.md` |
| Hackathon mentors/judges | `06b-call-for-judges-mentors-harvest.md` |
| Sponsorship opportunities | `06c-sponsorship-calls-harvest.md` |
| Volunteer roles | `06d-volunteer-facilitator-harvest.md` |

## Success metrics (per harvest)

- ≥3–5 verified entries per sub-prompt when ecosystem is active
- Every entry: official apply URL + `source_url` + bibliography access date
- Zero expired deadlines in main sections
- Coverage report with rejected items and reasons

## Next steps

1. Run 06a–06d in parallel for next contributor harvest
2. Merge into `responses/YYYY-MM-DD-contributor-calls-harvest.md`
3. `npm run opportunities:devpost` for Devpost judge/mentor supplement
4. Verify URLs → `opportunities:merge` → bump `lastUpdated`
