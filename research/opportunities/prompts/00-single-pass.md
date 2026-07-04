# Prompt 0: Single-pass harvest (legacy)

**Status:** legacy: prefer [01-source-map.md](./01-source-map.md) + [02-harvest.md](./02-harvest.md) for Issue #3+ 
**Run in:** Google Deep Research 
**Save output as:** `responses/YYYY-MM-DD-<slug>.md`

Used for early issues (e.g. Issue #1). One-shot prompt without a prior source-map phase.

## Research prompt

Curate opportunities for a Philippines-focused newsletter (tech-forward, not tech-only)

I'm building a monthly curated opportunities newsletter at **stimmie.dev/opportunities** for students, early-career builders, and curious generalists in the **Philippines** (with remote/online options welcome). The audience includes CS students, designers, founders, and people who like hackathons and learning programs: but **not everything should be tech**. I want a mix: tech + adjacent fields (design, product, startups, science, media, civic, arts, sports, scholarships, fellowships, competitions).

**Your task:** Find **25–40 high-quality opportunities** that are **open, upcoming, or currently accepting applications** as of today. Prioritize items with **clear official links** and **real deadlines**. Deprioritize spammy listicles, expired posts, and aggregator pages with no primary source.

### Categories to cover (use these exact types)

1. **hackathon**: hackathons, buildathons, case competitions with a build/demo component
2. **internship**: paid/unpaid internships, fellowships with work placement, GSoC-style programs
3. **event**: conferences, meetups, workshops, summits, career fairs, talks (attend or register)
4. **certificate**: free or scholarship-backed certs, microcredentials, professional courses with a credential
5. **program**: accelerators, incubators, grants, scholarships, exchange programs, long-form cohorts

**Target mix:** roughly **40% tech**, **60% broader** (design, business, research, policy, creative, campus life, etc.). Still include strong tech items: just don't make the list exclusively SWE/data.

### Geography

- **Primary:** Philippines (Metro Manila, Luzon, Visayas, Mindanao, nationwide)
- **Secondary:** Fully remote / online / "open to PH applicants"
- **Tertiary:** International programs that explicitly accept Philippine residents or are commonly accessible

### Time window

Focus on opportunities where **at least one key date** falls in the **next 90 days** (registration close, application deadline, event date, or program start). If something is evergreen (always-open cert portal), include only if it's widely useful and note that clearly.

### What to find for EACH opportunity

Return a structured entry with:

| Field | Notes |
| ----- | ----- |
| **title** | Official name |
| **type** | One of: hackathon, internship, event, certificate, program |
| **url** | Official registration/application/info page (not a random blog repost) |
| **org** | Organizer |
| **location** | City/region, "Online", or "Remote" |
| **dates** | **Multiple labeled dates** when they exist: do NOT collapse to one deadline |
| **blurb** | 1–2 sentences: who it's for + why it's worth applying |
| **eligibility** | Students only? PH citizens? Year level? Experience level? |
| **cost** | Free / paid / stipend / scholarship available |
| **confidence** | High / Medium / Low |

### Output format

1. **Executive summary**: 5–8 bullets: biggest themes, gaps, caveats
2. **Table of all opportunities** sorted by nearest upcoming deadline
3. **Detailed entries** grouped by type (JSON blocks matching `src/data/opportunities.js` items)
4. **"Worth watching"**: 5–10 opportunities opening soon but not yet actionable
5. **Sources bibliography**: every URL relied on, with dates accessed

Use **Asia/Manila (UTC+8)**. Today's date: **[INSERT DATE]**.

### Optional follow-up prompts

- "Narrow to **free** opportunities only and re-rank."
- "Give me **10 more hackathons** accepting beginners in PH."
- "Find **design/product** opportunities I missed."
- "Verify every **High confidence** item again; downgrade any where the official page disagrees."
