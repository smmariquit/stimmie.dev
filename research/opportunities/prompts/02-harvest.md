# Prompt 2 — Harvest (FIND opportunities)

**Phase:** 2 of 2 · **Run in:** Google Deep Research  
**Prerequisite:** [01-source-map.md](./01-source-map.md) response saved in `responses/`  
**Save output as:** `responses/YYYY-MM-DD-<issue-slug>-harvest.md`

Attach Prompt 1's output (or paste the "top 15 sources" section) when running this prompt.

## Research task

Harvest Q3/Q4 2026 opportunities using the Source Map

Using the source map from my prior research, find **40–50 opportunities** for a Philippines-focused newsletter. **Tech-forward but not tech-exclusive** (~40% tech, 60% broader: design, business, policy, health, campus, civic).

### Mandatory source coverage

You MUST search each of these buckets and report how many items came from each:

- [ ] Devpost (online + open to intl/PH)
- [ ] Devfolio
- [ ] DoraHacks + lablab.ai (AI hackathons)
- [ ] Codédex Discord / codedex.io events
- [ ] GitHub: SimplifyJobs/Summer2026-Internships (+ remote-friendly filter)
- [ ] Prosple PH + Intern.ph
- [ ] LinkedIn (employer career pages only — not search URLs)
- [ ] Reddit (r/Philippines, r/phcareers, r/devph, hackathon announcement threads)
- [ ] Facebook: DEVCON PH + 3 campus tech groups
- [ ] YC events.ycombinator.com + Luma SF/hybrid builder events
- [ ] CHED / DOST / DICT / GSIS official portals
- [ ] Friends of Figma / UXPH / Meetup Manila tech
- [ ] Outreachy + Google Summer of Code (if open)

### Per-opportunity output (JSON)

```json
{
  "title": "",
  "type": "hackathon|internship|event|certificate|program",
  "url": "OFFICIAL PAGE ONLY",
  "org": "",
  "location": "",
  "dates": [
    { "label": "Registration closes", "date": "YYYY-MM-DD" },
    { "label": "Event", "date": "YYYY-MM-DD", "endDate": "YYYY-MM-DD" }
  ],
  "blurb": "",
  "source_platform": "e.g. Devpost",
  "source_url": "where you found it",
  "confidence": "High|Medium|Low"
}
```

### Rules

- **No single deadline** — use labeled `dates` array (registration vs event vs program period)
- **Rolling** applications → omit deadline dates, say "rolling" in blurb
- **Dedupe** across platforms (same hackathon on Devpost + Devfolio = one entry)
- **Reject** Indeed/LinkedIn search-result URLs, Scribd PDFs, governmentph.com mirrors without official link
- Sort final table by nearest actionable date
- Flag items closing in **48 hours** at the top

### Also include

- **Coverage report**: items found per source bucket (show gaps honestly)
- **Horizon radar**: 10 opportunities opening in 30–60 days
- **Bibliography** with access dates

Asia/Manila timezone. Today: **[INSERT DATE]**.
