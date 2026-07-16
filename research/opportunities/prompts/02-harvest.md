# Prompt 2: Harvest (FIND opportunities)

**Phase:** 2 of 2 · **Run in:** Google Deep Research 
**Prerequisite:** source-map response in `responses/` (e.g. [01-source-map.md](../responses/01-source-map.md)) 
**Save output as:** `responses/YYYY-MM-DD-<issue-slug>-harvest.md`

Attach the source map (at minimum the **top 15 sources** + **anti-patterns** sections) when running this prompt.

## Research task

Harvest the next newsletter issue using the **Philippine Opportunity Source Map**. Find **40–50 opportunities** for students and early-career builders in the Philippines. **Tech-forward but not tech-exclusive** (~40% tech, 60% broader: design, business, policy, health, campus, civic).

### Core rule (from source map)

**Boards discover; official pages publish.** Every entry needs:

- `url`: canonical apply/register/info page (what readers click)
- `image_url`: optional screenshot/OG-friendly page when `url` blocks bots (see below)

Never ship Indeed search URLs, Prosple listing pages, LinkedIn job-search URLs, Scribd mirrors, or `governmentph.com` reposts as `url`.

---

### Mandatory source coverage

Search each bucket; report item count per bucket in the **coverage report**:

| Bucket | Sources | Cadence hint |
| ------ | ------- | ------------ |
| Internships | LinkedIn Boolean → **employer ATS**; JobStreet; Kalibrr; Prosple (discovery only) | Mon |
| Hackathons | Devpost (daily); MLH; HackerEarth; lablab.ai; Devfolio (online/global only) | Tue |
| Gov scholarships | `bpms.ched.gov.ph`, `sei.dost.gov.ph`, `unifast.gov.ph`, `dict.gov.ph/trainings`, `e-tesda.gov.ph`, `gsis.gov.ph`, `owwa.gov.ph` | Wed: deep pass: [08-scholarships-harvest.md](./08-scholarships-harvest.md) |
| Events | Meetup, Eventbrite Manila tech, Luma, DEVCON PH, Friends of Figma, UXPH | Thu |
| Campus / FB | DEVCON chapters, campus CS org pages, Internships.ph group: deep pass: [09-campus-university-events-harvest.md](./09-campus-university-events-harvest.md) | Fri |
| Global programs | Outreachy, GSoC, ADB careers, UN/ASEAN portals: youth exchanges: [10-exchange-programs-harvest.md](./10-exchange-programs-harvest.md) | Sat |
| Certs / courses | UPOU MODeL, TESDA TOP, Grow with Google PH, AWS Educate, ITU Academy, Cisco NetAcad | Sun |

Also check: GitHub `SimplifyJobs/Summer2026-Internships` (remote filter), Codédex Discord, Reddit signal subs, YC events (virtual/intl only). For a **humanities / civic / policy** pass, run [07-humanities-sociocivic-harvest.md](./07-humanities-sociocivic-harvest.md).

**Devpost vs Devfolio:** Devpost daily; Devfolio weekly: only elevate Devfolio items that are explicitly **online** or **global**, not India-only offline events.

**Reddit (signal only, not first publication):** `r/PinoyProgrammer`, `r/cscareerquestions`, `r/csMajors`, `r/startups`: corroborate, then link to official page.

**LinkedIn Boolean (copy-paste, then verify on employer site):**

```text
("intern" OR "internship" OR "ojt" OR "student intern") AND ("software" OR "data" OR "ai" OR "product" OR "design") AND ("Philippines" OR Manila OR Makati OR Taguig OR Cebu OR Davao OR Remote) NOT ("senior" OR "manager")
```

**ATS-first employer search:**

```text
site:jobs.lever.co ("intern" OR "internship") ("Philippines" OR Remote)
site:boards.greenhouse.io ("intern" OR "internship") ("Philippines" OR Remote)
site:myworkdayjobs.com ("intern" OR "internship") ("Philippines" OR Remote)
```

---

### Bot protection & cover images

Many PH government, embassy, careers, and aggregator pages block screenshot bots (Cloudflare, login walls, cookie challenges). **Do not use those as `image_url`.**

| If `url` is… | Use `image_url` instead |
| ------------ | ----------------------- |
| `.gov.ph` scholarship portal with CF challenge | Official **news/press** page on same domain, or parent program on `grow.google`, `aws.amazon.com`, etc. |
| Embassy / NOFO (YSEALI, Chevening) | `yseali.state.gov`, program homepage, or organizer media kit |
| `careers.unilever.com` / `careers.loreal.com` job requisition | Brand careers landing or `unilever.com` / `loreal.com` corporate page |
| Prosple / Indeed / JobStreet listing | **Employer career page**: never Prosple search URL as `url` or `image_url` |
| Reddit announcement thread | Subreddit, Devpost entry, or sponsor landing if hackathon |
| Startup Networks / third-party hackathon aggregator | Devpost, Devfolio, or DoraHacks listing for same event |
| `dict.gov.ph` article behind CF | `grow.google/intl/en_ph/certificates/` or `dict.gov.ph/trainings` |

If no clean `image_url` exists, omit it: the site falls back to a **type placeholder** (better than a CAPTCHA screenshot).

---

### Per-opportunity output (JSON)

```json
{
  "title": "",
  "type": "hackathon|internship|event|certificate|program",
  "url": "OFFICIAL APPLY/REGISTER PAGE ONLY",
  "image_url": "optional — bot-friendly page for cover image fetch",
  "org": "",
  "location": "",
  "dates": [
    { "label": "Registration closes", "date": "YYYY-MM-DD" },
    { "label": "Event", "date": "YYYY-MM-DD", "endDate": "YYYY-MM-DD" }
  ],
  "blurb": "1 sentence max — who it's for + why apply",
  "beginner_friendly": true,
  "source_platform": "e.g. Devpost",
  "source_url": "where you found it",
  "confidence": "High|Medium|Low"
}
```

### Rules

- **Blurbs:** one sentence. Dates carry the detail. Use commas, colons, or periods; no em dashes.
- **No single deadline**: labeled `dates` array (registration vs event vs program period).
- **Rolling** → omit deadline dates; say "rolling" in blurb.
- **`beginner_friendly`:** `true` if no prior domain experience is expected (intro courses, first hackathons, open certs); `false` if advanced/CPD/pro-only; omit if unclear.
- **Dedupe** across platforms (Devpost + Devfolio = one entry).
- **Devpost online hackathons:** after harvest, run `npm run opportunities:devpost` to pull open global online listings from the Devpost API (skips invite-only and region-locked titles).
- **Reject** anti-patterns from source map: Indeed/LinkedIn search URLs, Scribd PDFs, aggregator blogs as canonical `url`.
- **Internships from Prosple/JobStreet/Kalibrr:** Prosple finds the employer; `url` must be the employer ATS or careers site.
- **Government:** canonical domains only: `ched.gov.ph`, `bpms.ched.gov.ph`, `sei.dost.gov.ph`, `dict.gov.ph`, `e-tesda.gov.ph`, `gsis.gov.ph`.
- Sort by nearest actionable date; flag items closing in **48 hours** at top.

### Also include

1. **Coverage report**: items per source bucket; gaps stated honestly 
2. **Bot-risk audit**: entries where you set `image_url` ≠ `url` and why 
3. **Horizon radar**: 10 opportunities opening in 30–60 days 
4. **Bibliography** with access dates 

Asia/Manila (UTC+8). Today: **[INSERT DATE]**.
