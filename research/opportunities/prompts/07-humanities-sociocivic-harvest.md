# Prompt 7: Humanities & sociocivic harvest

**Phase:** Supplemental · **Run in:** Google Deep Research (or Cursor agent browse) 
**Prerequisite:** optional [01-source-map.md](./01-source-map.md) or [02-harvest.md](./02-harvest.md) for board context 
**Save output as:** `responses/YYYY-MM-DD-humanities-sociocivic-harvest.md` 
**Then:** verify URLs, merge with `npm run opportunities:merge` or add to `src/data/opportunity-board-items.js`, run `npm run opportunities:images`

Use this when you want a **humanities and sociocivic pass**: essay contests, fellowships, policy programs, civic leadership, journalism, debate/MUN, arts and culture, NGO internships, and social-impact opportunities that are **not primarily coding hackathons**.

Complements the tech-heavy [02-harvest.md](./02-harvest.md) and [03-devpost-harvest.md](./03-devpost-harvest.md). For contributor roles (CFP, judges, mentors), use [06-contributor-calls-harvest.md](./06-contributor-calls-harvest.md).

---

## Research task

Find **actionable opportunities** worth listing on **stimmie.dev/opportunities** for **students and early-career builders in the Philippines** whose interests skew **humanities, social sciences, civic engagement, policy, media, arts, and community work** (plus **online/global** programs they can join remotely).

**Time window:** applications, registrations, or submissions open now or opening within **45 days**. Include only calls whose primary deadline is still in the future as of today. 
**Today (Asia/Manila, UTC+8):** **[INSERT DATE]**

### Domain buckets (search each; report counts)

| Bucket | Examples | Typical `type` |
| ------ | -------- | -------------- |
| **Writing & essay** | Essay competitions, creative writing prizes, op-ed fellowships, youth journalism awards | `program` or `event` |
| **Debate & diplomacy** | Model UN conferences, parliamentary debate, moot court (undergrad), diplomacy simulations | `event` |
| **Civic & youth leadership** | YSEALI, Ayala Young Leaders, SK/youth council programs, campus leadership institutes, voter education fellowships | `program` |
| **Policy & governance** | Think-tank internships, legislative fellowships, local governance labs, disaster/resilience academies, climate policy challenges | `internship` or `program` |
| **Human rights & justice** | Legal aid volunteer intakes, human-rights documentation fellowships, access-to-justice clinics (student-eligible) | `program` or `internship` |
| **Media & MIL** | Investigative journalism fellowships, fact-checking labs, UNESCO/media literacy challenges, community radio training | `program` or `certificate` |
| **Arts & culture** | Film labs, photography contests, museum/archive internships, NCCA/KWF programs, literary magazine submissions with prizes | `event` or `program` |
| **Social impact build** | Civic-tech or social-good **competitions** where the deliverable is policy brief, campaign, documentary, or community project (not pure code) | `hackathon` or `event` |
| **NGO & multilateral** | UNDP, UNICEF, UN Volunteers, ADB youth, ASEAN internships with policy/communications/social development focus | `internship` or `program` |
| **Humanities scholarships** | CHED merit for social sciences, DOST non-STEM where applicable, Chevening, Fulbright → deep pass: [08-scholarships-harvest.md](./08-scholarships-harvest.md) | `program` |
| **Campus & PH orgs** | `.edu.ph` research fairs, social-science org calls: campus events: [09-campus-university-events-harvest.md](./09-campus-university-events-harvest.md) | `event` |

### Hard excludes

- **Expired** deadlines (opening in 30–60 days only → horizon radar)
- Pure **tech hackathons** with no humanities/civic track (→ [03-devpost-harvest.md](./03-devpost-harvest.md))
- **Game jams** (→ [05-game-jam-harvest.md](./05-game-jam-harvest.md))
- **Paid degree programs** marketed as "opportunities" (MA tuition ads)
- **Pay-to-win** essay mills, vanity awards, or competitions with mandatory purchase
- Generic job boards without a named program (Indeed/JobStreet search URLs)
- `governmentph.com`, Scribd mirrors, aggregator blogs as canonical `url`
- LinkedIn posts with no official apply page (discover only; find org site)
- Senior-only roles (10+ years) unless explicitly open to students/recent grads

### Core rule

**Boards discover; official pages publish.** Every entry needs a canonical **`url`**:

- Essay contest → organizer `/contest`, Submittable, Google Form on official org domain
- Fellowship/internship → employer or NGO **apply** page, not a news article about past fellows
- Conference/MUN → official registration page (MyMUN, conference site)
- Scholarship → `ched.gov.ph`, `sei.dost.gov.ph`, embassy NOFO, or program portal
- MOOC/certificate → UPOU MODeL, TESDA TOP, Coursera partner page with open enrollment

If the only link is Facebook, find the organizer website or official form first. Mark **Low confidence** if FB-only.

---

## Mandatory source coverage

Search each bucket; report item count in the **coverage report**:

| Source | Where to look |
| ------ | ------------- |
| **Embassies & intl programs** | `ph.usembassy.gov` NOFOs, YSEALI, Chevening, Fulbright, Japan Foundation Manila, British Council PH, Goethe-Institut |
| **Multilateral** | UNDP jobs/internships, UNICEF vacancies, UN Volunteers, ADB youth programs, ASEAN Secretariat |
| **PH government** | `ched.gov.ph`, `bpms.ched.gov.ph`, `ncca.gov.ph`, `kwf.gov.ph`, `nhcp.gov.ph`, `dict.gov.ph/trainings`, `tesda.gov.ph`, `dswd.gov.ph` youth programs |
| **Think tanks & NGOs** | Asia Foundation, IDEALS, PLCPD, ILO PH, PCIJ, Rappler MovePH, Foundation for Media Alternatives |
| **Universities** | UP, Ateneo, DLSU, UST public calls for student researchers, debate societies, moot court invites |
| **MUN & debate** | MyMUN, Best Delegate calendars, PH debate league pages, Asia World Model UN |
| **Writing & journalism** | Young Writers' Prize, Palanca (when open), investigative journalism fellowships, Pulitzer/ONA student categories with global intake |
| **Civic tech / social good** | Devpost filters: Social Good, Education, Health (read description; include only if deliverable is civic/policy/media, not app-only) |
| **Meetups & events** | Luma, Meetup: policy, journalism, civic tech, UX for social impact, DEVCON civic tracks |
| **Reddit signal** | `r/Philippines`, `r/Filipino`, `r/ModelUN`, `r/debate` → corroborate to official page |

**Search queries (copy-paste):**

```text
"essay competition" ("Philippines" OR Filipino OR youth) 2026 deadline apply
"call for fellows" (policy OR journalism OR human rights) Philippines 2026
site:ph.usembassy.gov NOFO YSEALI OR Chevening OR fellowship 2026
site:undp.org internship Philippines policy OR communications
"model united nations" Philippines 2026 registration
"moot court" undergraduate Philippines 2026
site:ched.gov.ph scholarship social sciences OR humanities 2026
site:ncca.gov.ph OR site:kwf.gov.ph program youth 2026
"young leaders" program Philippines apply 2026
"civic engagement" OR "youth council" fellowship Philippines
investigative journalism fellowship Asia apply 2026
site:devpost.com ("social good" OR education) ("policy" OR "civic" OR documentary) open
site:modelup.ph OR site:mymun.com Philippines 2026
"climate justice" OR "disaster resilience" fellowship youth Philippines
```

---

## Per-opportunity output (one JSON block each)

Use existing board types. Add harvest-only **`domain_tags`** for sorting (omit on merge if the board schema has no field).

```json
{
  "title": "",
  "type": "hackathon|internship|event|certificate|program",
  "url": "OFFICIAL APPLY / REGISTER / SUBMIT PAGE ONLY",
  "image_url": "optional bot-friendly page when url blocks bots",
  "org": "",
  "location": "Online / Manila / Hybrid / Global",
  "domain_tags": ["writing", "debate", "civic", "policy", "media", "arts", "human-rights", "scholarship", "ngo"],
  "dates": [
    { "label": "Applications close", "date": "YYYY-MM-DD" },
    { "label": "Program or event", "date": "YYYY-MM-DD", "endDate": "YYYY-MM-DD" }
  ],
  "blurb": "1 sentence: who it is for, discipline, and why apply. No em dashes.",
  "beginner_friendly": true,
  "source_platform": "e.g. embassy NOFO, MyMUN, NGO site",
  "source_url": "where you found it",
  "confidence": "High|Medium|Low",
  "skip_reason": "only if rejected"
}
```

### Rules

- **Blurbs:** one sentence. Dates carry the detail. No em dashes.
- **`beginner_friendly`:** `true` if first-time applicants, undergrads, or no prior field experience is expected.
- **`domain_tags`:** at least one tag per entry; use multiple when cross-cutting (e.g. `["media", "civic"]`).
- **Social-good hackathons:** include only when rules emphasize briefs, campaigns, documentaries, research, or community deliverables; note if coding is optional.
- **Internships:** `url` must be the NGO/employer apply page, not Prosple or LinkedIn search.
- **Government:** canonical `.gov.ph` or embassy domains only.
- **Dedupe** across Facebook + Luma + news coverage (one entry, best URL).
- Sort sections by **nearest deadline**; flag items closing in **7 days** at the top.
- Target **25–40 verified opportunities** when the ecosystem is active; state honestly if thin.

### Bot protection & cover images

Same rules as [02-harvest.md](./02-harvest.md): if `url` is a `.gov.ph` portal behind Cloudflare, set `image_url` to a press release or parent program page on the same domain.

---

## Output sections (required)

1. **Coverage report**: counts per domain bucket; sources checked; rejected + why 
2. **Closing this week**: deadline ≤ 7 days 
3. **Writing, media & essay**: JSON blocks 
4. **Debate, diplomacy & moot court**: JSON blocks 
5. **Civic, policy & leadership**: JSON blocks 
6. **Arts, culture & humanities programs**: JSON blocks 
7. **NGO, multilateral & internships**: JSON blocks 
8. **Scholarships & fellowships**: JSON blocks 
9. **Social-impact competitions** (non-code-primary): JSON blocks 
10. **Low-confidence queue**: FB-only, vague posts, missing apply links 
11. **Horizon radar**: opens in 30–60 days 
12. **Bibliography**: URLs with access dates 

### Citations (non-negotiable)

Every JSON block must include **`source_url`**. The bibliography must list each verification URL with an **access date**. No entry without a traceable source.

---

## After harvest

1. Verify deadlines on official pages (embassy NOFOs and university forms go stale fast). 
2. Merge into `src/data/opportunity-board-items.js`; bump `opportunitiesBoard.lastUpdated`. 
3. `npm run opportunities:images` for card covers. 
4. Cross-check against existing board entries to avoid dupes (especially YSEALI, UNDP, UPOU MOOCs).

---

Asia/Manila (UTC+8). Today: **[INSERT DATE]**.
