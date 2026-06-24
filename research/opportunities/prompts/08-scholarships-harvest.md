# Prompt 8 — Scholarships harvest

**Phase:** Supplemental · **Run in:** Google Deep Research (or Cursor agent browse)  
**Prerequisite:** optional [01-source-map.md](./01-source-map.md) or [02-harvest.md](./02-harvest.md) for board context  
**Save output as:** `responses/YYYY-MM-DD-scholarships-harvest.md`  
**Then:** verify URLs on official portals, merge with `npm run opportunities:merge` or add to `src/data/opportunity-board-items.js`, run `npm run opportunities:images`

Use this for a **scholarships-only pass**: government grants, corporate scholarships, embassy and multilateral study awards, university merit aid, and funded training grants for students in the Philippines.

Complements the broad [02-harvest.md](./02-harvest.md) (which touches gov scholarships in one bucket). For humanities/civic **non-scholarship** programs, use [07-humanities-sociocivic-harvest.md](./07-humanities-sociocivic-harvest.md).

---

## Research task

Find **actionable scholarships and study grants** worth listing on **stimmie.dev/opportunities** for **Filipino students and early-career learners** (high school seniors entering college, undergraduates, vocational trainees, and recent grads pursuing Master's or funded training).

**Time window:** application period **open now** or **opening within 60 days**. Include only programs whose primary application deadline is still in the future as of today. Rolling intakes are OK if the official page confirms ongoing applications.  
**Today (Asia/Manila, UTC+8):** **[INSERT DATE]**

### What counts as a scholarship (include)

| Category | Examples | Typical `type` |
| -------- | -------- | -------------- |
| **National government** | CHED merit/TES programs, DOST-SEI (RA 7687, Merit), GSIS, OWWA dependent grants, UniFAST TES | `program` |
| **LGUs & agencies** | Provincial/city scholarship boards, DSWD, DA, DENR youth grants | `program` |
| **Corporate & foundation** | SM Foundation, Ayala Foundation, BPI, Metrobank, Grab, Globe, Jollibee, San Miguel | `program` |
| **Embassy & bilateral** | Chevening, Fulbright, Australia Awards, Japan MEXT, Korea GKS, Taiwan MOE, US exchange scholarships — **degree** grants here; semester/youth mobility → [10-exchange-programs-harvest.md](./10-exchange-programs-harvest.md) |
| **Multilateral & intl** | ADB-Japan, World Bank JJ/WBG, Erasmus+, ASEAN scholarships | `program` |
| **University-specific** | UP, Ateneo, DLSU, UST, state U merit/aid calls with public application pages | `program` |
| **Vocational & TESDA** | TESDA scholarship slots, dual-training grants, industry-sponsored TVET | `program` or `certificate` |
| **Field-specific** | Aviation, maritime, medicine, law, agriculture, teacher education priority grants | `program` |
| **Cert & training grants** | Fully funded cert cohorts (Grow with Google, AWS AI/ML Scholars, Coursera/DICT pathways) when framed as competitive scholarships | `program` or `certificate` |

### Hard excludes

- **Expired** application windows (opening in 61–120 days → horizon radar only)
- **Automatic entitlements** with no application step (e.g. RA 10931 free SUC tuition in general; list only if a separate **apply** portal exists, such as TES)
- **Student loans** and pay-later products (unless a clear **grant + loan** hybrid with a named subsidy program page)
- **Pay-to-apply** scams, WhatsApp/Telegram "scholarship" groups, unverified Facebook posts
- `governmentph.com`, Scribd mirrors, scholarship aggregator blogs as canonical `url`
- News articles **about** a scholarship without a link to the current apply portal
- Generic scholarship search engines (InternationalScholarships.com search results) as `url`
- **Discount coupons** or single-course vouchers with no competitive selection
- Internships mislabeled as scholarships (→ general harvest)

### Core rule

**Boards discover; official pages publish.** Every entry needs a canonical **`url`**:

- CHED → `bpms.ched.gov.ph`, regional `*.ched.gov.ph` program pages, or `ched.gov.ph` official releases linking to BPMS
- DOST → `sei.dost.gov.ph`, `ugs.science-scholarships.ph`, or official DOST regional pages
- Embassies → `ph.usembassy.gov`, `britishcouncil.ph`, `australia.org.ph`, etc.
- Corporate → foundation or careers CSR page with apply form/PDF on official domain
- Universities → `.edu.ph` registrar or scholarship office page

If BPMS/portal is behind Cloudflare, set `image_url` to an official press page on the same `.gov.ph` domain (see [02-harvest.md](./02-harvest.md) bot rules).

---

## Mandatory source coverage

Search each bucket; report counts in the **coverage report**:

| Bucket | Official entry points |
| ------ | --------------------- |
| **CHED** | [bpms.ched.gov.ph](https://bpms.ched.gov.ph/), `ched.gov.ph` scholarship advisories, regional CHED offices |
| **DOST-SEI** | [sei.dost.gov.ph](https://www.sei.dost.gov.ph/), [ugs.science-scholarships.ph](https://ugs.science-scholarships.ph/) |
| **UniFAST / TES** | [unifast.gov.ph](https://unifast.gov.ph/), TES application announcements |
| **GSIS** | [gsis.gov.ph](https://www.gsis.gov.ph/) educational assistance |
| **OWWA** | [owwa.gov.ph](https://owwa.gov.ph/) education and training programs |
| **TESDA** | [e-tesda.gov.ph](https://www.e-tesda.gov.ph/), TESDA regional scholarship calls |
| **DICT / industry** | `dict.gov.ph`, Grow with Google PH, AWS Educate/Scholars announcements |
| **Embassies** | US, UK, Australia, Japan, Korea, EU delegation PH sites |
| **Corporations** | SM Foundation, Ayala Foundation, BPI Foundation, Metrobank Foundation, corporate CSR pages |
| **Universities** | Top SUCs and private U scholarship offices (public intake pages) |
| **LGUs** | Quezon City, Makati, Pasig, Cebu, Davao scholarship portals (sample major cities) |
| **Reddit / FB signal** | `r/Philippines`, `r/PhStudents`, campus FB groups → corroborate to official portal |

**Search queries (copy-paste):**

```text
site:bpms.ched.gov.ph scholarship 2026 application
site:sei.dost.gov.ph OR site:ugs.science-scholarships.ph scholarship 2026 deadline
site:unifast.gov.ph TES application 2026
site:gsis.gov.ph educational assistance 2026
site:owwa.gov.ph scholarship OR education grant 2026
"scholarship" site:sm-foundation.org OR site:ayalafoundation.org 2026 apply
site:ph.usembassy.gov scholarship OR fulbright 2026
site:britishcouncil.ph chevening 2026 philippines
"Australia Awards" Philippines 2026 application
site:ched.gov.ph merit scholarship OR bagong pilipinas 2026
corporate scholarship Philippines 2026 GWA apply deadline
site:tesda.gov.ph scholarship slot 2026
"ADB-Japan Scholarship" OR "JJ/WBG" Philippines 2026
scholarship high school senior STEM Philippines 2026 official
```

---

## Per-opportunity output (one JSON block each)

Use **`type`: `"program"`** for degree and grant scholarships. Use **`certificate`** only for funded cert cohorts with a named scholarship selection process.

```json
{
  "title": "",
  "type": "program",
  "url": "OFFICIAL APPLICATION PORTAL OR PROGRAM PAGE ONLY",
  "image_url": "optional bot-friendly page when url blocks bots",
  "org": "",
  "location": "Philippines / Japan / Global / Online",
  "scholarship_level": "hs_senior|college_freshman|undergrad|graduate|vocational|open",
  "coverage": "full_tuition|partial_tuition|allowance|training_grant|mixed",
  "benefit_summary": "e.g. up to 104k per semester plus allowance",
  "field_restriction": "STEM|any|priority_courses|social_sciences|etc or omit if open",
  "eligibility_notes": "optional: GWA floor, income cap, exam required",
  "dates": [
    { "label": "Application opens", "date": "YYYY-MM-DD" },
    { "label": "Application deadline", "date": "YYYY-MM-DD" }
  ],
  "blurb": "1 sentence: who qualifies, what you get, key requirement. No em dashes.",
  "beginner_friendly": true,
  "source_platform": "e.g. BPMS, DOST-SEI, embassy",
  "source_url": "where you found it",
  "confidence": "High|Medium|Low",
  "skip_reason": "only if rejected"
}
```

### Rules

- **Blurbs:** one sentence. Put peso amounts and GWA floors in `benefit_summary` or `eligibility_notes` when known.
- **Quantify benefits** where possible (₱/semester, monthly allowance, full tuition + stipend).
- **`beginner_friendly`:** `true` if open to first-time applicants or no prior degree in field required.
- **Dedupe** CHED regional reposts vs BPMS national portal (prefer BPMS when same program).
- **DOST RA 7687 vs Merit:** separate entries only if distinct application windows; otherwise one entry with notes.
- **Corporate:** verify the program is for **2026** (or current cycle), not a recycled 2024 blog post.
- Sort by **nearest application deadline**; flag closing in **7 days** at top.
- Target **30–50 verified scholarships** when intake season is active; state honestly if thin.

Harvest-only fields (`scholarship_level`, `coverage`, `benefit_summary`, `field_restriction`, `eligibility_notes`) may be dropped on merge if the board schema has no slot; keep them in the harvest file for verification.

---

## Output sections (required)

1. **Coverage report** — counts per bucket (gov, corporate, embassy, university, vocational); rejected + why  
2. **Closing this week** — deadline ≤ 7 days  
3. **National government** — JSON blocks  
4. **LGUs & agencies** — JSON blocks  
5. **Corporate & foundations** — JSON blocks  
6. **Embassy, bilateral & multilateral** — JSON blocks  
7. **University-specific** — JSON blocks  
8. **Vocational, TESDA & training grants** — JSON blocks  
9. **Low-confidence queue** — FB-only posts, aggregator mirrors, unclear eligibility  
10. **Horizon radar** — applications opening in 61–120 days  
11. **Bibliography** — URLs with access dates  

### Citations (non-negotiable)

Every JSON block must include **`source_url`**. The bibliography must list each verification URL with an **access date**.

---

## Verify against existing board

Before merge, compare with `src/data/opportunity-board-items.js`. Mark each entry **new**, **update** (deadline/benefit changed), or **skip** (duplicate). Flag bad URLs on the live board (e.g. `governmentph.com` mirrors) for replacement.

---

## After harvest

1. Re-check deadlines on BPMS, DOST, and embassy portals (dates shift without notice).  
2. Merge into `src/data/opportunity-board-items.js`; bump `opportunitiesBoard.lastUpdated`.  
3. `npm run opportunities:images` — prefer `image_url` on bot-friendly `.gov.ph` press pages when portals block screenshots.

---

Asia/Manila (UTC+8). Today: **[INSERT DATE]**.
