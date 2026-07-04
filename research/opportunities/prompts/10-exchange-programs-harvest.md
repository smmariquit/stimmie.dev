# Prompt 10: Exchange programs harvest

**Phase:** Supplemental · **Run in:** Google Deep Research (or Cursor agent browse) 
**Prerequisite:** optional [01-source-map.md](./01-source-map.md) or [02-harvest.md](./02-harvest.md) for board context 
**Save output as:** `responses/YYYY-MM-DD-exchange-programs-harvest.md` 
**Then:** verify URLs and eligibility, merge into `src/data/opportunity-board-items.js`, run `npm run opportunities:images`

Use this for **student exchange and mobility programs**: semester abroad, youth delegations, cultural exchanges, short-term study visits, and university bilateral swaps where Filipinos **study or represent the PH overseas** (or join inbound exchanges open to PH students).

**Not this prompt:** full-degree scholarships with no mobility component (→ [08-scholarships-harvest.md](./08-scholarships-harvest.md)). Campus org events (→ [09-campus-university-events-harvest.md](./09-campus-university-events-harvest.md)). Youth policy summits that are local only (→ [07-humanities-sociocivic-harvest.md](./07-humanities-sociocivic-harvest.md)).

---

## Research task

Find **actionable exchange and mobility programs** worth listing on **stimmie.dev/opportunities** for **Filipino students and recent graduates** (high school, undergraduate, graduate, and early-career where programs allow).

**Time window:** application period **open now** or **opening within 60 days**. Program start may be later (next semester or summer). Include only calls whose application deadline is still in the future as of today. 
**Today (Asia/Manila, UTC+8):** **[INSERT DATE]**

### What counts (include)

| Category | Examples | Typical `type` |
| -------- | -------- | -------------- |
| **University semester exchange** | UPD/OIA, DLSU, Ateneo, UST outbound mobility; partner university swap programs; tuition-waiver exchanges | `program` |
| **Government mobility** | CHED outbound programs, DOST international training visits, DepEd youth exchanges when publicly listed | `program` |
| **Embassy & bilateral youth** | JENESYS, IVLP youth tracks, UK exchange schemes, Australia short programs, Taiwan Huayu, Korea exchange camps | `program` |
| **Youth delegations** | SSEAYP, ASEAN youth exchanges, ship-based or forum-based delegations with apply forms | `program` or `event` |
| **NGO cultural exchange** | AFS, Rotary Youth Exchange, CISV, youth ambassador programs with PH intake | `program` |
| **Summer / short-term abroad** | Summer schools, language immersions, faculty-led study tours with student application portals | `program` or `event` |
| **Virtual exchange** | Competitive cohort programs with cross-border projects (e.g. Soliya, virtual exchange alliances) | `program` |
| **Internship abroad (exchange-framed)** | IAESTE, AIESEC global talent, J1-style programs administered through official PH partners | `internship` or `program` |
| **Inbound to PH** | International student programs hosted in the PH that explicitly welcome Filipino participants or peer buddies with apply paths | `program` |

### Scholarship vs exchange (dedupe rule)

| List here (10) | List under scholarships (08) |
| -------------- | ---------------------------- |
| Semester swap, youth delegation, cultural visit, language camp | Full Master's/PhD abroad funded as a **degree scholarship** |
| Exchange with mobility + cultural component as primary framing | Chevening/Fulbright **degree** awards (unless a separate short exchange track exists) |
| University **outbound mobility** call via international office | CHED/DOST **tuition grants** with no travel component |

If a program is both (e.g. MEXT research student), list once under the best fit and note the other in `related_program`.

### Hard excludes

- **Expired** application windows (opening in 61–120 days → horizon radar)
- **Pay-to-play** "exchange" packages with no competitive selection or accredited partner (tourism disguised as exchange)
- **Au pair / domestic work** programs without student or trainee framing
- Visa-agency Facebook ads with no official program page
- `governmentph.com`, Scribd mirrors, study-abroad aggregators as canonical `url`
- Programs **not open to Philippine passport holders** (note in rejected list)
- Generic "study abroad consultants" with no named program intake

### Core rule

**Boards discover; official pages publish.** Every entry needs a canonical **`url`**:

- University → international office / OIA / global engagement `.edu.ph` page
- Embassy → embassy or cultural institute official program page
- NGO → `afs.org`, Rotary district, official national chapter site
- Government → `ched.gov.ph`, `deped.gov.ph`, `dfa.gov.ph` program posts linking to apply

---

## Mandatory source coverage

Search each bucket; report counts in the **coverage report**:

| Bucket | Where to look |
| ------ | ------------- |
| **UP System** | [oia.up.edu.ph](https://oia.up.edu.ph/), constituent campus international offices (UPD, UPLB, UPM) |
| **DLSU** | DLSU International Center, Animo Global, outbound mobility announcements |
| **Ateneo** | Ateneo Internationalization, OAA global programs |
| **UST** | UST Office for International Relations and Programs |
| **Other PH universities** | PUP, FEU, La Salle schools, Mapúa, Benilde global pages |
| **CHED** | CHED internationalization / outbound mobility advisories |
| **Japan** | JENESYS, JASSO student exchange, MEXT short-term, Japan Foundation Manila |
| **US** | EducationUSA Philippines, Youth Exchange & Study (YES), Global UGRAD, embassy exchanges |
| **UK / EU** | British Council PH, Erasmus+ calls open to PH partners, Chevening **short** programs if distinct from degree |
| **Australia / NZ** | Australia Awards short courses, New Zealand MFAT programs |
| **Korea / Taiwan** | GKS exchange tracks, Huayu Enrichment, KOFICE youth |
| **ASEAN** | ASEAN Secretariat youth, SSEAYP Philippines committee |
| **NGOs** | AFS Philippines, Rotary Youth Exchange District 3780/3850, IAESTE Philippines |
| **Multilateral** | UNESCO, UN youth exchanges, ADB youth programs with travel |

**Search queries (copy-paste):**

```text
site:oia.up.edu.ph OR site:up.edu.ph exchange program application 2026
"student exchange" (DLSU OR "De La Salle" OR Ateneo OR UST) outbound 2026 apply
site:ched.gov.ph outbound OR exchange OR mobility 2026
JENESYS Philippines youth application 2026
site:educationusa.state.gov Philippines exchange OR undergraduate
SSEAYP Philippines application 2026
site:afs.org Philippines intercultural exchange
IAESTE Philippines internship abroad apply
Rotary Youth Exchange Philippines 2026
Erasmus+ exchange Philippines partner 2026
"summer school" abroad scholarship Philippines students apply 2026
British Council Philippines exchange program 2026
virtual exchange program apply students Philippines 2026
```

---

## Per-opportunity output (one JSON block each)

Use **`type`: `"program"`** unless it is a fixed-date delegation with no ongoing intake (then `event`).

```json
{
  "title": "",
  "type": "program",
  "url": "OFFICIAL APPLICATION OR PROGRAM PAGE ONLY",
  "image_url": "optional bot-friendly page (embassy, university, org homepage)",
  "org": "",
  "location": "Destination country or Online + host city",
  "exchange_type": "semester_abroad|summer|youth_delegation|virtual|internship_abroad|language|faculty_led",
  "duration": "e.g. 2 weeks, 1 semester, 10 months",
  "funding": "fully_funded|partial|self_funded|unspecified",
  "eligible_level": "hs|undergrad|graduate|recent_grad|open",
  "dates": [
    { "label": "Application deadline", "date": "YYYY-MM-DD" },
    { "label": "Program start", "date": "YYYY-MM-DD", "endDate": "YYYY-MM-DD" }
  ],
  "blurb": "1 sentence: who can apply, where you go, what is covered. No em dashes.",
  "beginner_friendly": true,
  "source_platform": "university|embassy|ngo|government|multilateral",
  "source_url": "where you found it",
  "confidence": "High|Medium|Low",
  "related_program": "optional cross-link e.g. also see Chevening degree track",
  "skip_reason": "only if rejected"
}
```

### Rules

- **Blurbs:** one sentence. Put duration and funding in dedicated fields when known.
- **`funding`:** `fully_funded` only when airfare, tuition, or stipend is explicitly covered.
- **`eligible_level`:** note if program requires nomination through home university (common for semester exchange).
- **Nomination flows:** if students apply via their school's international office, say so in blurb and set `url` to the **official mobility page** (not a random FB post).
- **Dedupe** embassy reposts vs university partner pages (one entry, best apply URL).
- Sort by **nearest application deadline**; flag closing in **7 days** at top.
- Target **20–35 verified programs** when intake season is active.

Harvest-only fields (`exchange_type`, `duration`, `funding`, `eligible_level`, `related_program`) may be dropped on merge.

---

## Output sections (required)

1. **Coverage report**: counts per bucket; PH-eligible confirmed Y/N; rejected + why 
2. **Closing this week**: deadline ≤ 7 days 
3. **University outbound mobility**: JSON blocks 
4. **Embassy & bilateral exchanges**: JSON blocks 
5. **Youth delegations & ASEAN**: JSON blocks 
6. **NGO & cultural exchange**: JSON blocks 
7. **Summer, virtual & short-term**: JSON blocks 
8. **Internship-abroad (exchange-administered)**: JSON blocks 
9. **Low-confidence queue**: FB-only, agency reposts, unclear PH eligibility 
10. **Horizon radar**: applications opening in 61–120 days 
11. **Bibliography**: URLs with access dates 

### Citations (non-negotiable)

Every JSON block must include **`source_url`**. Bibliography must list verification URLs with **access dates**.

---

## After harvest

1. Confirm **Philippine eligibility** on the official page (many exchanges are country-restricted). 
2. Cross-check [08-scholarships-harvest.md](./08-scholarships-harvest.md) output to avoid duplicate degree-scholarship entries. 
3. Merge into board; bump `opportunitiesBoard.lastUpdated`. 
4. For university exchanges, note if **GWA or language requirement** exists in `blurb` or harvest notes.

---

Asia/Manila (UTC+8). Today: **[INSERT DATE]**.
