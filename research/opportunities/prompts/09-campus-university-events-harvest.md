# Prompt 9 — Campus & university events harvest

**Phase:** Supplemental · **Run in:** Google Deep Research (or Cursor agent browse)  
**Prerequisite:** optional [01-source-map.md](./01-source-map.md) for source context  
**Save output as:** `responses/YYYY-MM-DD-campus-events-harvest.md`  
**Then:** verify dates and RSVP links, merge into `src/data/opportunity-board-items.js`, run `npm run opportunities:images`

Use this for **university-specific events and org activities** in the Philippines: campus hackathons, org general assemblies, workshops, career fairs, symposiums, competitions, and volunteer drives posted by **student organizations** on Facebook and sister channels.

Most discovery starts on **Facebook**; your job is to corroborate dates, find the best public RSVP link, and flag access rules (open campus vs ID-only).

Complements [02-harvest.md](./02-harvest.md) (Campus / FB bucket). For national Devpost hackathons, use [03-devpost-harvest.md](./03-devpost-harvest.md). For contributor roles (judges, speakers), use [06-contributor-calls-harvest.md](./06-contributor-calls-harvest.md).

---

## Research task

Find **upcoming campus events** worth listing on **stimmie.dev/opportunities** for **students and early-career builders** who can attend in person (or hybrid with a clear online RSVP).

**Time window:** event date or registration deadline within the **next 45 days**, still in the future as of today.  
**Today (Asia/Manila, UTC+8):** **[INSERT DATE]**

### What counts (include)

| Category | Examples | Typical `type` |
| -------- | -------- | -------------- |
| **Tech & builder orgs** | GDG on Campus, AWS Cloud Club, JPCS, MISOSA, CSS, programming guilds, hackathons, build nights | `hackathon` or `event` |
| **Career & industry** | Job fairs, company info sessions, resume clinics, OJT orientations (open signup) | `event` |
| **Academic & research** | College symposia, research fairs, thesis forums, department lecture series with public RSVP | `event` |
| **Design & media** | UXPH student chapters, film org screenings, designathons, publication launches | `event` |
| **Business & entrepreneurship** | BIZMATES, IDEA, startup weekends on campus, pitch nights | `event` |
| **Civic & humanities orgs** | Debate tournaments, MUN on campus, legal aid clinics, volunteer drives with a form | `event` |
| **University-wide** | Org fairs, university week, foundation day activities with external guest registration | `event` |
| **Cross-campus collabs** | Inter-university competitions hosted at a named campus with public registration | `hackathon` or `event` |

### Hard excludes

- **Past** events (date already passed)
- **Private Facebook groups** you cannot verify without membership (note in horizon only if a public Page mirrors it later)
- Posts with **no date** and no registration link
- Memes, alumni homecoming **without** a public guest path, or closed org-only GAs **unless** they welcome outsiders (mark `access: org_members` and skip listing)
- Scammy "earn money" campus promos
- National events not tied to a campus host (→ general harvest)
- Devpost-only hackathons with no campus host (→ [03](./03-devpost-harvest.md))

### Core rule (Facebook-heavy)

**Discover on Facebook; publish the best public link.**

Priority order for `url`:

1. **Luma / Eventbrite / university events portal** (`.edu.ph` calendar)
2. **Google Form / Typeform** on an org or university domain
3. **Devpost / Devfolio** for campus hackathons
4. **Org Linktree / Carrd / Notion** with RSVP
5. **Public Facebook Event** URL (`facebook.com/events/...`) when nothing else exists
6. **Instagram bio link** only if it lands on one of the above

Always record **`facebook_source_url`** when Facebook was how you found it, even if `url` is a Luma link.

**Confidence:**

| Confidence | When |
| ---------- | ---- |
| **High** | Luma, `.edu.ph`, Devpost, or form on official org/university domain |
| **Medium** | Public FB Event with clear date + venue + registration CTA |
| **Low** | FB post only, screenshot date, or "DM to register" → **low-confidence queue** unless you find a form |

---

## Mandatory university coverage

Search **each campus** below. Report counts per university in the **coverage report**.

### Priority universities (search all)

| Code | Campus | Facebook / web entry points |
| ---- | ------ | ----------------------------- |
| **UPD** | UP Diliman | UP Diliman official page, UPD CS orgs, UP System ITTC, college FB pages (`cs.upd.edu.ph` orgs) |
| **UPLB** | UP Los Baños | UPLB official, CAS/UPLB ICS, GDG on Campus UPLB, UPLB Dev societies |
| **UPM** | UP Manila | UPM official, college org pages, health/campus tech groups |
| **DLSU** | De La Salle Manila (Taft) | DLSU official, Animo Labs, MISOSA, GDG on Campus DLSU, college orgs |
| **ADMU** | Ateneo de Manila | Ateneo official, Ateneo CS orgs, `{AUX} Labs`, school/college pages |
| **UST** | University of Santo Tomas | UST official, faculty/org pages, Thomasian tech & business orgs |
| **PUP** | Polytechnic U of the Philippines | PUP official, GDG on Campus PUP, college orgs |
| **FEU** | Far Eastern University | FEU official, Institute of Tech orgs |
| **NU** | National University | NU official, tech/business student orgs |
| **Mapúa** | Mapúa University | Mapúa official, IEEE/CS student branches |

### Also check when time allows

UP Cebu, UP Baguio, UP Mindanao, Adamson, UAP, Benilde (CSB), UE, PLM, PNU, MSU-IIT, ADMU Loyola schools' public collabs, DLSU Laguna.

### Org archetypes (search on each campus)

- `GDG on Campus` + university name
- `AWS Cloud Club` / `AWS Student Community` + campus
- `Junior Philippine Computer Society` / `JPCS` + chapter
- `Microsoft Learn Student Ambassadors` + campus
- College **CS / IT / CCS** student council & org week
- **Symposium**, **hackathon**, **ideathon**, **career fair**, **general assembly** (open guests)
- Department **Facebook Page** (not personal accounts)

**Facebook search queries (copy-paste per campus):**

```text
site:facebook.com "UP Los Baños" OR UPLB (hackathon OR symposium OR "register now") 2026
site:facebook.com "De La Salle" OR DLSU (workshop OR hackathon OR "career fair") 2026
site:facebook.com "Ateneo" (GDG OR hackathon OR symposium OR org week) 2026
site:facebook.com UST (hackathon OR seminar OR "call for") 2026
site:facebook.com "UP Diliman" OR UPD (event OR hackathon OR registration) 2026
"GDG on Campus" (UPLB OR UPD OR DLSU OR Ateneo OR UST OR PUP) 2026
site:lu.ma (UPLB OR "De La Salle" OR Ateneo OR UST OR "UP Diliman")
site:facebook.com/events ("University of the Philippines" OR "De La Salle" OR Ateneo OR UST) Manila
```

**Non-Facebook corroboration:**

- `site:*.edu.ph` event calendars and college news pages
- Org **Linktree** / Instagram bios linked from FB Pages
- [GDG community map](https://gdg.community.dev/) for on-campus chapters

---

## Per-opportunity output (one JSON block each)

```json
{
  "title": "",
  "type": "hackathon|event|internship|program",
  "url": "BEST PUBLIC RSVP LINK (see priority order above)",
  "facebook_source_url": "optional — public FB post or event where you found it",
  "image_url": "optional — org Page cover, Luma OG, or .edu.ph event graphic",
  "org": "e.g. GDG on Campus UPLB",
  "university": "UPLB|UPD|DLSU|ADMU|UST|PUP|FEU|NU|Mapúa|other",
  "location": "Campus building or Online/Hybrid — city",
  "access": "open_public|students_any_school|host_university_students|invite_only",
  "dates": [
    { "label": "Registration closes", "date": "YYYY-MM-DD" },
    { "label": "Event", "date": "YYYY-MM-DD", "endDate": "YYYY-MM-DD" }
  ],
  "blurb": "1 sentence: what happens, who should come, how to register. No em dashes.",
  "beginner_friendly": true,
  "source_platform": "facebook|luma|google_forms|devpost|edu_portal",
  "source_url": "first link you found (often Facebook)",
  "confidence": "High|Medium|Low",
  "skip_reason": "only if rejected"
}
```

### Rules

- **Blurbs:** one sentence. Put venue and access in `location` / `access`, not the blurb.
- **`access`:** if ID or enrollment at host university is required, set `host_university_students` and say so in the blurb.
- **Hybrid:** note if online attendees need approval (common on Luma).
- **Dedupe** the same event across FB, IG, and Luma (one entry, best `url`).
- **Org week / multiple sub-events:** list **separate entries** only when each has its own RSVP; otherwise one umbrella entry with date range.
- Sort by **nearest event or registration deadline**; flag **closing in 7 days** at top.
- Target **5–15 verified events per priority university** when orgs are active; state honestly if a campus is quiet.

Harvest-only fields (`university`, `access`, `facebook_source_url`) may be dropped on merge; keep them in the harvest file.

---

## Output sections (required)

1. **Coverage report** — event count **per university**; FB Pages checked; rejected + why  
2. **Closing this week** — registration or event within 7 days  
3. **UP System** (UPD, UPLB, UPM, + others found) — JSON blocks  
4. **DLSU** — JSON blocks  
5. **Ateneo (ADMU)** — JSON blocks  
6. **UST** — JSON blocks  
7. **Other universities** (PUP, FEU, NU, Mapúa, etc.) — JSON blocks  
8. **Low-confidence queue** — FB-only, DM to register, missing dates  
9. **Horizon radar** — announced but registration not open yet (30–60 days)  
10. **Bibliography** — URLs with access dates (include Facebook Page/Event URLs)  

### Citations (non-negotiable)

Every entry needs **`source_url`** (usually Facebook) plus the **`url`** readers should click. Bibliography must list both when they differ.

---

## Facebook workflow tips (for the agent)

1. Start from the university or org **Facebook Page** (blue check or long-standing page with org name).  
2. Open **Events** tab; filter upcoming.  
3. Open each event → check **Discussion** / description for Google Form or Luma link.  
4. Check Page **About** for website, Linktree, or Instagram.  
5. If the post is a **shared** flyer, trace to the **original** org Page.  
6. Do not invent dates; if the year is missing on a graphic, corroborate with a comment or org Story highlight.

---

## After harvest

1. Re-check RSVP links (campus events cancel or go waitlist-only fast).  
2. Prefer `image_url` from Luma or a public org banner (FB CDN URLs expire; use org website when possible).  
3. Merge into board; bump `opportunitiesBoard.lastUpdated`.  
4. Replace any national duplicate already listed from Devpost with the **campus-specific** entry if it adds venue/org context.

---

Asia/Manila (UTC+8). Today: **[INSERT DATE]**.
