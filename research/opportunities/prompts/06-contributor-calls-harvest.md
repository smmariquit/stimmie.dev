# Prompt 6: Contributor & sponsor calls (orchestrator)

**Phase:** Supplemental · **Run in:** Google Deep Research (or Cursor agent browse) 
**Prerequisite:** none (optional: [02-harvest.md](./02-harvest.md) for board context) 
**Save output as:** `responses/YYYY-MM-DD-contributor-calls-harvest.md` 
**Then:** verify URLs, merge with `npm run opportunities:merge` or add to `src/data/opportunity-board-items.js`, run `npm run opportunities:images`

---

## Recommended: run focused sub-prompts in parallel

The full harvest below is valid but heavy. For clearer results, run **one sub-prompt per role** (same day, parallel agents), then merge into one response file:

| Focus | Sub-prompt | Output section |
| ----- | ---------- | -------------- |
| Speakers / CFP | [06a-call-for-speakers-harvest.md](./06a-call-for-speakers-harvest.md) | Call for speakers |
| Judges & mentors | [06b-call-for-judges-mentors-harvest.md](./06b-call-for-judges-mentors-harvest.md) | Judges + mentors |
| Sponsors & funding | [06c-sponsorship-calls-harvest.md](./06c-sponsorship-calls-harvest.md) | Sponsor calls + programs |
| Volunteers | [06d-volunteer-facilitator-harvest.md](./06d-volunteer-facilitator-harvest.md) | Volunteers / facilitators |

**Merge step:** dedupe by canonical `url`, combine coverage reports, sort each section by nearest deadline.

**Success metrics (per run):** ≥3 verified entries per sub-prompt when the ecosystem is active; every entry has an official apply URL; zero expired deadlines in the main sections; bibliography with access dates for each source.

---

## What this harvest covers

Opportunities that are **not “sign up as a participant”** but **contributor or org-level roles**:

- Call for speakers · judges · mentors · volunteers/facilitators
- Call for sponsors (event recruiting partners)
- Sponsorship programs (funds campus orgs apply to)

**Audience:** students, early-career builders, and campus org officers in the Philippines (plus online/global calls they can join remotely).

**Time window:** applications open now or opening within **45 days**; deadline must still be in the future as of today. 
**Today (Asia/Manila, UTC+8):** **[INSERT DATE]**

---

## Shared rules (all sub-prompts)

### Hard excludes

- Participant registration disguised as “call for speakers” with no separate speaker track
- **Expired** deadlines (opening within 60 days only → horizon radar)
- Paid speaking slots / pay-to-present with no free student tier
- Generic “contact us” with no form, email, or PDF prospectus → **Low confidence** queue only
- LinkedIn posts with no canonical landing page (discover only; find org site)
- `governmentph.com`, Scribd mirrors, aggregator blogs as canonical `url`
- Senior-only judging (10+ years) unless students/alumni judges are explicitly welcome

### Core rule

**Boards discover; official pages publish.** Every entry needs a canonical **`url`**:

- Speaker CFP → Sessionize, Papercall, Pretalx, or conference `/cfp` page
- Judge / mentor → Google Form, Typeform, Devpost description anchor, or `/volunteer` page
- Call for sponsors → sponsorship deck, `mailto:sponsors@...`, or “Partner with us” on official event site
- Sponsorship program → corporate/community **apply** page (not a news article about someone else getting funded)

Organizer Google Drive PDF is OK if it is the official prospectus.

### Citations (non-negotiable)

Every JSON block must include **`source_url`** (where you found it) and the bibliography must list each URL with an **access date**. No entry without a traceable source.

---

## JSON schema (shared)

### Contributor calls (speakers, judges, mentors, volunteers)

Use **`type`: `"event"`**. Title: **`Call for {Role}s: {Event or Program Name}`**.

```json
{
  "title": "Call for Speakers — Example Conference 2026",
  "type": "event",
  "url": "OFFICIAL APPLY / CFP / FORM PAGE ONLY",
  "image_url": "optional bot-friendly page (event homepage, not login-gated)",
  "org": "",
  "location": "Online / Manila / Hybrid",
  "contributor_role": "speaker|judge|mentor|panelist|volunteer|facilitator",
  "dates": [
    { "label": "Applications close", "date": "YYYY-MM-DD" },
    { "label": "Event", "date": "YYYY-MM-DD", "endDate": "YYYY-MM-DD" }
  ],
  "blurb": "1 sentence: role, who should apply, commitment. No em dashes.",
  "beginner_friendly": true,
  "source_platform": "Sessionize|Devpost|Luma|organizer",
  "source_url": "where you found it",
  "confidence": "High|Medium|Low",
  "parent_event_url": "optional main event if different from apply url"
}
```

`beginner_friendly`: `true` if students, first-time speakers, or early-career mentors are welcome.

### Call for sponsors (event-level)

```json
{
  "title": "Call for Sponsors — Example Hackathon 2026",
  "type": "event",
  "url": "SPONSORSHIP PAGE, PROSPECTUS, OR PARTNER INQUIRY FORM",
  "org": "Organizer name",
  "location": "Manila / Online / Hybrid",
  "contributor_role": "sponsor",
  "dates": [{ "label": "Sponsorship inquiries close", "date": "YYYY-MM-DD" }],
  "blurb": "1 sentence: event, sponsor tiers if known, who should inquire. No em dashes.",
  "source_url": "where you found it",
  "confidence": "High|Medium|Low",
  "sponsor_notes": "optional: student orgs welcome, in-kind only"
}
```

### Sponsorship programs (org applies for funding)

```json
{
  "title": "Google Developer Groups Community Support",
  "type": "program",
  "url": "OFFICIAL PROGRAM APPLY PAGE",
  "org": "Google Developer Groups",
  "location": "Philippines / Global",
  "dates": [
    { "label": "Applications open", "date": "YYYY-MM-DD" },
    { "label": "Applications close", "date": "YYYY-MM-DD" }
  ],
  "blurb": "1 sentence: funding offered and which orgs qualify. No em dashes.",
  "source_url": "where you found it",
  "confidence": "High|Medium|Low",
  "org_level": true
}
```

---

## Full harvest (single run)

Use this section only if you are **not** running 06a–06d separately.

### Mandatory source coverage

| Bucket | Where to look |
| ------ | ------------- |
| Hackathons | Devpost ([03](./03-devpost-harvest.md) automates judge/mentor scan), Devfolio, MLH |
| Conferences / CFP | Sessionize, Papercall, DEVCON PH, PyCon PH, UXPH, campus summits |
| Meetups | luma.com, Meetup chapters |
| Campus / PH | GDG on Campus, MLSA, AWS Cloud Clubs, JPCS, `.edu.ph` orgs |
| Sponsors | Event prospectuses, partner inquiry forms, corporate CSR apply pages |
| Email | [04-email-inbox-harvest.md](./04-email-inbox-harvest.md) for “CFP”, “judge”, “mentor”, “sponsor” |

### Output sections (required)

1. **Coverage report**: counts per role; rejected + why 
2. **Closing this week**: deadline ≤ 7 days 
3. **Call for speakers**: JSON blocks 
4. **Call for judges**: JSON blocks 
5. **Call for mentors & facilitators**: JSON blocks 
6. **Call for sponsors**: JSON blocks 
7. **Sponsorship programs**: JSON blocks 
8. **Low-confidence queue** 
9. **Horizon radar**: opens in 30–60 days 
10. **Bibliography**: URLs + access dates 

Target **20–35 verified calls** if the ecosystem is active; state honestly if thin.

### Automated supplement (Devpost)

```bash
npm run opportunities:devpost -- --dry-run
npm run opportunities:devpost
```

---

## Implementation checklist (~3–4 days)

| Step | Task | Time |
| ---- | ---- | ---- |
| 1 | Define scope; list sources per sub-prompt | 0.5 day |
| 2 | Formulate search queries per role | 0.5 day |
| 3 | Run 06a–06d (parallel) or full harvest | 1.5 days |
| 4 | Filter expired; verify canonical URLs | 1 day |
| 5 | Merge JSON, run `opportunities:merge`, images | 0.5 day |

---

Asia/Manila (UTC+8). Today: **[INSERT DATE]**.
