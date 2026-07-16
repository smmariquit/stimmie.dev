# Prompt 4: Email inbox harvest (Gemini in Gmail)

**Phase:** Supplemental · **Run in:** Gmail Gemini (side panel or `@Gemini` in compose) 
**Save output as:** `responses/YYYY-MM-DD-email-harvest.md` 
**Then:** verify URLs manually, merge with `npm run opportunities:merge` or add by hand

Paste everything below the line into Gemini while Gmail context is enabled.

---

## Research task

You have access to my Gmail. Scan my inbox (and Promotions / Updates if relevant) for **actionable opportunities** I could list on **stimmie.dev/opportunities**: hackathons, internships, scholarships, events, certificates, and programs for **students and early-career builders in the Philippines** (or **online/global** opportunities PH applicants can join).

**Time window:** last **45 days**, plus any email with a deadline still in the future as of today. 
**Today (Asia/Manila, UTC+8):** **[INSERT DATE]**

### What counts as an opportunity

- Hackathons, buildathons, ideathons (Devpost, Devfolio, Luma, campus org, corporate)
- Internships, OJT, fellowships (employer or program page, not job-board search URLs)
- Scholarships, grants, competitions with prizes
- Events: meetups, workshops, watch parties, conferences (Luma, Meetup, Eventbrite, org newsletters)
- Free certs, MOOCs, training cohorts (Google, AWS, TESDA, university, vendor)

### Senders and signals to prioritize

Look for mail from or mentioning: Luma, Devpost, Devfolio, MLH, lablab.ai, Meetup, Eventbrite, DEVCON, Friends of Figma, UXPH, Globe, Grab, CHED, DOST, TESDA, DICT, university career offices, `.edu.ph`, `.gov.ph`, Outreachy, GSoC, ASEAN Foundation, UNESCO, YC / startup newsletters, and campus CS/org lists I am subscribed to.

Also search subject/body keywords: `hackathon`, `internship`, `OJT`, `scholarship`, `grant`, `fellowship`, `RSVP`, `register`, `application deadline`, `call for`, `cohort`, `workshop`, `deadline`, `Philippines`, `Manila`, `remote`, `online`.

### Hard excludes (do not list)

- Pure marketing with no apply link or event date
- LinkedIn / Indeed / JobStreet **search result** URLs (only list if you can find the **employer ATS** or careers page)
- `governmentph.com`, Scribd PDFs, random repost blogs as canonical links
- Already-expired deadlines (unless the email announces something opening in the next 60 days)
- Recruiting spam, "earn money fast", crypto airdrops, unrelated job ads for seniors
- Newsletters that only link to paywalled aggregators

### Core rule

**Email discovers; official pages publish.** For each hit, find the **canonical apply/register/info URL** (Luma event page, Devpost, company careers, `.gov.ph`, etc.). If the email only has a Google Form, that form URL is OK. If the email only has a Facebook post, note it as **Low confidence** and try to find an official landing page.

### Per-opportunity output (one JSON block each)

```json
{
  "title": "",
  "type": "hackathon|internship|event|certificate|program",
  "url": "OFFICIAL APPLY/REGISTER PAGE ONLY",
  "image_url": "optional bot-friendly page for cover image",
  "org": "",
  "location": "City / Online / Hybrid",
  "dates": [
    { "label": "Registration closes", "date": "YYYY-MM-DD" }
  ],
  "blurb": "1 sentence. No em dashes.",
  "beginner_friendly": true,
  "source_platform": "Gmail",
  "source_url": "email subject line + sender + approximate date received",
  "confidence": "High|Medium|Low",
  "email_received": "YYYY-MM-DD",
  "skip_reason": "only if rejected; omit field when included"
}
```

### Rules

- **Blurbs:** one sentence. Dates carry the detail. No em dashes.
- **`beginner_friendly`:** true if no prior experience expected; omit if unclear.
- **Dedupe** obvious repeats (same Luma event mailed twice).
- Flag items closing in **7 days** at the top.
- Do **not** paste full email bodies or personal data. Subject + sender + date is enough for `source_url`.

### Also include

1. **Inbox coverage report**: how many emails scanned, how many opportunities found, how many rejected and why 
2. **Sender leaderboard**: which senders produced the most real opportunities 
3. **Low-confidence queue**: items that need me to verify manually (FB-only links, vague flyers) 
4. **Horizon radar**: "save the date" or "applications open soon" from email, even without a firm deadline 

Sort by nearest actionable date. Be honest about gaps ("no internship emails this window") rather than inventing listings.
