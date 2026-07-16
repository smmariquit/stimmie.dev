# Prompt 3: Devpost harvest (hackathons + role calls)

**Phase:** Supplemental · **Run in:** Cursor / local script (API) or Google Deep Research (manual browse) 
**Prerequisite:** none (optional: [02-harvest.md](./02-harvest.md) for full board context) 
**Save output as:** `responses/YYYY-MM-DD-devpost-harvest.md`

Use this when you want a **Devpost-only pass** without re-running the full multi-bucket harvest.

## Automated path (preferred)

```bash
npm run opportunities:devpost
npm run opportunities:devpost -- --dry-run      # preview without writing board
npm run opportunities:devpost -- --skip-roles   # hackathons only, skip description scan
npm run opportunities:images
```

The script pulls **open, online, non-invite-only** hackathons from `https://devpost.com/api/hackathons`, skips obvious region-locked titles (India IIT, Africa-only tracks, etc.), dedupes against the live board, and merges into `src/data/opportunity-board-items.js`.

It also fetches each hackathon's **`#challenge-description`** and looks for calls to recruit **judges, mentors, speakers, panelists, volunteers, coaches, or facilitators**. Matches become separate **`event`** listings (apply form, `mailto:`, or description anchor) plus a **Role opportunities** section in the harvest markdown.

Signals include phrases like `call for judges`, `seeking mentors`, `volunteer applications open`, or a Google Form near role keywords in the description.

For a broader manual pass (speakers, CFPs, sponsor decks, org sponsorship programs), use [05-game-jam-harvest.md](./05-game-jam-harvest.md) for jams only or [06-contributor-calls-harvest.md](./06-contributor-calls-harvest.md) (sub-prompts [06a](./06a-call-for-speakers-harvest.md)–[06d](./06d-volunteer-facilitator-harvest.md)) for contributor and sponsor calls.

## Manual / Deep Research path

If the API is down or you need PH-adjacent in-person Devpost listings:

1. Browse [devpost.com/hackathons](https://devpost.com/hackathons) with filters: **Open**, **Upcoming**, **Online** (add **In person** only when Philippines-accessible).
2. For each hackathon, use the **Devpost project page** (`*.devpost.com`) as `url` (canonical).
3. Set `image_url` to the same Devpost URL (OG thumbnail fetch works).
4. Output one JSON block per hackathon (same schema as [02-harvest.md](./02-harvest.md)).
5. Set `source_platform`: `"Devpost"` and `source_url` to the listing URL.

### Inclusion rules

- **Include:** global online hackathons PH students can join remotely; sponsor hackathons (AWS, GitLab, Reddit, MLH on Devpost, etc.).
- **Exclude:** invite-only; submission period already closed; titles clearly limited to one country/region (e.g. "India High School", "IIT India", "Africa Deep Tech" unless explicitly open globally).
- **Dedupe:** if the same event exists on Devfolio or lablab.ai, keep **one** entry with the Devpost URL when available.

### Per-opportunity JSON (hackathon)

```json
{
  "title": "",
  "type": "hackathon",
  "url": "https://example.devpost.com/",
  "image_url": "https://example.devpost.com/",
  "org": "",
  "location": "Online",
  "dates": [
    { "label": "Submission deadline", "date": "YYYY-MM-DD" }
  ],
  "blurb": "1 sentence. No em dashes.",
  "beginner_friendly": true,
  "source_platform": "Devpost",
  "source_url": "https://devpost.com/hackathons",
  "confidence": "High"
}
```

### Per-opportunity JSON (role call from description)

Use `type`: `"event"` when the Devpost description recruits judges, mentors, speakers, or volunteers.

```json
{
  "title": "Example Hackathon: Call for Judges",
  "type": "event",
  "url": "https://forms.gle/example or mailto:organizer@example.com",
  "image_url": "https://example.devpost.com/",
  "org": "",
  "location": "Online",
  "dates": [
    { "label": "Hackathon submission deadline", "date": "YYYY-MM-DD" }
  ],
  "blurb": "Devpost hackathon recruiting judges. 1 sentence. No em dashes.",
  "source_platform": "Devpost",
  "source_url": "https://example.devpost.com/#challenge-description",
  "confidence": "Medium"
}
```

### Also include

1. **Coverage report**: count fetched, merged, role calls found, skipped (duplicate / blocklist / closed)
2. **Closing soon**: hackathons with submission deadline within 7 days
3. **Role opportunities**: judges / mentors / speakers / volunteers with apply links
4. **Bibliography**: Devpost API or browse URL with access date

Asia/Manila (UTC+8). Today: **[INSERT DATE]**.
