# Prompt 3 — Devpost harvest (hackathons only)

**Phase:** Supplemental · **Run in:** Cursor / local script (API) or Google Deep Research (manual browse)  
**Prerequisite:** none (optional: [02-harvest.md](./02-harvest.md) for full board context)  
**Save output as:** `responses/YYYY-MM-DD-devpost-harvest.md`

Use this when you want a **Devpost-only pass** without re-running the full multi-bucket harvest.

## Automated path (preferred)

```bash
npm run opportunities:devpost
npm run opportunities:devpost -- --harvest   # also writes responses/YYYY-MM-DD-devpost-harvest.md
npm run opportunities:images
```

The script pulls **open, online, non-invite-only** hackathons from `https://devpost.com/api/hackathons`, skips obvious region-locked titles (India IIT, Africa-only tracks, etc.), dedupes against the live board, and merges into `src/data/opportunity-board-items.js`.

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

### Per-opportunity JSON

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

### Also include

1. **Coverage report** — count fetched, merged, skipped (duplicate / blocklist / closed)
2. **Closing soon** — hackathons with submission deadline within 7 days
3. **Bibliography** — Devpost API or browse URL with access date

Asia/Manila (UTC+8). Today: **[INSERT DATE]**.
