# Proton Mail inbox harvest

- **Date researched:** 2026-06-24
- **Tool:** `scripts/scan-proton-export.mjs` + manual URL verification
- **Source:** Proton export `parazeez@protonmail.com/mail_20260624_035224`
- **Status:** ready
- **Entries:** 2 verified

---

## Inbox coverage report

- **Emails scanned:** 9,954 (full export)
- **Recent window (60 days):** 266 emails
- **Opportunity signals found:** 8 subject hits, ~120 keyword hits (many false positives from Firefox Pocket, Grab receipts, Coursera promos)
- **Verified new listings:** 2
- **Rejected:** 18+ (expired DEV challenges, marketing with no canonical apply page, tracker-only links, non-opportunity mail)

### Sender leaderboard (real opportunities)

| Sender | Hits | Notes |
|--------|------|-------|
| `team@mail.stellarph.io` | 1 program | PH100 application nags |
| `yo@dev.to` | 1 event | MLH 100 Days of Solana link in digest |
| Others | 0 | Grab (57 receipts/promos), Harvard Online (cohort beta, no clean URL), ADPList (tracker links only) |

### Low-confidence queue

- **Founder Institute Vibe Coding Pro Bootcamp** (`fi.co/bootcamp/vibe-coding-pro`) — paid commercial bootcamp; skipped unless you want paid programs on the board.
- **DEV challenges** (Finish-Up-A-Thon, Hermes Agent, June Solstice Game Jam, Gemma 4, Google Cloud NEXT Writing) — all closed; winners announced or submissions past due.
- **LinkedIn newsletter** `lu.ma/hf0reyr6` — from April 2025; stale.

### Horizon radar

- No "applications open soon" items with firm future dates beyond PH100 (closes Jun 30).

---

## Closing within seven days

### CLOSES IN 7 DAYS — StellarPH PH100 2026

```json
{
  "title": "StellarPH PH100 2026",
  "type": "program",
  "url": "https://stellarph.io/programs/ph100/2026/apply",
  "image_url": "https://stellarph.io/programs/ph100",
  "org": "StellarPH",
  "location": "Philippines",
  "dates": [
    { "label": "Application deadline", "date": "2026-06-30" }
  ],
  "blurb": "Annual list spotlighting 100 brightest Filipino talents under 30 in the startup ecosystem. Selected honorees get recognition, network access, and growth opportunities.",
  "source_platform": "Proton Mail",
  "source_url": "You're Not Done Yet — team@mail.stellarph.io — 2026-06-23",
  "confidence": "High",
  "email_received": "2026-06-23"
}
```

---

## Dated upcoming opportunities

### MLH 100 Days of Solana

```json
{
  "title": "MLH 100 Days of Solana",
  "type": "program",
  "url": "https://events.mlh.io/events/13995-100-days-of-solana",
  "image_url": "https://events.mlh.io/events/13995-100-days-of-solana",
  "org": "Major League Hacking",
  "location": "Online",
  "dates": [
    { "label": "Series starts", "date": "2026-04-20" },
    { "label": "Series ends", "date": "2026-07-26" }
  ],
  "blurb": "Free online challenge series with daily hands-on Solana tasks for builders moving from web3 curiosity to shipped projects.",
  "beginner_friendly": true,
  "source_platform": "Proton Mail",
  "source_url": "DEV Digest — yo@dev.to — 2026-05-27",
  "confidence": "High",
  "email_received": "2026-05-27"
}
```
