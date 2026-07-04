# Prompt 5: Game jam harvest

**Phase:** Supplemental · **Run in:** Google Deep Research (or Cursor agent browse) 
**Prerequisite:** none (optional: [02-harvest.md](./02-harvest.md) for full board context) 
**Save output as:** `responses/YYYY-MM-DD-game-jam-harvest.md` 
**Then:** verify URLs, merge with `npm run opportunities:merge` or add to `src/data/opportunity-board-items.js`, run `npm run opportunities:images`

Use this when you want a **game-jam-only pass**: itch.io jams, Ludum Dare, DEV game challenges, Devpost game jams, GMTK, and campus/indie jams PH students can join online.

---

## Research task

Find **open or upcoming game jams** worth listing on **stimmie.dev/opportunities** for **students and early-career game builders in the Philippines** (or **online/global** jams anyone can enter remotely).

**Time window:** submissions open now or opening within the next **45 days**. Include jams whose submission deadline is still in the future as of today. 
**Today (Asia/Manila, UTC+8):** **[INSERT DATE]**

### What counts as a game jam

- **Game jams** on itch.io (theme reveal + build window + itch submission)
- **Ludum Dare**, **GMTK Game Jam**, **Global Game Jam** (online/global tracks only unless PH site is confirmed)
- **DEV game challenges** ([dev.to/challenges](https://dev.to/challenges) with game / game jam / webgame tags)
- **Devpost** listings whose primary deliverable is a **playable game** (not generic hackathons unless game dev is the main track)
- **Campus / community jams** (GDG, game dev clubs, PIG Squad Summer Slow Jams, etc.) when PH students can join online
- **Game-adjacent build sprints** only when rules require shipping a **playable game** or game demo

### Hard excludes

- Generic hackathons with no game deliverable (route those to [03-devpost-harvest.md](./03-devpost-harvest.md))
- Jams already **closed** (submission deadline passed)
- **Invite-only** or paid-entry jams with no free tier (note paid jams in horizon radar only)
- **Region-locked** in-person jams unless clearly in the Philippines or hybrid with remote submission
- Aggregator reposts as canonical `url` (use itch.io jam page, Ludum Dare event page, or organizer site)
- `governmentph.com`, Scribd mirrors, random Facebook posts without an official jam page (mark **Low confidence** if FB-only)

### Core rule

**Boards discover; official pages publish.** For each jam, `url` must be the **official join/register/submit page** (itch.io `/jam/...`, Devpost jam, Ludum Dare compo page, DEV challenge page, organizer landing with rules + dates).

If the jam only has a Discord + Google Form, the **form or itch jam page** is OK. If only a Facebook event exists, try to find itch.io / Devpost / org site first.

---

## Mandatory source coverage

Search each source; report item count per source in the **coverage report**:

| Source | Where to look | Notes |
| ------ | ------------- | ----- |
| **itch.io** | [itch.io/jams](https://itch.io/jams), featured/active jams | Prefer jams with clear submission end dates |
| **Ludum Dare** | [ldjam.com](https://ldjam.com), [ludumdare.com](https://ludumdare.com) | LD compo + jam modes; global online |
| **DEV** | [dev.to/challenges](https://dev.to/challenges) | Filter: game, webgame, game jam, solstice, glitch |
| **Devpost** | Search `game jam`, `gamedev`, `GLITCHED`, `games with a hook` | Game-first listings only |
| **GMTK** | Mark Brown / GMTK community announcements | Usually annual; verify dates |
| **Global Game Jam** | [globalgamejam.org](https://globalgamejam.org) | January main jam; note PH sites if any |
| **Indie communities** | PIG Squad Summer Slow Jams, IndieCade Climate Jam, game dev Discords with public jam pages | Remote-friendly only unless PH |
| **Reddit signal** | `r/gamedev`, `r/itchio`, `r/PinoyProgrammer` | Corroborate → official jam URL |
| **YouTube / dev logs** | GMTK, Brackeys alumni channels, coding.kitty | Only if linking to live jam page |

**Search queries (copy-paste):**

```text
site:itch.io/jam ("submissions open" OR "game jam") 2026
site:dev.to/challenges game jam 2026
site:devpost.com "game jam" open online
"Ludum Dare" 2026 dates registration
"Global Game Jam" Philippines 2026
"game jam" ("Philippines" OR Manila OR online) submissions due
```

---

## Per-opportunity output (one JSON block each)

Use **`type`: `"game-jam"`** (not `hackathon`) for all game jam entries.

```json
{
  "title": "",
  "type": "game-jam",
  "url": "OFFICIAL JAM / REGISTER / SUBMIT PAGE ONLY",
  "image_url": "optional bot-friendly page for cover image (itch.io jam URL usually works)",
  "org": "",
  "location": "Online / Hybrid / City, Philippines",
  "dates": [
    { "label": "Jam starts", "date": "YYYY-MM-DD" },
    { "label": "Submission deadline", "date": "YYYY-MM-DD" }
  ],
  "blurb": "1 sentence. Who can join, theme or format, what you ship. No em dashes.",
  "beginner_friendly": true,
  "source_platform": "itch.io|Devpost|DEV|Ludum Dare|organizer",
  "source_url": "where you found it",
  "confidence": "High|Medium|Low",
  "team_size": "optional e.g. solo-4, pairs only",
  "engine_notes": "optional e.g. any engine, Godot-only, browser games preferred"
}
```

### Rules

- **Blurbs:** one sentence. Dates carry the detail. No em dashes.
- **`beginner_friendly`:** `true` if rules say all skill levels welcome, no prior game dev required, or jam FAQ encourages first-timers; omit if unclear.
- **Dedupe** the same jam across itch + Discord announcement + Reddit (one entry, best canonical URL).
- **DEV challenges:** use `https://dev.to/challenges/{slug}` as `url`; confirm submission deadline on the challenge page, not the announcement post alone.
- **itch.io:** submission window is authoritative; if theme is hidden until start, note that in blurb.
- **In-person PH jams** (e.g. campus jam at PUP): include if students can register; set `location` to city + venue; hybrid OK.
- Sort by **nearest submission deadline**; flag jams closing in **7 days** at the top.

### Role opportunities (optional section)

Some jam pages recruit **judges, mentors, speakers, or playtesters**. If the jam description has a separate apply link, add a second JSON block:

```json
{
  "title": "Call for Judges — Example Game Jam",
  "type": "event",
  "url": "https://forms.gle/... or mailto:...",
  "image_url": "https://itch.io/jam/example",
  "org": "",
  "location": "Online",
  "dates": [
    { "label": "Submission deadline", "date": "YYYY-MM-DD" }
  ],
  "blurb": "Game jam recruiting judges. 1 sentence. No em dashes.",
  "source_platform": "itch.io",
  "source_url": "jam page URL#rules or description anchor",
  "confidence": "Medium"
}
```

---

## Also include

1. **Coverage report**: jams found per source; how many rejected (closed, regional, not a game jam) and why 
2. **Closing this week**: jams with submission deadline within 7 days 
3. **Horizon radar**: jams opening in 30–60 days (GMTK, GGJ, LD) even without firm submission dates yet 
4. **Low-confidence queue**: FB-only, Discord-only, or vague flyers needing manual verify 
5. **Bibliography**: URLs with access dates 

Target **15–25 verified game jams** per run if the ecosystem is active; be honest if the window is thin ("only 6 open global jams found") rather than padding with expired listings.

Asia/Manila (UTC+8). Today: **[INSERT DATE]**.
