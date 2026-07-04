# Prompt 6a: Call for speakers harvest

**Phase:** Supplemental (sub-prompt of [06-contributor-calls-harvest.md](./06-contributor-calls-harvest.md)) 
**Run in:** Google Deep Research or Cursor agent browse 
**Save as:** section inside `responses/YYYY-MM-DD-contributor-calls-harvest.md` (header: `## Call for speakers`) 
**Shared rules & JSON schema:** see Prompt 6

---

## Task

List all **currently open calls for speakers** (talks, panels, workshops, lightning talks) at tech and academic events relevant to **students and early-career builders** in the Philippines, plus **online/global** CFPs they can submit to remotely.

**Deadline rule:** CFP closes within the **next 45 days** and is still in the future as of today. 
**Today (Asia/Manila, UTC+8):** **[INSERT DATE]**

### Include

- Sessionize, Papercall, Pretalx, and conference `/cfp` pages
- Campus tech summits, meetup chapters, DEVCON PH, PyCon PH, UXPH, Friends of Figma
- “Submit a talk”, “call for presenters”, lightning-talk tracks

### Exclude

- Expired CFPs
- Pay-to-present / paid speaker slots with no student tier
- Hackathon **participant** signup mislabeled as speaking
- LinkedIn-only posts (find canonical CFP page or skip)

---

## Required fields per entry

| Field | Requirement |
| ----- | ----------- |
| Event name | Official conference or meetup name |
| Role | talk / panel / workshop / lightning |
| Deadline | Applications close (YYYY-MM-DD, Manila) |
| URL | Official CFP page only |
| Audience fit | Note if students or first-time speakers welcome |

Output as JSON blocks per Prompt 6 schema (`contributor_role: speaker`).

---

## Where to search

```text
"call for speakers" ("Philippines" OR Manila OR online OR remote) 2026 apply
site:sessionize.com Philippines OR online 2026
site:papercall.io 2026
site:pretalx.com CFP 2026
"call for proposals" (PyCon OR DevCon OR UXPH OR "Google Developer") Philippines
site:lu.ma ("CFP" OR "call for speakers")
```

| Source | Check |
| ------ | ----- |
| [Sessionize](https://sessionize.com) | PH + online events |
| [Papercall](https://www.papercall.io) | Open CFPs |
| Campus / chapter sites | GDG, MLSA, PSITE, `.edu.ph` orgs |
| [Luma](https://lu.ma) | Organizer CFP posts |

---

## Output (this section only)

1. **Speakers coverage**: count found, sources checked, rejected + why 
2. **Closing this week**: speaker CFPs with deadline ≤ 7 days 
3. **JSON blocks**: sorted by nearest deadline 
4. **Low-confidence**: vague posts, missing apply links 
5. **Horizon**: CFPs opening in 30–60 days 
6. **Citations**: URL + access date for each entry 

Target **≥5 verified speaker calls** when the ecosystem is active.

---

Asia/Manila (UTC+8). Today: **[INSERT DATE]**.
