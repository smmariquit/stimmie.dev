# Prompt 6b: Call for judges & mentors harvest

**Phase:** Supplemental (sub-prompt of [06-contributor-calls-harvest.md](./06-contributor-calls-harvest.md)) 
**Run in:** Google Deep Research or Cursor agent browse 
**Save as:** sections inside `responses/YYYY-MM-DD-contributor-calls-harvest.md` (`## Call for judges`, `## Call for mentors`) 
**Shared rules & JSON schema:** see Prompt 6

---

## Task

Identify **open calls for judges and mentors** at hackathons, pitch competitions, accelerators, and builder programs aimed at student developers.

**Deadline rule:** applications close within **45 days**, still in the future today. 
**Today (Asia/Manila, UTC+8):** **[INSERT DATE]**

### Judges: include

- Hackathon judge signup (Devpost description, Google Form, organizer page)
- Pitch competition jury, demo day panel, case competition scoring roles
- Roles where **students or alumni judges** are explicitly welcome (`beginner_friendly: true`)

### Mentors: include

- “Coach teams”, mentor applications, office hours volunteers
- Accelerator / Startup Weekend mentor intake with a clear apply path
- 1:1 or office-hour programs (ADPList community programs with apply forms, not generic profiles)

### Exclude

- Senior-only judging (10+ years required) unless students welcome
- Generic event staffing / registration desk volunteers (→ use [06d](./06d-volunteer-facilitator-harvest.md))
- Expired forms; pay-to-mentor schemes

---

## Required fields per entry

| Field | Requirement |
| ----- | ----------- |
| Event / program | Official name |
| Role | judge or mentor |
| Deadline | Application close date |
| URL | Form or official volunteer page |
| Commitment | e.g. “score 10 projects”, “2h office hours” if stated |

Title format: `Call for Judges: {Event}` or `Call for Mentors: {Event}`.

---

## Where to search

```text
"call for judges" (hackathon OR "pitch competition") apply form 2026
"call for mentors" (hackathon OR accelerator OR startup) application 2026
site:devpost.com ("call for judges" OR "seeking mentors" OR "mentor application")
site:devfolio.com judge OR mentor apply
"startup weekend" mentor application Philippines
```

| Source | Check |
| ------ | ----- |
| Devpost | Full **challenge description** (forms often mid-page) |
| Devfolio, MLH | Event volunteer pages |
| Campus hackathons | `.edu.ph`, GDG on Campus, org Facebook → find canonical form |
| Accelerators | Techstars, Founder Institute mentor intake |

### Automated supplement

```bash
npm run opportunities:devpost -- --dry-run
npm run opportunities:devpost
```

---

## Output (these sections only)

1. **Judges & mentors coverage**: counts per role; sources; rejected 
2. **Closing this week**: deadline ≤ 7 days 
3. **Call for judges**: JSON blocks 
4. **Call for mentors**: JSON blocks 
5. **Low-confidence queue** 
6. **Citations**: URL + access date 

Target **≥5 verified judge/mentor calls** combined when active.

---

Asia/Manila (UTC+8). Today: **[INSERT DATE]**.
