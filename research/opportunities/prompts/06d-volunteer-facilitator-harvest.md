# Prompt 6d — Volunteer & facilitator harvest

**Phase:** Supplemental (sub-prompt of [06-contributor-calls-harvest.md](./06-contributor-calls-harvest.md))  
**Run in:** Google Deep Research or Cursor agent browse  
**Save as:** section inside `responses/YYYY-MM-DD-contributor-calls-harvest.md` (header: `## Call for mentors & facilitators` — volunteers subsection)  
**Shared rules & JSON schema:** see Prompt 6

---

## Task

Compile **open calls for volunteers and facilitators** at student or tech events: emcee, workshop host, track chair, student ambassador, registration lead, AV helper, etc.

**Not in scope here:** hackathon judges/mentors (→ [06b](./06b-call-for-judges-mentors-harvest.md)); speakers (→ [06a](./06a-call-for-speakers-harvest.md)).

**Deadline rule:** sign-up closes within **45 days**, still open today.  
**Today (Asia/Manila, UTC+8):** **[INSERT DATE]**

### Include

- Contributor roles with a **specific application form** or signup page
- Student ambassador, campus rep, or facilitator programs with published intake
- Workshop hosts and track chairs at conferences with named apply links

### Exclude

- Generic “sign up to volunteer” with no form or role description
- Paid staffing agencies; full-time job postings
- Participant registration disguised as volunteering

---

## Required fields per entry

| Field | Requirement |
| ----- | ----------- |
| Role type | emcee, facilitator, ambassador, etc. |
| Event / org | Official name |
| Deadline | Sign-up close date |
| URL | Official signup page |
| Time commitment | If stated in the posting |

Use `contributor_role: volunteer` or `facilitator`.

---

## Where to search

```text
"call for volunteers" (tech OR hackathon OR conference) apply 2026 Philippines
"student ambassador" application (Google OR Microsoft OR AWS) 2026
site:lu.ma volunteer OR facilitator OR ambassador
"workshop host" OR "track chair" apply CFP OR volunteer 2026
```

| Source | Check |
| ------ | ----- |
| Luma / Meetup | Organizer volunteer posts |
| Vendor programs | MLSA, AWS Cloud Club, GitHub Campus ambassador intake |
| Conferences | DEVCON, campus tech weeks |
| Hackathons | Non-judge volunteer roles (logistics, emcee) |

---

## Output (this section only)

1. **Volunteer coverage** — count, sources, rejected  
2. **Closing this week** — deadline ≤ 7 days  
3. **JSON blocks** — sorted by deadline  
4. **Low-confidence** — generic sign-ups  
5. **Citations** — URL + access date  

Target **≥3 verified volunteer calls** when active.

---

Asia/Manila (UTC+8). Today: **[INSERT DATE]**.
