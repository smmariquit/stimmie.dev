# Prompt 6c: Sponsorship calls & programs harvest

**Phase:** Supplemental (sub-prompt of [06-contributor-calls-harvest.md](./06-contributor-calls-harvest.md)) 
**Run in:** Google Deep Research or Cursor agent browse 
**Save as:** sections inside `responses/YYYY-MM-DD-contributor-calls-harvest.md` (`## Call for sponsors`, `## Sponsorship programs`) 
**Shared rules & JSON schema:** see Prompt 6

---

## Task

Find **sponsorship opportunities** for campus orgs and student-led events: (A) events **recruiting sponsors**, and (B) **programs** that fund communities.

**Audience:** campus org officers, hackathon organizers, and student leaders in the Philippines (+ global programs PH orgs can apply to).

**Deadline rule:** inquiries or applications close within **45 days**, still open today. 
**Today (Asia/Manila, UTC+8):** **[INSERT DATE]**

### A: Call for sponsors (event-level)

**Include:** “Become a sponsor”, sponsorship prospectus, partner inquiry form on an official event site.

**Exclude:** generic “contact us” with no deck or form; news articles about past sponsors only.

Use `type: event`, `contributor_role: sponsor`.

### B: Sponsorship programs (org-level funding)

**Include:** corporate/community programs where **student orgs apply for budget** (GDG community support, GitHub Campus, AWS Cloud Club funding, CSR innovation grants, DICT/CHED programs).

**Exclude:** single-event sponsor decks (those are A); programs with no public apply path.

Use `type: program`, `org_level: true`.

---

## Required fields per entry

| Field | A (event) | B (program) |
| ----- | --------- | ----------- |
| Name | Event + “Call for Sponsors” | Program name |
| Organization | Event organizer | Funding org |
| Deadline | Inquiry close | Application close |
| URL | Prospectus / partner form | Official apply page |
| Notes | Tiers, in-kind, student-friendly | Who qualifies, what funding covers |

---

## Where to search

```text
"become a sponsor" OR "sponsorship opportunities" (conference OR hackathon OR meetup) Philippines
"sponsorship prospectus" hackathon 2026
"community sponsorship" (Google OR GitHub OR AWS) student OR campus apply
site:lu.ma sponsor OR "partner with us"
"innovation grant" (CHED OR DICT OR campus) student organization apply
```

| Source | Check |
| ------ | ----- |
| Conference / hackathon sites | Sponsor pages with forms or PDF decks |
| Google Developer Groups | Community support programs |
| GitHub Education | Campus Experts / campus program |
| Corporate CSR | Globe, PLDT, telco innovation pages |
| Government | DICT, CHED, UniFAST innovation calls |

---

## Output (these sections only)

1. **Sponsorship coverage**: event-level vs program counts; rejected 
2. **Closing this week**: deadline ≤ 7 days 
3. **Call for sponsors**: JSON blocks (event-level) 
4. **Sponsorship programs**: JSON blocks (`org_level: true`) 
5. **Low-confidence**: vague contact pages 
6. **Citations**: URL + access date 

Target **≥3 verified entries** per subsection when active.

---

Asia/Manila (UTC+8). Today: **[INSERT DATE]**.
