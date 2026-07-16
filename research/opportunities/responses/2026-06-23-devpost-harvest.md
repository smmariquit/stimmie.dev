# Devpost harvest: 2026-06-23

- **Date researched:** 2026-06-23
- **Tool:** `npm run opportunities:devpost` (Devpost API + description scan)
- **Prompt:** [../prompts/03-devpost-harvest.md](../prompts/03-devpost-harvest.md)
- **Status:** published
- **Issue slug:** devpost-2026-06-23

---

## Coverage report

| Metric | Count |
| ------ | -----: |
| API candidates (open, online) | 35 |
| Role opportunities (judges, mentors, speakers, volunteers) | 7 |
| New hackathons merged this run | 0 |
| New role listings merged this run | 0 |
| Board total after merge | 144 |
| Prior board count | 144 |

Filter: open + online + not invite-only; blocklist for region-locked titles; skip closed submission deadlines. Role scan parses each Devpost `#challenge-description` for judge, mentor, speaker, and volunteer calls.

---

## Hackathons

### Build with Gemini XPRIZE

```json
{
  "title": "Build with Gemini XPRIZE",
  "type": "hackathon",
  "url": "https://xprize.devpost.com/",
  "org": "XPRIZE",
  "location": "Online",
  "blurb": "Machine Learning/AI, Education, Productivity hackathon on Devpost. Prizes: $2,000,000. about 2 months left.",
  "source_platform": "Devpost",
  "source_url": "https://xprize.devpost.com/",
  "confidence": "High",
  "image_url": "https://xprize.devpost.com/",
  "dates": [
    {
      "label": "Hackathon window",
      "date": "2026-05-19",
      "endDate": "2026-08-17"
    },
    {
      "label": "Submission deadline",
      "date": "2026-08-17"
    }
  ]
}
```

### H0: Hack the Zero Stack with Vercel v0 and AWS Databases

```json
{
  "title": "H0: Hack the Zero Stack with Vercel v0 and AWS Databases",
  "type": "hackathon",
  "url": "https://h01.devpost.com/",
  "org": "Amazon",
  "location": "Online",
  "blurb": "Databases, Open Ended, Web hackathon on Devpost. Prizes: $80,000. 6 days left.",
  "source_platform": "Devpost",
  "source_url": "https://h01.devpost.com/",
  "confidence": "High",
  "image_url": "https://h01.devpost.com/",
  "dates": [
    {
      "label": "Hackathon window",
      "date": "2026-05-27",
      "endDate": "2026-06-29"
    },
    {
      "label": "Submission deadline",
      "date": "2026-06-29"
    }
  ]
}
```

### Global AI Hackathon Series with Qwen Cloud

```json
{
  "title": "Global AI Hackathon Series with Qwen Cloud",
  "type": "hackathon",
  "url": "https://qwencloud-hackathon.devpost.com/",
  "org": "Alibaba Cloud",
  "location": "Online",
  "blurb": "Machine Learning/AI, Design, Productivity hackathon on Devpost. Prizes: $45,000. 16 days left.",
  "source_platform": "Devpost",
  "source_url": "https://qwencloud-hackathon.devpost.com/",
  "confidence": "High",
  "image_url": "https://qwencloud-hackathon.devpost.com/",
  "dates": [
    {
      "label": "Hackathon window",
      "date": "2026-05-26",
      "endDate": "2026-07-09"
    },
    {
      "label": "Submission deadline",
      "date": "2026-07-09"
    }
  ]
}
```

### UiPath AgentHack

```json
{
  "title": "UiPath AgentHack",
  "type": "hackathon",
  "url": "https://uipath-agenthack.devpost.com/",
  "org": "UiPath",
  "location": "Online",
  "blurb": "Enterprise, Machine Learning/AI, Robotic Process Automation hackathon on Devpost. Prizes: $50,000. 6 days left.",
  "source_platform": "Devpost",
  "source_url": "https://uipath-agenthack.devpost.com/",
  "confidence": "High",
  "image_url": "https://uipath-agenthack.devpost.com/",
  "dates": [
    {
      "label": "Hackathon window",
      "date": "2026-05-15",
      "endDate": "2026-06-29"
    },
    {
      "label": "Submission deadline",
      "date": "2026-06-29"
    }
  ]
}
```

### Slack Agent Builder Challenge

```json
{
  "title": "Slack Agent Builder Challenge",
  "type": "hackathon",
  "url": "https://slackhack.devpost.com/",
  "org": "Salesforce",
  "location": "Online",
  "blurb": "Beginner Friendly, Enterprise, Low/No Code hackathon on Devpost. Prizes: $42,000. 20 days left.",
  "source_platform": "Devpost",
  "source_url": "https://slackhack.devpost.com/",
  "confidence": "High",
  "image_url": "https://slackhack.devpost.com/",
  "dates": [
    {
      "label": "Hackathon window",
      "date": "2026-05-20",
      "endDate": "2026-07-13"
    },
    {
      "label": "Submission deadline",
      "date": "2026-07-13"
    }
  ],
  "beginner_friendly": true
}
```

### GitLab Transcend Hackathon

```json
{
  "title": "GitLab Transcend Hackathon",
  "type": "hackathon",
  "url": "https://gitlab-transcend.devpost.com/",
  "org": "GitLab",
  "location": "Online",
  "blurb": "Machine Learning/AI, DevOps, Productivity hackathon on Devpost. Prizes: $20,000. about 22 hours left.",
  "source_platform": "Devpost",
  "source_url": "https://gitlab-transcend.devpost.com/",
  "confidence": "High",
  "image_url": "https://gitlab-transcend.devpost.com/",
  "dates": [
    {
      "label": "Hackathon window",
      "date": "2026-06-10",
      "endDate": "2026-06-24"
    },
    {
      "label": "Submission deadline",
      "date": "2026-06-24"
    }
  ]
}
```

### Reddit’s Games with a Hook Hackathon

```json
{
  "title": "Reddit’s Games with a Hook Hackathon",
  "type": "hackathon",
  "url": "https://redditgameswithahook.devpost.com/",
  "org": "reddit",
  "location": "Online",
  "blurb": "Beginner Friendly, Gaming, Web hackathon on Devpost. Prizes: $40,000. 22 days left.",
  "source_platform": "Devpost",
  "source_url": "https://redditgameswithahook.devpost.com/",
  "confidence": "High",
  "image_url": "https://redditgameswithahook.devpost.com/",
  "dates": [
    {
      "label": "Hackathon window",
      "date": "2026-06-17",
      "endDate": "2026-07-15"
    },
    {
      "label": "Submission deadline",
      "date": "2026-07-15"
    }
  ],
  "beginner_friendly": true
}
```

### Arm Create: AI Optimization Challenge

```json
{
  "title": "Arm Create: AI Optimization Challenge",
  "type": "hackathon",
  "url": "https://arm-ai-optimization-challenge.devpost.com/",
  "org": "arm",
  "location": "Online",
  "blurb": "Machine Learning/AI hackathon on Devpost. Prizes: $8,000. about 2 months left.",
  "source_platform": "Devpost",
  "source_url": "https://arm-ai-optimization-challenge.devpost.com/",
  "confidence": "High",
  "image_url": "https://arm-ai-optimization-challenge.devpost.com/",
  "dates": [
    {
      "label": "Hackathon window",
      "date": "2026-06-04",
      "endDate": "2026-08-14"
    },
    {
      "label": "Submission deadline",
      "date": "2026-08-14"
    }
  ]
}
```

### Youth Code x AI

```json
{
  "title": "Youth Code x AI",
  "type": "hackathon",
  "url": "https://youth-code-x-ai-29376.devpost.com/",
  "org": "Youth Code Foundation",
  "location": "Online",
  "blurb": "Beginner Friendly, Mobile, Web hackathon on Devpost. Prizes: $2,700. 4 days left.",
  "source_platform": "Devpost",
  "source_url": "https://youth-code-x-ai-29376.devpost.com/",
  "confidence": "High",
  "image_url": "https://youth-code-x-ai-29376.devpost.com/",
  "dates": [
    {
      "label": "Hackathon window",
      "date": "2026-06-20",
      "endDate": "2026-06-27"
    },
    {
      "label": "Submission deadline",
      "date": "2026-06-27"
    }
  ],
  "beginner_friendly": true
}
```

### Backblaze Generative Media Hackathon: Build with Genblaze on B2

```json
{
  "title": "Backblaze Generative Media Hackathon: Build with Genblaze on B2",
  "type": "hackathon",
  "url": "https://backblaze-generative-media.devpost.com/",
  "org": "Backblaze",
  "location": "Online",
  "blurb": "Machine Learning/AI, Music/Art, Voice skills hackathon on Devpost. Prizes: $10,000. about 1 month left.",
  "source_platform": "Devpost",
  "source_url": "https://backblaze-generative-media.devpost.com/",
  "confidence": "High",
  "image_url": "https://backblaze-generative-media.devpost.com/",
  "dates": [
    {
      "label": "Hackathon window",
      "date": "2026-06-22",
      "endDate": "2026-08-03"
    },
    {
      "label": "Submission deadline",
      "date": "2026-08-03"
    }
  ]
}
```

### LUMA Hackathon (July 3rd - 10th)

```json
{
  "title": "LUMA Hackathon (July 3rd - 10th)",
  "type": "hackathon",
  "url": "https://luma-hackathon-500.devpost.com/",
  "org": "LUMA",
  "location": "Online",
  "blurb": "Beginner Friendly, Machine Learning/AI, Open Ended hackathon on Devpost. Prizes: $0. 17 days left.",
  "source_platform": "Devpost",
  "source_url": "https://luma-hackathon-500.devpost.com/",
  "confidence": "High",
  "image_url": "https://luma-hackathon-500.devpost.com/",
  "dates": [
    {
      "label": "Hackathon window",
      "date": "2026-04-11",
      "endDate": "2026-07-10"
    },
    {
      "label": "Submission deadline",
      "date": "2026-07-10"
    }
  ],
  "beginner_friendly": true
}
```

### Creator Colosseum Startup Competition: Student Founders. Real Startups.

```json
{
  "title": "Creator Colosseum Startup Competition: Student Founders. Real Startups.",
  "type": "hackathon",
  "url": "https://creatorcolosseum.devpost.com/",
  "org": "Creator Colosseum",
  "location": "Online",
  "blurb": "Beginner Friendly, Low/No Code, Social Good hackathon on Devpost. Prizes: $575. 7 days left.",
  "source_platform": "Devpost",
  "source_url": "https://creatorcolosseum.devpost.com/",
  "confidence": "High",
  "image_url": "https://creatorcolosseum.devpost.com/",
  "dates": [
    {
      "label": "Hackathon window",
      "date": "2026-04-24",
      "endDate": "2026-06-30"
    },
    {
      "label": "Submission deadline",
      "date": "2026-06-30"
    }
  ],
  "beginner_friendly": true
}
```

### PhysTech 2026: Physical Activity and Technology Hack Day

```json
{
  "title": "PhysTech 2026: Physical Activity and Technology Hack Day",
  "type": "hackathon",
  "url": "https://phystech-2026.devpost.com/",
  "org": "Binnovative",
  "location": "Online",
  "blurb": "Education, Health, IoT hackathon on Devpost. Prizes: $0. 4 days left.",
  "source_platform": "Devpost",
  "source_url": "https://phystech-2026.devpost.com/",
  "confidence": "High",
  "image_url": "https://phystech-2026.devpost.com/",
  "dates": [
    {
      "label": "Hackathon window",
      "date": "2026-02-08",
      "endDate": "2026-06-27"
    },
    {
      "label": "Submission deadline",
      "date": "2026-06-27"
    }
  ]
}
```

### VoltHacks

```json
{
  "title": "VoltHacks",
  "type": "hackathon",
  "url": "https://volthacks.devpost.com/",
  "org": "Dialogate",
  "location": "Online",
  "blurb": "IoT, Machine Learning/AI, Beginner Friendly hackathon on Devpost. Prizes: $2,905. 2 months left.",
  "source_platform": "Devpost",
  "source_url": "https://volthacks.devpost.com/",
  "confidence": "High",
  "image_url": "https://volthacks.devpost.com/",
  "dates": [
    {
      "label": "Hackathon window",
      "date": "2026-05-22",
      "endDate": "2026-09-05"
    },
    {
      "label": "Submission deadline",
      "date": "2026-09-05"
    }
  ],
  "beginner_friendly": true
}
```

### Build the Future with AI: From Code to No-Code

```json
{
  "title": "Build the Future with AI — From Code to No-Code",
  "type": "hackathon",
  "url": "https://build-the-future-with-ai.devpost.com/",
  "org": "Innovation Hacks",
  "location": "Online",
  "blurb": "Low/No Code, Machine Learning/AI, Web hackathon on Devpost. Prizes: $0. 7 days left.",
  "source_platform": "Devpost",
  "source_url": "https://build-the-future-with-ai.devpost.com/",
  "confidence": "High",
  "image_url": "https://build-the-future-with-ai.devpost.com/",
  "dates": [
    {
      "label": "Hackathon window",
      "date": "2026-05-31",
      "endDate": "2026-06-30"
    },
    {
      "label": "Submission deadline",
      "date": "2026-06-30"
    }
  ]
}
```

### Github readme generation

```json
{
  "title": "Github readme generation",
  "type": "hackathon",
  "url": "https://github-readme-generation.devpost.com/",
  "org": "presentme",
  "location": "Online",
  "blurb": "Beginner Friendly, Education, Low/No Code hackathon on Devpost. Prizes: £0. 13 days left.",
  "source_platform": "Devpost",
  "source_url": "https://github-readme-generation.devpost.com/",
  "confidence": "High",
  "image_url": "https://github-readme-generation.devpost.com/",
  "dates": [
    {
      "label": "Hackathon window",
      "date": "2026-04-22",
      "endDate": "2026-07-06"
    },
    {
      "label": "Submission deadline",
      "date": "2026-07-06"
    }
  ],
  "beginner_friendly": true
}
```

### Moonshot Hackathon

```json
{
  "title": "Moonshot Hackathon",
  "type": "hackathon",
  "url": "https://moonshot-aethra.devpost.com/",
  "org": "Aethra",
  "location": "Online",
  "blurb": "Beginner Friendly, Machine Learning/AI, Open Ended hackathon on Devpost. Prizes: $33,532. 7 days left.",
  "source_platform": "Devpost",
  "source_url": "https://moonshot-aethra.devpost.com/",
  "confidence": "High",
  "image_url": "https://moonshot-aethra.devpost.com/",
  "dates": [
    {
      "label": "Hackathon window",
      "date": "2026-06-03",
      "endDate": "2026-06-30"
    },
    {
      "label": "Submission deadline",
      "date": "2026-06-30"
    }
  ],
  "beginner_friendly": true
}
```

### Hoobit Hacks 2026

```json
{
  "title": "Hoobit Hacks 2026",
  "type": "hackathon",
  "url": "https://hoobit-hacks-2026.devpost.com/",
  "org": "Hoobit",
  "location": "Online",
  "blurb": "Beginner Friendly, Machine Learning/AI, Social Good hackathon on Devpost. Prizes: $0. 25 days left.",
  "source_platform": "Devpost",
  "source_url": "https://hoobit-hacks-2026.devpost.com/",
  "confidence": "High",
  "image_url": "https://hoobit-hacks-2026.devpost.com/",
  "dates": [
    {
      "label": "Hackathon window",
      "date": "2026-03-30",
      "endDate": "2026-07-18"
    },
    {
      "label": "Submission deadline",
      "date": "2026-07-18"
    }
  ],
  "beginner_friendly": true
}
```

### FutureAI Global Hackathon 2026

```json
{
  "title": "FutureAI Global Hackathon 2026",
  "type": "hackathon",
  "url": "https://futureai-global-hackthon.devpost.com/",
  "org": "Innovation Hacks",
  "location": "Online",
  "blurb": "Beginner Friendly, Machine Learning/AI, Open Ended hackathon on Devpost. Prizes: $0. 12 days left.",
  "source_platform": "Devpost",
  "source_url": "https://futureai-global-hackthon.devpost.com/",
  "confidence": "High",
  "image_url": "https://futureai-global-hackthon.devpost.com/",
  "dates": [
    {
      "label": "Hackathon window",
      "date": "2026-05-29",
      "endDate": "2026-07-05"
    },
    {
      "label": "Submission deadline",
      "date": "2026-07-05"
    }
  ],
  "beginner_friendly": true
}
```

### Global Tech Innovation 2026

```json
{
  "title": "Global Tech Innovation 2026",
  "type": "hackathon",
  "url": "https://global-tech-innovation-2026.devpost.com/",
  "org": "Innovation Hacks",
  "location": "Online",
  "blurb": "Cybersecurity, IoT, Web hackathon on Devpost. Prizes: $0. 7 days left.",
  "source_platform": "Devpost",
  "source_url": "https://global-tech-innovation-2026.devpost.com/",
  "confidence": "High",
  "image_url": "https://global-tech-innovation-2026.devpost.com/",
  "dates": [
    {
      "label": "Hackathon window",
      "date": "2026-06-10",
      "endDate": "2026-06-30"
    },
    {
      "label": "Submission deadline",
      "date": "2026-06-30"
    }
  ]
}
```

### Built with Python Hackathon

```json
{
  "title": "Built with Python Hackathon",
  "type": "hackathon",
  "url": "https://built-with-python-hackathon.devpost.com/",
  "org": "CS4Everyone",
  "location": "Online",
  "blurb": "Beginner Friendly, Education, Machine Learning/AI hackathon on Devpost. Prizes: $0. 4 days left.",
  "source_platform": "Devpost",
  "source_url": "https://built-with-python-hackathon.devpost.com/",
  "confidence": "High",
  "image_url": "https://built-with-python-hackathon.devpost.com/",
  "dates": [
    {
      "label": "Hackathon window",
      "date": "2026-06-06",
      "endDate": "2026-06-27"
    },
    {
      "label": "Submission deadline",
      "date": "2026-06-27"
    }
  ],
  "beginner_friendly": true
}
```

### Brainwave 2026

```json
{
  "title": "Brainwave 2026",
  "type": "hackathon",
  "url": "https://brainwaves.devpost.com/",
  "org": "ACT House",
  "location": "Online",
  "blurb": "AR/VR, Blockchain, Communication hackathon on Devpost. Prizes: $1,000. about 2 months left.",
  "source_platform": "Devpost",
  "source_url": "https://brainwaves.devpost.com/",
  "confidence": "High",
  "image_url": "https://brainwaves.devpost.com/",
  "dates": [
    {
      "label": "Hackathon window",
      "date": "2026-06-12",
      "endDate": "2026-08-09"
    },
    {
      "label": "Submission deadline",
      "date": "2026-08-09"
    }
  ]
}
```

### Build with Gemini

```json
{
  "title": "Build with Gemini",
  "type": "hackathon",
  "url": "https://build-with-gemini-0.devpost.com/",
  "org": "MLH",
  "location": "Online",
  "blurb": "Beginner Friendly, Education, Web hackathon on Devpost. Prizes: $0. 3 days left.",
  "source_platform": "Devpost",
  "source_url": "https://build-with-gemini-0.devpost.com/",
  "confidence": "High",
  "image_url": "https://build-with-gemini-0.devpost.com/",
  "dates": [
    {
      "label": "Hackathon window",
      "date": "2026-06-09",
      "endDate": "2026-06-26"
    },
    {
      "label": "Submission deadline",
      "date": "2026-06-26"
    }
  ],
  "beginner_friendly": true
}
```

### Assistive Innovation Challenge 2026

```json
{
  "title": "Assistive Innovation Challenge 2026",
  "type": "hackathon",
  "url": "https://assistive-innovation-challenge.devpost.com/",
  "org": "Student Innovators Without Borders",
  "location": "Online",
  "blurb": "Beginner Friendly, Health, Social Good hackathon on Devpost. Prizes: $0. about 1 month left.",
  "source_platform": "Devpost",
  "source_url": "https://assistive-innovation-challenge.devpost.com/",
  "confidence": "High",
  "image_url": "https://assistive-innovation-challenge.devpost.com/",
  "dates": [
    {
      "label": "Hackathon window",
      "date": "2026-05-04",
      "endDate": "2026-08-01"
    },
    {
      "label": "Submission deadline",
      "date": "2026-08-01"
    }
  ],
  "beginner_friendly": true
}
```

### GLITCHED GAMES

```json
{
  "title": "GLITCHED GAMES",
  "type": "hackathon",
  "url": "https://glitch-to-win-games.devpost.com/",
  "org": "GB3-Productions",
  "location": "Online",
  "blurb": "Gaming, Mobile, Web hackathon on Devpost. Prizes: $0. 7 days left.",
  "source_platform": "Devpost",
  "source_url": "https://glitch-to-win-games.devpost.com/",
  "confidence": "High",
  "image_url": "https://glitch-to-win-games.devpost.com/",
  "dates": [
    {
      "label": "Hackathon window",
      "date": "2026-04-16",
      "endDate": "2026-06-30"
    },
    {
      "label": "Submission deadline",
      "date": "2026-06-30"
    }
  ]
}
```

### Cyber_Coders

```json
{
  "title": "Cyber_Coders",
  "type": "hackathon",
  "url": "https://cybercoders2026.devpost.com/",
  "org": "Idustries",
  "location": "Online",
  "blurb": "Cybersecurity, Web hackathon on Devpost. Prizes: $18. 28 minutes left.",
  "source_platform": "Devpost",
  "source_url": "https://cybercoders2026.devpost.com/",
  "confidence": "High",
  "image_url": "https://cybercoders2026.devpost.com/",
  "dates": [
    {
      "label": "Hackathon window",
      "date": "2026-06-13",
      "endDate": "2026-06-23"
    },
    {
      "label": "Submission deadline",
      "date": "2026-06-23"
    }
  ]
}
```

### InnovatorsX: Startup Sprint

```json
{
  "title": "InnovatorsX: Startup Sprint",
  "type": "hackathon",
  "url": "https://innovatorsx.devpost.com/",
  "org": "InnovatorsX",
  "location": "Online",
  "blurb": "Beginner Friendly, Design, E-commerce/Retail hackathon on Devpost. Prizes: $0. 6 days left.",
  "source_platform": "Devpost",
  "source_url": "https://innovatorsx.devpost.com/",
  "confidence": "High",
  "image_url": "https://innovatorsx.devpost.com/",
  "dates": [
    {
      "label": "Hackathon window",
      "date": "2026-06-05",
      "endDate": "2026-06-29"
    },
    {
      "label": "Submission deadline",
      "date": "2026-06-29"
    }
  ],
  "beginner_friendly": true
}
```

### Next Byte Hacks V3

```json
{
  "title": "Next Byte Hacks V3",
  "type": "hackathon",
  "url": "https://next-byte-hacks-v3.devpost.com/",
  "org": "Next Bytes",
  "location": "Online",
  "blurb": "Beginner Friendly, Open Ended, Social Good hackathon on Devpost. Prizes: $100. 22 days left.",
  "source_platform": "Devpost",
  "source_url": "https://next-byte-hacks-v3.devpost.com/",
  "confidence": "High",
  "image_url": "https://next-byte-hacks-v3.devpost.com/",
  "dates": [
    {
      "label": "Hackathon window",
      "date": "2026-06-15",
      "endDate": "2026-07-15"
    },
    {
      "label": "Submission deadline",
      "date": "2026-07-15"
    }
  ],
  "beginner_friendly": true
}
```

### Hyperbloom Summer Hackathon

```json
{
  "title": "Hyperbloom Summer Hackathon",
  "type": "hackathon",
  "url": "https://hyperbloom-summer-hackathon.devpost.com/",
  "org": "hyperbloom hacks",
  "location": "Online",
  "blurb": "Education, Open Ended, Social Good hackathon on Devpost. Prizes: $12,000. 7 days left.",
  "source_platform": "Devpost",
  "source_url": "https://hyperbloom-summer-hackathon.devpost.com/",
  "confidence": "High",
  "image_url": "https://hyperbloom-summer-hackathon.devpost.com/",
  "dates": [
    {
      "label": "Hackathon window",
      "date": "2026-06-07",
      "endDate": "2026-06-30"
    },
    {
      "label": "Submission deadline",
      "date": "2026-06-30"
    }
  ]
}
```

### 1st Nature's Blueprint Ideathon 2026 !!

```json
{
  "title": "1st Nature's Blueprint Ideathon 2026 !!",
  "type": "hackathon",
  "url": "https://naturesblueprint.devpost.com/",
  "org": "Yaod Multifaceted Company",
  "location": "Online",
  "blurb": "Beginner Friendly, Education, Social Good hackathon on Devpost. Prizes: $0. 7 days left.",
  "source_platform": "Devpost",
  "source_url": "https://naturesblueprint.devpost.com/",
  "confidence": "High",
  "image_url": "https://naturesblueprint.devpost.com/",
  "dates": [
    {
      "label": "Hackathon window",
      "date": "2026-05-31",
      "endDate": "2026-06-30"
    },
    {
      "label": "Submission deadline",
      "date": "2026-06-30"
    }
  ],
  "beginner_friendly": true
}
```

### Hack Begin

```json
{
  "title": "Hack Begin",
  "type": "hackathon",
  "url": "https://hack-begin.devpost.com/",
  "org": "Shankara Institute of Technology ",
  "location": "Online",
  "blurb": "Blockchain, Machine Learning/AI, Open Ended hackathon on Devpost. Prizes: $0. 1 day left.",
  "source_platform": "Devpost",
  "source_url": "https://hack-begin.devpost.com/",
  "confidence": "High",
  "image_url": "https://hack-begin.devpost.com/",
  "dates": [
    {
      "label": "Hackathon window",
      "date": "2026-05-19",
      "endDate": "2026-06-25"
    },
    {
      "label": "Submission deadline",
      "date": "2026-06-25"
    }
  ]
}
```

### AceSAT Education AI-Agent

```json
{
  "title": "AceSAT Education AI-Agent",
  "type": "hackathon",
  "url": "https://acesat-ai-agent.devpost.com/",
  "org": "AceSAT",
  "location": "Online",
  "blurb": "Education, Machine Learning/AI hackathon on Devpost. Prizes: $100. about 2 months left.",
  "source_platform": "Devpost",
  "source_url": "https://acesat-ai-agent.devpost.com/",
  "confidence": "High",
  "image_url": "https://acesat-ai-agent.devpost.com/",
  "dates": [
    {
      "label": "Hackathon window",
      "date": "2026-06-12",
      "endDate": "2026-08-15"
    },
    {
      "label": "Submission deadline",
      "date": "2026-08-15"
    }
  ]
}
```

### Web Champ

```json
{
  "title": "Web Champ",
  "type": "hackathon",
  "url": "https://web-champ.devpost.com/",
  "org": "The Fusion",
  "location": "Online",
  "blurb": "Design, E-commerce/Retail, Web hackathon on Devpost. Prizes: $0. 23 days left.",
  "source_platform": "Devpost",
  "source_url": "https://web-champ.devpost.com/",
  "confidence": "High",
  "image_url": "https://web-champ.devpost.com/",
  "dates": [
    {
      "label": "Hackathon window",
      "date": "2026-06-16",
      "endDate": "2026-07-16"
    },
    {
      "label": "Submission deadline",
      "date": "2026-07-16"
    }
  ]
}
```

### AQX Sports Analytics Hackathon

```json
{
  "title": "AQX Sports Analytics Hackathon",
  "type": "hackathon",
  "url": "https://aqxanalytics.devpost.com/",
  "org": "James Logan High School",
  "location": "Online",
  "blurb": "Beginner Friendly, Databases, Gaming hackathon on Devpost. Prizes: $0. 2 days left.",
  "source_platform": "Devpost",
  "source_url": "https://aqxanalytics.devpost.com/",
  "confidence": "High",
  "image_url": "https://aqxanalytics.devpost.com/",
  "dates": [
    {
      "label": "Hackathon window",
      "date": "2026-06-10",
      "endDate": "2026-06-26"
    },
    {
      "label": "Submission deadline",
      "date": "2026-06-26"
    }
  ],
  "beginner_friendly": true
}
```

### Ventura Challenge

```json
{
  "title": "Ventura Challenge",
  "type": "hackathon",
  "url": "https://ventura-challenge.devpost.com/",
  "org": "Student Council",
  "location": "Online",
  "blurb": "Beginner Friendly, Design, E-commerce/Retail hackathon on Devpost. Prizes: $0. about 1 month left.",
  "source_platform": "Devpost",
  "source_url": "https://ventura-challenge.devpost.com/",
  "confidence": "High",
  "image_url": "https://ventura-challenge.devpost.com/",
  "dates": [
    {
      "label": "Hackathon window",
      "date": "2026-06-01",
      "endDate": "2026-07-29"
    },
    {
      "label": "Submission deadline",
      "date": "2026-07-29"
    }
  ],
  "beginner_friendly": true
}
```

---

## Role opportunities

Calls for judges, mentors, speakers, or volunteers found in Devpost hackathon descriptions. `url` points to the apply form, email, or description anchor when no direct link exists.

### H0: Hack the Zero Stack with Vercel v0 and AWS Databases: Call for Judges

```json
{
  "title": "H0: Hack the Zero Stack with Vercel v0 and AWS Databases: Call for Judges",
  "type": "event",
  "url": "https://forms.gle/FzLd8BLqzzrkuBMU7",
  "org": "Amazon",
  "location": "Online",
  "blurb": "Devpost hackathon recruiting judges. judge opportunity mentioned in hackathon description.",
  "source_platform": "Devpost",
  "source_url": "https://forms.gle/FzLd8BLqzzrkuBMU7",
  "confidence": "High",
  "image_url": "https://h01.devpost.com/",
  "dates": [
    {
      "label": "Hackathon window",
      "date": "2026-05-27",
      "endDate": "2026-06-29"
    },
    {
      "label": "Hackathon submission deadline",
      "date": "2026-06-29"
    }
  ]
}
```

### PhysTech 2026: Physical Activity and Technology Hack Day: Call for Judges

```json
{
  "title": "PhysTech 2026: Physical Activity and Technology Hack Day: Call for Judges",
  "type": "event",
  "url": "mailto:phystech@yahoo.cim\">phystech@yahoo.com</a></span></strong></p>\\n<p><strong><span",
  "org": "Binnovative",
  "location": "Online",
  "blurb": "Devpost hackathon recruiting judges. judge opportunity mentioned in hackathon description.",
  "source_platform": "Devpost",
  "source_url": "mailto:phystech@yahoo.cim\">phystech@yahoo.com</a></span></strong></p>\\n<p><strong><span",
  "confidence": "High",
  "image_url": "https://phystech-2026.devpost.com/",
  "dates": [
    {
      "label": "Hackathon window",
      "date": "2026-02-08",
      "endDate": "2026-06-27"
    },
    {
      "label": "Hackathon submission deadline",
      "date": "2026-06-27"
    }
  ]
}
```

### PhysTech 2026: Physical Activity and Technology Hack Day: Call for Coaches

```json
{
  "title": "PhysTech 2026: Physical Activity and Technology Hack Day: Call for Coaches",
  "type": "event",
  "url": "mailto:phystech@yahoo.cim\">phystech@yahoo.com</a></span></strong></p>\\n<p><strong><span",
  "org": "Binnovative",
  "location": "Online",
  "blurb": "Devpost hackathon recruiting coaches. coach opportunity mentioned in hackathon description.",
  "source_platform": "Devpost",
  "source_url": "mailto:phystech@yahoo.cim\">phystech@yahoo.com</a></span></strong></p>\\n<p><strong><span",
  "confidence": "High",
  "image_url": "https://phystech-2026.devpost.com/",
  "dates": [
    {
      "label": "Hackathon window",
      "date": "2026-02-08",
      "endDate": "2026-06-27"
    },
    {
      "label": "Hackathon submission deadline",
      "date": "2026-06-27"
    }
  ]
}
```

### VoltHacks: Call for Judges

```json
{
  "title": "VoltHacks: Call for Judges",
  "type": "event",
  "url": "mailto:ehsansadiq141@gmail.com",
  "org": "Dialogate",
  "location": "Online",
  "blurb": "Devpost hackathon recruiting judges. judge opportunity mentioned in hackathon description.",
  "source_platform": "Devpost",
  "source_url": "mailto:ehsansadiq141@gmail.com",
  "confidence": "High",
  "image_url": "https://volthacks.devpost.com/",
  "dates": [
    {
      "label": "Hackathon window",
      "date": "2026-05-22",
      "endDate": "2026-09-05"
    },
    {
      "label": "Hackathon submission deadline",
      "date": "2026-09-05"
    }
  ]
}
```

### Brainwave 2026: Call for Mentors

```json
{
  "title": "Brainwave 2026: Call for Mentors",
  "type": "event",
  "url": "https://forms.gle/AtrJWgV56dbyQ7BJ6",
  "org": "ACT House",
  "location": "Online",
  "blurb": "Devpost hackathon recruiting mentors. mentor opportunity mentioned in hackathon description.",
  "source_platform": "Devpost",
  "source_url": "https://forms.gle/AtrJWgV56dbyQ7BJ6",
  "confidence": "High",
  "image_url": "https://brainwaves.devpost.com/",
  "dates": [
    {
      "label": "Hackathon window",
      "date": "2026-06-12",
      "endDate": "2026-08-09"
    },
    {
      "label": "Hackathon submission deadline",
      "date": "2026-08-09"
    }
  ]
}
```

### AceSAT Education AI-Agent: Call for Judges

```json
{
  "title": "AceSAT Education AI-Agent: Call for Judges",
  "type": "event",
  "url": "mailto:acesat.tx@gmail.com",
  "org": "AceSAT",
  "location": "Online",
  "blurb": "Devpost hackathon recruiting judges. judge opportunity mentioned in hackathon description.",
  "source_platform": "Devpost",
  "source_url": "mailto:acesat.tx@gmail.com",
  "confidence": "High",
  "image_url": "https://acesat-ai-agent.devpost.com/",
  "dates": [
    {
      "label": "Hackathon window",
      "date": "2026-06-12",
      "endDate": "2026-08-15"
    },
    {
      "label": "Hackathon submission deadline",
      "date": "2026-08-15"
    }
  ]
}
```

### Web Champ: Call for Judges

```json
{
  "title": "Web Champ: Call for Judges",
  "type": "event",
  "url": "mailto:abhishekparindya2007may3@gmail.com",
  "org": "The Fusion",
  "location": "Online",
  "blurb": "Devpost hackathon recruiting judges. judge opportunity mentioned in hackathon description.",
  "source_platform": "Devpost",
  "source_url": "mailto:abhishekparindya2007may3@gmail.com",
  "confidence": "High",
  "image_url": "https://web-champ.devpost.com/",
  "dates": [
    {
      "label": "Hackathon window",
      "date": "2026-06-16",
      "endDate": "2026-07-16"
    },
    {
      "label": "Hackathon submission deadline",
      "date": "2026-07-16"
    }
  ]
}
```
