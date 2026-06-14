// src/data/talks.js

// Shared talks data used by the home bento, the /talks listing, and per-talk pages.
// Slugs are intentionally explicit (not derived from title) so URLs remain stable
// even if a title is edited. Add new talks at the bottom; never reuse a slug.
//
// `audienceSize` is a rough estimate of how many people attended each talk/
// workshop. It's summed on the /talks page to show a running total reach.
// These are estimates — adjust them to match your real numbers.

export const talks = [
  {
    slug: "dsg-codeit-day1-what-is-data-science",
    title: "UPLB DSG x UPRHS CodeIT Workshop Day 1 - What is Data Science?",
    actionPhoto: null,
    slidesThumbnail: "/talks/uprhs-day1-thumbnail.png",
    date: "2025-04-04T10:00:00+08:00",
    event: "UPLB Data Science Guild x UPRHS CodeIT",
    type: "Workshop",
    audienceSize: 40,
    description:
      "An introductory workshop on data science fundamentals for high school students.",
    slidesLink: "/r/workshops/dsg-codeit-day1",
  },
  {
    slug: "dsg-codeit-day2-storytelling-with-data",
    title: "UPLB DSG x UPRHS CodeIT Workshop Day 2 - Storytelling with Data",
    actionPhoto: "/talks/uprhs-day2.jpg",
    slidesThumbnail: null,
    date: "2025-04-11T10:00:00+08:00",
    event: "UPLB Data Science Guild x UPRHS CodeIT",
    type: "Workshop",
    audienceSize: 40,
    description:
      "Teaching students how to communicate insights effectively through data visualization and narrative.",
    slidesLink: "/r/workshops/dsg-codeit-day2",
  },
  {
    slug: "nextstep-hacks-2025-winning-by-talking",
    title: "NextStep Hacks 2025 - Winning by Talking",
    actionPhoto: "/talks/talk3.jpg",
    slidesThumbnail: null,
    date: "2025-01-20T14:00:00+08:00",
    event: "NextStep Hacks 2025",
    type: "Talk",
    audienceSize: 100,
    description:
      "A talk on presentation skills and how to pitch your hackathon projects effectively.",
    slidesLink: "/r/workshops/nextstep-hacks",
  },
  {
    slug: "jpcs-qcu-logic-unlocked-ml-with-python",
    title: "JPCS - QCU Logic Unlocked Day 1 - Machine Learning with Python",
    actionPhoto: null,
    slidesThumbnail: "/talks/jpcs-qcu-thumbnail.jpg",
    date: "2025-09-25T09:00:00+08:00",
    event: "JPCS QCU Logic Unlocked",
    type: "Workshop",
    audienceSize: 60,
    description:
      "Hands-on workshop introducing machine learning concepts using Python and popular ML libraries.",
    slidesLink: "/r/talks/qcu-ml-python",
  },
  {
    slug: "dsg-applicants-data-storytelling-canva",
    title: "UPLB DSG Applicants' Workshop - Data Storytelling with Canva",
    actionPhoto: "/talks/talk5.jpg",
    slidesThumbnail: null,
    date: "2025-10-05T13:00:00+08:00",
    event: "UPLB Data Science Guild",
    type: "Workshop",
    audienceSize: 60,
    description:
      "Teaching applicants how to create compelling data visualizations using Canva.",
    slidesLink: null,
  },
  {
    slug: "dep-ai-use-cases-that-actually-matter",
    title:
      "Data Engineering Pilipinas AI Study Group - AI Use Cases That Actually Matter",
    actionPhoto: "/talks/talk6.jpg",
    slidesThumbnail: null,
    date: "2025-10-12T10:00:00+08:00",
    event: "Data Engineering Pilipinas",
    type: "Talk",
    audienceSize: 150,
    description:
      "Discussing practical AI applications that create real-world impact.",
    slidesLink: "https://canva.link/xn1ghgvhtk628aw",
    replayLink: "https://youtu.be/fL23CKJDGN8?si=tDRsPeaSlTmVUG0f",
    moderator: {
      name: "Allan Tan",
      link: "https://www.linkedin.com/in/allanctan/",
    },
  },
  {
    slug: "dlsu-eces-agile-edge-swift-project-workflows",
    title: "DLSU ECES - Agile Edge: Swift Project Workflows",
    actionPhoto: "/talks/dlsu-eces-agile-edge.jpg",
    slidesThumbnail: null,
    date: "2025-11-05T14:00:00+08:00",
    event: "DLSU ECES",
    type: "Talk",
    audienceSize: 80,
    description:
      "Talk on agile methodologies and efficient project management for the DLSU ECES.",
    slidesLink: null,
  },
  {
    slug: "ml-iris-classification-org-applicants",
    title: "ML Workshop: Iris Classification",
    actionPhoto: "/talks/dsg-apps-workshop.jpeg",
    slidesThumbnail: null,
    date: "2026-03-02T10:00:00+08:00",
    event: "UPLB Data Science Guild Apps' Workshop",
    type: "Workshop",
    audienceSize: 50,
    description:
      "Introductory ML workshop focusing on the classic Iris dataset classification.",
    slidesLink: "https://workshops.stimmie.dev/ml-iris-slides",
  },
  {
    slug: "python-in-research-ust",
    title: "Python in Research",
    actionPhoto: null,
    slidesThumbnail: "/talks/python-in-research-thumbnail.png",
    date: "2026-04-24T13:30:00+08:00",
    event:
      "Decode to Defend: A Research Refresher by UST Computer Science Society",
    type: "Workshop",
    audienceSize: 70,
    description:
      "Applying Python for data analysis and automation in academic research.",
    slidesLink: "https://workshops.stimmie.dev/python",
  },
  {
    slug: "ml-workshop-smcl",
    title: "Machine Learning Workshop",
    actionPhoto: "/talks/ml-workshop-smcl.jpg",
    slidesThumbnail: null,
    date: "2024-12-05T09:00:00+08:00",
    event: "Saint Michael's College of Laguna",
    type: "Workshop",
    audienceSize: 40,
    description:
      "A comprehensive machine learning workshop for students at SMCL.",
    slidesLink: null,
  },
  {
    slug: "hearthcraft-40000-player-minecraft-server",
    title: "HearthCraft: A 40,000-Player Minecraft Server",
    actionPhoto: null,
    slidesThumbnail: "/talks/hearthcraft-thumbnail.png",
    date: "2026-02-02T10:00:00+08:00",
    event: "HearthCraft",
    type: "Talk",
    audienceSize: 100,
    description:
      "Deep dive into managing and scaling a 40,000-player Minecraft server community.",
    slidesLink: null,
    replayLink: "#",
    moderator: {
      name: "Renzi Vidal",
      link: "https://www.linkedin.com/in/ultrenz-vidal/",
    },
  },
  {
    slug: "present-insights-better-batangas-eastern-colleges",
    title: "Present Insights Better",
    actionPhoto: null,
    slidesThumbnail: "/talks/present-insights-better-thumbnail.png",
    date: "2026-02-12T10:00:00+08:00",
    event: "Batangas Eastern Colleges",
    type: "Talk",
    audienceSize: 60,
    description:
      "A talk on the intersection of data science and data analytics to better present insights.",
    slidesLink: null,
  },
];

export function getTalkBySlug(slug) {
  return talks.find((t) => t.slug === slug);
}

/** Estimated total number of people reached across all talks and workshops. */
export function getTotalAudience() {
  return talks.reduce((sum, talk) => sum + (talk.audienceSize || 0), 0);
}
