// Frozen v2.0 talks snapshot — thumbnails live under public/archive/v2/talks/.

import { ARCHIVE_V2 } from "@/lib/archive-paths";

const t = (file) => `${ARCHIVE_V2}/talks/${file}`;

export const talks = [
  {
    slug: "dsg-codeit-day1-what-is-data-science",
    title: "UPLB DSG x UPRHS CodeIT Workshop Day 1 - What is Data Science?",
    thumbnail: t("uprhs-day1-thumbnail.png"),
  },
  {
    slug: "dsg-codeit-day2-storytelling-with-data",
    title: "UPLB DSG x UPRHS CodeIT Workshop Day 2 - Storytelling with Data",
    thumbnail: t("uprhs-day2.jpg"),
  },
  {
    slug: "nextstep-hacks-2025-winning-by-talking",
    title: "NextStep Hacks 2025 - Winning by Talking",
    thumbnail: t("talk3.jpg"),
  },
  {
    slug: "jpcs-qcu-logic-unlocked-ml-with-python",
    title: "JPCS - QCU Logic Unlocked Day 1 - Machine Learning with Python",
    thumbnail: t("jpcs-qcu-thumbnail.jpg"),
  },
  {
    slug: "dsg-applicants-data-storytelling-canva",
    title: "UPLB DSG Applicants' Workshop - Data Storytelling with Canva",
    thumbnail: t("talk5.jpg"),
  },
  {
    slug: "dep-ai-use-cases-that-actually-matter",
    title:
      "Data Engineering Pilipinas AI Study Group - AI Use Cases That Actually Matter",
    thumbnail: t("talk6.jpg"),
  },
  {
    slug: "dlsu-eces-agile-edge-swift-project-workflows",
    title: "DLSU ECES - Agile Edge: Swift Project Workflows",
    thumbnail: t("dlsu-eces-agile-edge.jpg"),
  },
  {
    slug: "ml-iris-classification-org-applicants",
    title: "ML Workshop: Iris Classification",
    thumbnail: t("dsg-apps-workshop.jpeg"),
  },
  {
    slug: "python-in-research-ust",
    title: "Python in Research",
    thumbnail: t("python-in-research-thumbnail.png"),
  },
  {
    slug: "ml-workshop-smcl",
    title: "Machine Learning Workshop",
    thumbnail: t("ml-workshop-smcl.jpg"),
  },
  {
    slug: "hearthcraft-40000-player-minecraft-server",
    title: "HearthCraft: A 40,000-Player Minecraft Server",
    thumbnail: t("hearthcraft-thumbnail.png"),
  },
  {
    slug: "present-insights-better-batangas-eastern-colleges",
    title: "Present Insights Better",
    thumbnail: t("present-insights-better-thumbnail.png"),
  },
];
