// Shared talks data used by the home bento, the /talks listing, and per-talk pages.
// Slugs are intentionally explicit (not derived from title) so URLs remain stable
// even if a title is edited. Add new talks at the bottom; never reuse a slug.

export const talks = [
  {
    slug: "dsg-codeit-day1-what-is-data-science",
    title: "UPLB DSG x UPRHS CodeIT Workshop Day 1 - What is Data Science?",
    src: "/talks/talk1.jpg",
    date: "2024",
    event: "UPLB Data Science Guild x UPRHS CodeIT",
    type: "Workshop",
    description:
      "An introductory workshop on data science fundamentals for high school students.",
    slidesLink: "/r/workshops/dsg-codeit-day1",
  },
  {
    slug: "dsg-codeit-day2-storytelling-with-data",
    title: "UPLB DSG x UPRHS CodeIT Workshop Day 2 - Storytelling with Data",
    src: "/talks/talk2.jpg",
    date: "2024",
    event: "UPLB Data Science Guild x UPRHS CodeIT",
    type: "Workshop",
    description:
      "Teaching students how to communicate insights effectively through data visualization and narrative.",
    slidesLink: "/r/workshops/dsg-codeit-day2",
  },
  {
    slug: "nextstep-hacks-2025-winning-by-talking",
    title: "NextStep Hacks 2025 - Winning by Talking",
    src: "/talks/talk3.jpg",
    date: "2025",
    event: "NextStep Hacks 2025",
    type: "Talk",
    description:
      "A talk on presentation skills and how to pitch your hackathon projects effectively.",
    slidesLink: "/r/workshops/nextstep-hacks",
  },
  {
    slug: "jpcs-qcu-logic-unlocked-ml-with-python",
    title:
      "JPCS - QCU Logic Unlocked Day 1 - Machine Learning with Python",
    src: "/talks/talk4.jpg",
    date: "2025",
    event: "JPCS QCU Logic Unlocked",
    type: "Workshop",
    description:
      "Hands-on workshop introducing machine learning concepts using Python and popular ML libraries.",
    slidesLink: "/r/talks/qcu-ml-python",
  },
  {
    slug: "dsg-applicants-data-storytelling-canva",
    title: "UPLB DSG Applicants' Workshop - Data Storytelling with Canva",
    src: "/talks/talk5.jpg",
    date: "2024",
    event: "UPLB Data Science Guild",
    type: "Workshop",
    description:
      "Teaching applicants how to create compelling data visualizations using Canva.",
    slidesLink: null,
  },
  {
    slug: "dep-ai-use-cases-that-actually-matter",
    title:
      "Data Engineering Pilipinas AI Study Group - AI Use Cases That Actually Matter",
    src: "/talks/talk6.jpg",
    date: "2025",
    event: "Data Engineering Pilipinas",
    type: "Talk",
    description: "Discussing practical AI applications that create real-world impact.",
    slidesLink: "/r/talks/dep-ai-study",
  },
  {
    slug: "dlsu-eces-agile-edge-swift-project-workflows",
    title: "DLSU ECES - Agile Edge: Swift Project Workflows",
    src: "/talks/talk7.gif",
    date: "2025",
    event: "DLSU ECES",
    type: "Talk",
    description: "Upcoming talk on agile methodologies and efficient project management.",
    slidesLink: null,
    upcoming: true,
  },
  {
    slug: "ml-iris-classification-org-applicants",
    title: "ML Workshop: Iris Classification",
    src: "/talks/talk1.jpg",
    date: "2024",
    event: "Org Applicants",
    type: "Workshop",
    description: "Introductory ML workshop focusing on the classic Iris dataset classification.",
    slidesLink: "https://workshops.stimmie.dev/ml-iris-slides",
  },
  {
    slug: "python-in-research-ust",
    title: "Python in Research",
    src: "/talks/talk2.jpg",
    date: "2024",
    event: "UST",
    type: "Workshop",
    description: "Applying Python for data analysis and automation in academic research.",
    slidesLink: "https://workshops.stimmie.dev/python",
  },
  {
    slug: "ml-workshop-smcl",
    title: "Machine Learning Workshop",
    src: "/talks/talk3.jpg",
    date: "2024",
    event: "Saint Michael's College of Laguna",
    type: "Workshop",
    description: "A comprehensive machine learning workshop for students at SMCL.",
    slidesLink: null,
  },
];

export function getTalkBySlug(slug) {
  return talks.find((t) => t.slug === slug);
}
