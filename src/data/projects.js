// src/data/projects.js

// Shared projects data used by the home bento, the /projects listing, and per-project pages.
// Slugs are intentionally explicit (not derived from title) so URLs remain stable
// even if a title is edited. Add new projects at the bottom; never reuse a slug.

export const projects = [
  {
    slug: "hearthcraft",
    title: "HearthCraft",
    src: "/projects/project1.jpg",
    date: "2018 – 2025",
    description:
      "A long-running, non-pay-to-win Minecraft survival server and community I built and led for seven years — 40,000+ players, zero resets, and a lot of lessons in infra, moderation, and keeping a place actually safe.",
    body: [
      "At 13 I turned a hobby into HearthCraft: a survival Minecraft server I ran from 2018 through 2025. What started as learning to host on managed and bare-metal boxes — plugins, Docker, backups, all of it — grew into a community of 40,000+ players over seven years.",
      "The server never reset. That was the point: a persistent world where people could build, hang out, and come back to the same place. I kept it non-pay-to-win, wrote and configured Java plugins, handled moderation, and treated it less like a game and more like a small platform with real ops work.",
      "Revenue from optional cosmetics helped fund college and occasional charitable givebacks. Running HearthCraft taught me more about reliability, community dynamics, and saying no to bad ideas than most of my early coding jobs combined.",
    ],
    tags: ["Minecraft", "Docker", "Java", "Community"],
    link: "https://hearthcraft.stimmie.dev",
  },
  {
    slug: "atlas-of-my-skies",
    title: "Atlas Of My Skies",
    src: "/projects/project2.jpg",
    date: "2025",
    description:
      "A photography portfolio mapping the skies I've chased — sunsets, storms, and the quiet in-between — told as a visual story rather than a grid of pretty pictures.",
    body: [
      "Atlas of My Skies is a photography project I launched in 2025. I've been taking photos of the sky for years — on commutes, photowalks, and random Tuesdays when the light looked right — and this site is my attempt to give that habit a narrative.",
      "Instead of dumping images into an album, Atlas organizes them as a story: places, moods, and sequences that connect how I see the world above the horizon. It's part portfolio, part journal, and a reminder that not every project has to ship code.",
    ],
    tags: ["Photography", "Storytelling"],
    link: "https://atlas-of-my-skies.stimmie.dev",
  },
  {
    slug: "room-tba",
    title: "Room TBA",
    src: "/projects/room-tba.png",
    date: "2025",
    description:
      "An immersive 3D campus viewer and search tool for UPLB — find rooms, dorms, buildings, colleges, and divisions without getting lost on day one.",
    body: [
      "Room TBA is a campus navigation project I built in 2025 for the University of the Philippines Los Baños. Freshies and returning students alike struggle to figure out where their classes actually are — room numbers buried in PDFs, buildings with names that don't match the map in your head.",
      "The site combines a searchable directory of rooms, dorms, buildings, colleges, and divisions with a 3D campus viewer built on Astro. You can look up where you need to be and orient yourself in space before you end up late to class because you took the wrong footpath.",
    ],
    tags: ["3D", "Astro", "Campus"],
    link: "https://room-tba.stimmie.dev",
  },
  {
    slug: "elbi-gradesim",
    title: "Elbi GradeSim",
    src: "/projects/elbi-gradesim.png",
    date: "2025",
    description:
      "A grade simulator and course-planning tool for UPLB students — model your GWA, chase Latin honors, and stress-test schedules before enrollment locks you in.",
    body: [
      "Elbi GradeSim started from a familiar student problem: you won't know your final GWA until grades are in, but you still need to plan next semester and figure out whether that 1-unit elective is worth the risk.",
      "Built in 2025 with Next.js, it lets UPLB students simulate grades, track progress toward Latin honors, and experiment with course loads. It's the kind of nerdy tooling I wish I'd had in my own freshman year.",
    ],
    tags: ["Tooling", "Education", "Next.js"],
    link: "https://gradesim.stimmie.dev",
  },
  {
    slug: "uplb-dsg-website",
    title: "UPLB Data Science Guild Website",
    src: "/projects/uplb-dsg-website.png",
    date: "2025",
    description:
      "The official website for the UPLB Data Science Guild — designed and built as project lead for the org where I help run workshops and community events.",
    body: [
      "I led the design and development of the UPLB Data Science Guild's official website in 2025. DSG runs workshops, study groups, and events for students interested in data science and software — the site needed to reflect that energy without feeling like generic org-page template sludge.",
      "I handled the visual design, information architecture, and implementation so members had a proper home for announcements, resources, and recruitment. It's one of those projects where leadership and craft overlap: you're building for a community you're already part of.",
    ],
    tags: ["Web", "Leadership", "Design"],
    link: null,
  },
];

export function getProjectBySlug(slug) {
  return projects.find((p) => p.slug === slug);
}
