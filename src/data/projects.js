// src/data/projects.js

// Shared projects data used by the home bento, the /projects listing, and per-project pages.
// Slugs are intentionally explicit (not derived from title) so URLs remain stable
// even if a title is edited. Add new projects at the bottom; never reuse a slug.

export const projects = [
  {
    slug: "hearthcraft",
    title: "HearthCraft",
    src: "/projects/project1.jpg",
    date: "2016 – 2022",
    description:
      "At 13 years old, I decided to take my deep enjoyment of Minecraft, learn to set up a Minecraft server, worked with both managed and bare-metal servers, set up Java plugins and Docker instances, and created a multiplayer experience that served as a safe space for 10,000+ over the span of 6+ years.",
    tags: ["Minecraft", "Docker", "Java", "Community"],
    link: null,
  },
  {
    slug: "atlas-of-my-skies",
    title: "Atlas Of My Skies",
    src: "/projects/project2.jpg",
    date: "2023",
    description: "Telling the story of the skies.",
    tags: ["Photography", "Storytelling"],
    link: null,
  },
  {
    slug: "room-tba",
    title: "Room TBA",
    src: "/projects/room-tba.png",
    date: "2025",
    description:
      "A fully immersive 3D campus viewer for UPLB. Navigate through academic buildings and find your classes easily.",
    tags: ["3D", "Astro", "Campus"],
    link: "https://room-tba.stimmie.dev",
  },
  {
    slug: "elbi-gradesim",
    title: "Elbi GradeSim",
    src: "/projects/project2.jpg",
    date: "2025",
    description:
      "An advanced grade simulation and course planning tool tailored for UPLB students.",
    tags: ["Tooling", "Education", "Next.js"],
    link: "https://gradesim.stimmie.dev",
  },
];

export function getProjectBySlug(slug) {
  return projects.find((p) => p.slug === slug);
}
