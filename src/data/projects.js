// Shared projects data used by the home bento, the /projects listing, and per-project pages.
// Slugs are intentionally explicit (not derived from title) so URLs remain stable
// even if a title is edited. Add new projects at the bottom; never reuse a slug.

export const projects = [
  {
    slug: "hearthcraft",
    title: "HearthCraft",
    src: "/projects/project1.jpg",
    description:
      "At 13 years old, I decided to take my deep enjoyment of Minecraft, learn to set up a Minecraft server, worked with both managed and bare-metal servers, set up Java plugins and Docker instances, and created a multiplayer experience that served as a safe space for 10,000+ over the span of 6+ years.",
    tags: ["Minecraft", "Docker", "Java", "Community"],
    link: null,
  },
  {
    slug: "atlas-of-my-skies",
    title: "Atlas Of My Skies",
    src: "/projects/project2.jpg",
    description: "Telling the story of the skies.",
    tags: ["Photography", "Storytelling"],
    link: null,
  },
  {
    slug: "barlo-bayani-alert-and-response",
    title: "BARLO: Bayani Alert and Response for Local Operations",
    src: "/projects/project3.jpg",
    description:
      "Predict a storm's economic impact from typhoon forecast data. Get insights on how to pre-emptively place logistics.",
    tags: ["Data Science", "ML", "Disaster Response"],
    link: null,
  },
  {
    slug: "pharmadash",
    title: "Pharmadash",
    src: "/projects/project4.jpg",
    description:
      "A hackathon project for efficient pharmaceutical inventory management and distribution.",
    tags: ["Hackathon", "Healthcare", "Inventory"],
    link: null,
  },
  {
    slug: "punnett-square-visualizer",
    title: "Punnett Square Visualizer",
    src: "/projects/project5.jpg",
    description:
      "An interactive tool to visualize genetic crosses. Used in tutorials for science high school students.",
    tags: ["Education", "Biology", "Visualization"],
    link: null,
  },
];

export function getProjectBySlug(slug) {
  return projects.find((p) => p.slug === slug);
}
