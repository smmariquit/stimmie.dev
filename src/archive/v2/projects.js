// Frozen v2.0 projects snapshot — images under public/archive/v2/projects/.

import { ARCHIVE_V2 } from "@/lib/archive-paths";

const p = (file) => `${ARCHIVE_V2}/projects/${file}`;

export const projects = [
  {
    slug: "hearthcraft",
    title: "HearthCraft",
    src: p("project1.jpg"),
    description:
      "At 13 years old, I decided to take my deep enjoyment of Minecraft, learn to set up a Minecraft server, worked with both managed and bare-metal servers, set up Java plugins and Docker instances, and created a multiplayer experience that served as a safe space for 10,000+ over the span of 6+ years.",
    tags: ["Minecraft", "Docker", "Java", "Community"],
  },
  {
    slug: "atlas-of-my-skies",
    title: "Atlas Of My Skies",
    src: p("project2.jpg"),
    description: "Telling the story of the skies.",
    tags: ["Photography", "Storytelling"],
  },
  {
    slug: "barlo-bayani-alert-and-response",
    title: "BARLO: Bayani Alert and Response for Local Operations",
    src: p("project3.jpg"),
    description:
      "Predict a storm's economic impact from typhoon forecast data. Get insights on how to pre-emptively place logistics.",
    tags: ["Data Science", "ML", "Disaster Response"],
  },
  {
    slug: "pharmadash",
    title: "Pharmadash",
    src: p("project4.jpg"),
    description:
      "A hackathon project for efficient pharmaceutical inventory management and distribution.",
    tags: ["Hackathon", "Healthcare", "Inventory"],
  },
  {
    slug: "punnett-square-visualizer",
    title: "Punnett Square Visualizer",
    src: p("project5.jpg"),
    description:
      "An interactive tool to visualize genetic crosses. Used in tutorials for science high school students.",
    tags: ["Education", "Biology", "Visualization"],
  },
  {
    slug: "room-tba",
    title: "Room TBA",
    src: p("project1.jpg"),
    description:
      "A fully immersive 3D campus viewer for UPLB. Navigate through academic buildings and find your classes easily.",
    tags: ["3D", "Astro", "Campus"],
    link: "https://room-tba.stimmie.dev",
  },
  {
    slug: "elbi-gradesim",
    title: "Elbi GradeSim",
    src: p("project2.jpg"),
    description:
      "An advanced grade simulation and course planning tool tailored for UPLB students.",
    tags: ["Tooling", "Education", "Next.js"],
    link: "https://gradesim.stimmie.dev",
  },
  {
    slug: "typhoonomics",
    title: "Typhoonomics",
    src: p("project3.jpg"),
    description:
      "Data-driven insights predicting the economic impact of typhoons in the Philippines.",
    tags: ["Data Science", "Python", "Analytics"],
    link: "https://github.com/smmariquit/typhoonomics",
  },
  {
    slug: "kape",
    title: "Kape",
    src: p("project4.jpg"),
    description:
      "A free, open-source 'Buy Me A Coffee' alternative tailored for local creators.",
    tags: ["Monetization", "Creator Economy"],
    link: "https://kape.stimmie.dev",
  },
];
