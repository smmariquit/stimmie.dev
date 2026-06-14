// Frozen v1.0 content — do not import live @/data/* modules.

import { ARCHIVE_V1 } from "@/lib/archive-paths";

const a = (subdir, file) => `${ARCHIVE_V1}/${subdir}/${file}`;

export const v1Talks = [
  {
    title: "UPLB DSG x UPRHS CodeIT Workshop Day 1 - What is Data Science?",
    src: a("talks", "talk1.jpg"),
  },
  {
    title: "UPLB DSG x UPRHS CodeIT Workshop Day 2 - Storytelling with Data",
    src: a("talks", "talk2.jpg"),
  },
  {
    title: "NextStep Hacks 2025 - Winning by Talking",
    src: a("talks", "talk3.jpg"),
  },
  {
    title: "JPCS - QCU Logic Unlocked Day 1 - Machine Learning with Python",
    src: a("talks", "talk4.jpg"),
  },
  {
    title: "UPLB DSG Applicants' Workshop - Data Storytelling with Canva",
    src: a("talks", "talk5.jpg"),
  },
  {
    title:
      "Data Engineering Pilipinas AI Study Group - AI Use Cases That Actually Matter",
    src: a("talks", "talk6.jpg"),
  },
  {
    title: "(upcoming) DLSU ECES - Agile Edge: Swift Project Workflows",
    src: a("talks", "talk7.gif"),
  },
];

export const v1Projects = [
  {
    title: "HearthCraft",
    src: a("projects", "project1.jpg"),
    description:
      "At 13 years old, I decided to take my deep enjoyment of Minecraft, learn to set up a Minecraft server, worked with both managed and bare-metal servers, set up Java plugins and Docker instances, and created a multiplayer experience that served as a safe space for 10,000+ over the span of 6+ years.",
  },
  {
    title: "Atlas Of My Skies",
    src: a("projects", "project2.jpg"),
    description: "Telling the story of the skies.",
  },
  {
    title: "BARLO: Bayani Alert and Response for Local Operations",
    src: a("projects", "project3.jpg"),
    description:
      "Predict a storms economic impact from typhoon forecast data. Get insights on how to pre-emptively place logistics.",
  },
  {
    title: "Pharmadash",
    src: a("projects", "project4.jpg"),
    description:
      "A hackathon project for efficient pharmaceutical inventory management and distribution.",
  },
  {
    title: "Punnett Square Visualizer",
    src: a("projects", "project5.jpg"),
    description:
      "An interactive tool to visualize genetic crosses. Used in tutorials for science high school students",
  },
];

export const v1SocialLinks = [
  { name: "LinkedIn", href: "https://www.linkedin.com/in/stimmie", icon: a("logos", "linkedin.png") },
  { name: "Instagram", href: "https://www.instagram.com/friedicecrm", icon: a("logos", "instagram.png") },
  { name: "GitHub", href: "https://www.github.com/smmariquit", icon: a("logos", "github.png") },
  { name: "Medium", href: "https://medium.com/@semariquit", icon: a("logos", "medium.png") },
  { name: "MyAnimeList", href: "https://myanimelist.net/profile/amorgosposter", icon: a("logos", "myanimelist.png") },
  { name: "Pizza and Friends", href: "https://pizza-and-friends.webflow.io", icon: a("logos", "pizza.png") },
  { name: "Letterboxd", href: "http://letterboxd.com/stimmieuwu", icon: a("logos", "letterboxd.png") },
  { name: "Goodreads", href: "https://goodreads.com/stimmie", icon: a("logos", "goodreads.png") },
  { name: "Spotify", href: "https://open.spotify.com/user/opzo90f4votlfqmg9rl94qrra", icon: a("logos", "spotify.png") },
  { name: "Last.fm", href: "https://www.last.fm/user/mistakenpog", icon: a("logos", "lastfm.png") },
  { name: "Strava", href: "https://www.strava.com/athletes/129023200", icon: a("logos", "strava.png") },
  { name: "Email", href: "mailto:semariquit@gmail.com", icon: a("logos", "email.png") },
];
