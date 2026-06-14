export const STIMMIEVERSE_CATEGORIES = [
  {
    label: "Services",
    blurb: "Apps, tools, and services I make and offer",
    links: [
      { name: "Room TBA", url: "https://room-tba.stimmie.dev" },
      { name: "GradeSim", url: "https://gradesim.stimmie.dev" },
      { name: "The Crib", url: "https://crib.stimmie.dev" },
      { name: "Tutor", url: "https://tutor.stimmie.dev" },
    ],
  },
  {
    label: "Portfolios",
    blurb: "Showcases of my work across different domains",
    links: [
      { name: "Atlas", url: "https://atlas-of-my-skies.stimmie.dev" },
      { name: "Minecraft", url: "https://minecraft.stimmie.dev" },
      { name: "Web & Mobile", url: "https://web.stimmie.dev" },
      { name: "Data", url: "https://data.stimmie.dev" },
      { name: "Resume", url: "https://resume.stimmie.dev" },
      { name: "CV", url: "https://cv.stimmie.dev" },
    ],
  },
  {
    label: "Guides",
    blurb: "Walkthroughs I wrote to help you out",
    links: [
      { name: "All Guides", url: "https://guide.stimmie.dev" },
      { name: "Freshie Guide", url: "https://guide.stimmie.dev/freshie" },
      { name: "Hackathon Guide", url: "https://guide.stimmie.dev/hackathons" },
    ],
  },
];

// Flattened list kept for legacy consumers (SiteNav, MobileHub).
export const STIMMIEVERSE_LINKS = STIMMIEVERSE_CATEGORIES.flatMap(
  (category) => category.links,
);

// Top-of-page navigation shared by every tab.
export const NAV_LINKS = [
  { label: "home", href: "/" },
  { label: "projects", href: "/projects" },
  { label: "talks", href: "/talks" },
  { label: "blog", href: "/blog" },
  { label: "links", href: "/links" },
  { label: "changelog", href: "/changelog" },
  { label: "archive", href: "/archive" },
  { label: "old bento site", href: "/v2" },
];

// Friends & neighbors on the web. Newest friends go at the top.
export const FRIENDS = [
  { name: "John Yumul", url: "https://johnyumul.com", blurb: "fellow web tinkerer" },
];

export const MARQUEE_TEXT =
  "★ welcome to my corner of the web ★ software engineer ★ tinkerer ★ builder of weird things ★ last updated june 2026 ★ ";

export const ABOUT_TEXT =
  "Creator, tinkerer, and builder of digital experiences. Whether it's writing code, designing interfaces, or crafting data stories, I love bringing wild ideas to life on the internet.";

export const MOBILE_TAGLINE =
  "Creator, tinkerer, and builder of digital experiences on the internet.";
