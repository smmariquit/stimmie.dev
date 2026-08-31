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
      { name: "Resume Guide", url: "https://guide.stimmie.dev/resume" },
    ],
  },
  {
    label: "Support",
    blurb: "Ways to support my work",
    links: [{ name: "Kape", url: "https://kape.stimmie.dev" }],
  },
];

// Flattened list kept for legacy consumers (SiteNav, MobileHub).
export const STIMMIEVERSE_LINKS = STIMMIEVERSE_CATEGORIES.flatMap(
  (category) => category.links,
);

// Top-of-page navigation shared by every tab.
export const NAV_LINKS = [
  { label: "home", href: "/" },
  { label: "career", href: "/career" },
  { label: "projects", href: "/projects" },
  { label: "talks", href: "/talks" },
  { label: "blog", href: "/blog" },
  { label: "opportunities", href: "/opportunities" },
  { label: "resume workshop", href: "/resume-workshop" },
  { label: "links", href: "/links" },
  { label: "changelog", href: "/changelog" },
  { label: "archive", href: "/archive" },
  { label: "old bento site", href: "/v2" },
];

// Friends & neighbors on the web. Newest friends go at the top.
export const FRIENDS = [
  {
    name: "Lyra Rivera (LyraPlasma)",
    url: "https://lyraplasma.github.io",
    blurb: "programmer sharing code projects & running a community Discord",
  },
  {
    name: "Imogen Inocentes",
    url: "https://www.imogen.dev",
    blurb: "lead full-stack dev & AI specialist building custom systems",
  },
  {
    name: "Marsh (brickwall2900)",
    url: "https://brickwall2900.github.io",
    blurb: "Java dev who builds stuff for the love of it 😅",
  },
  {
    name: "Ken Ramiscal",
    url: "https://kendan.dev",
    blurb: "web developer building SvelteKit apps & IoT projects",
  },
  {
    name: "Kalinaw Lukas Aom Bebis",
    url: "https://lukasbebis.com",
    blurb: "CS student & fullstack dev exploring SolidJS and systems",
  },
  {
    name: "Keith Fajardo",
    url: "https://keithfajardo.com",
    blurb: "analytics engineer, dbt developer & trainer",
  },
  {
    name: "Arron Parejas",
    url: "https://domodomo.site",
    blurb: "makes free browser tools & AI utilities",
  },
  {
    name: "Eirmon John Paculan",
    url: "https://riecodes.com",
    blurb: "full-stack dev building web, mobile, IoT & computer vision",
  },
  {
    name: "John Carlo Santos",
    url: "https://kuyacarlo.dev",
    blurb: "data engineer, DevOps builder & hackathon competitor",
  },
  {
    name: "Richard Parayno",
    url: "https://richardparayno.com",
    blurb: "HCI researcher, community leader & design engineer",
  },
  {
    name: "Artemio Arcega",
    url: "https://artemiui.vercel.app",
    blurb: "a collection of ideas and pixels",
  },
  {
    name: "Iyel Villanueva Rodriguez",
    url: "https://gilrodriguez.rxrcode.dev/",
    blurb: "Shopify developer & commerce engineer",
  },
  {
    name: "Rovic",
    url: "https://rovicdesign.framer.website/",
    blurb: "GOATed UX guy, first hackathon teammate",
  },
  {
    name: "Adriel Magalona",
    url: "https://www.adrielmagalona.dev/",
    blurb: "fellow hackathoner",
  },
  {
    name: "Cedrick Guevara",
    url: "https://guevix.vercel.app",
    blurb: "fellow software engineer",
  },
  {
    name: "Anne Reyes",
    url: "https://ryeolabs.org",
    blurb: "fellow ambitious girlie",
  },
  {
    name: "Bernard Jezua",
    url: "https://bernardjezua.is-a.dev",
    blurb: "frontend engineer",
  },
  {
    name: "John Yumul",
    url: "https://johnyumul.com",
    blurb: "does product, Pizza & Friends cofounder",
  },
];

export const SECTION_STAR = "✹";

function formatLastUpdated(iso) {
  const date = new Date(iso);
  const month = date.toLocaleString("en-US", {
    month: "long",
    timeZone: "UTC",
  });
  const day = date.getUTCDate();
  const year = date.getUTCFullYear();
  return `${month} ${day}, ${year}`.toLowerCase();
}

export function getMarqueeText() {
  const buildDate =
    process.env.NEXT_PUBLIC_BUILD_DATE ?? new Date().toISOString();
  const updated = formatLastUpdated(buildDate);
  return `${SECTION_STAR} welcome to my corner of the web ${SECTION_STAR} software engineer ${SECTION_STAR} tinkerer ${SECTION_STAR} builder of weird things ${SECTION_STAR} linkedin shitposter ${SECTION_STAR} last updated ${updated} ${SECTION_STAR} `;
}

export const ABOUT_TEXT =
  "Creator, tinkerer, and builder of digital experiences. Whether it's writing code, designing interfaces, or crafting data stories, I love bringing wild ideas to life on the internet.";

export const MOBILE_TAGLINE =
  "Creator, tinkerer, and builder of digital experiences on the internet.";
