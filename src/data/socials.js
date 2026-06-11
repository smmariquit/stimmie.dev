// src/data/socials.js

// Categorised social / profile links rendered in the Hero section of the
// home bento. Keeping this data here (rather than inline in HomeClient.js)
// matches the pattern used by talks.js / projects.js, and makes it trivial
// to add/remove a profile without touching the rendering code.
//
// Each link has:
//   - name:   accessible name (used for the title attribute and aria-label)
//   - href:   destination URL, or null/undefined for display-only entries
//             (e.g. Discord usernames, where there's no profile URL)
//   - icon:   path under /public; PNG and SVG both work
//   - alt:    optional richer alt text; falls back to `name` when absent
//
// Order within a category matters: it's the visual order in the rendered nav.

export const socialCategories = [
  {
    label: "Connect",
    links: [
      {
        name: "LinkedIn",
        href: "https://www.linkedin.com/in/stimmie",
        icon: "/logos/linkedin.png",
        alt: "LinkedIn - Professional networking profile",
      },
      {
        name: "Email",
        href: "mailto:semariquit@gmail.com",
        icon: "/logos/email.png",
      },
      {
        name: "Instagram",
        href: "https://www.instagram.com/friedicecrm",
        icon: "/logos/instagram.png",
      },
      {
        name: "Facebook",
        href: "https://www.facebook.com/stimmieuwu/",
        icon: "/logos/facebook.svg",
      },
      {
        // Discord has no public profile URL; we render the username as a
        // non-clickable badge that exposes it via the tooltip.
        name: "Discord: @pataponz",
        href: null,
        icon: "/logos/discord.svg",
        alt: "Discord - @pataponz",
      },
    ],
  },
  {
    label: "Code",
    links: [
      {
        name: "GitHub",
        href: "https://www.github.com/smmariquit",
        icon: "/logos/github.png",
      },
      {
        name: "LeetCode",
        href: "https://leetcode.com/u/stimmers/",
        icon: "/logos/leetcode.png",
      },
      {
        name: "Kattis",
        href: "https://open.kattis.com/users/simonee",
        icon: "/logos/kattis.png",
        alt: "Kattis - Competitive programming profile",
      },
      {
        name: "Kaggle",
        href: "https://www.kaggle.com/stimmie",
        icon: "/logos/kaggle.svg",
        alt: "Kaggle - Data science competitions and notebooks",
      },
    ],
  },
  {
    label: "Watching · Reading · Listening",
    links: [
      {
        name: "Letterboxd",
        href: "http://letterboxd.com/stimmieuwu",
        icon: "/logos/letterboxd.png",
      },
      {
        name: "IMDb",
        href: "https://www.imdb.com/name/nm12149035/",
        icon: "/logos/imdb.png",
        alt: "IMDb profile",
      },
      {
        name: "MyAnimeList",
        // Was previously pointing at the MAL homepage; the real profile
        // (`amorgosposter`) is already referenced in HomeClient's media
        // section, so re-use that here.
        href: "https://myanimelist.net/profile/amorgosposter",
        icon: "/logos/myanimelist.png",
      },
      {
        name: "Goodreads",
        href: "https://goodreads.com/stimmie",
        icon: "/logos/goodreads.png",
      },
      {
        name: "Spotify",
        href: "https://open.spotify.com/user/opzo90f4votlfqmg9rl94qrra?si=538e1f5748424274",
        icon: "/logos/spotify.png",
      },
      {
        name: "Last.fm",
        href: "https://www.last.fm/user/mistakenpog",
        icon: "/logos/lastfm.png",
      },
    ],
  },
  {
    label: "Writing & Resources",
    links: [
      {
        name: "Medium",
        href: "https://medium.com/@semariquit",
        icon: "/logos/medium.png",
      },
      {
        name: "Pizza and Friends",
        href: "https://pizza-and-friends.webflow.io",
        icon: "/logos/pizza.png",
      },
      {
        name: "Hackathon Guide",
        href: "https://docs.google.com/document/d/1cJya3Zb2ck9vkxIKc1LQjJomQS_LFtBOABlzHrb7Z5s/edit?usp=sharing",
        icon: "/logos/hackathon.png",
      },
      {
        name: "Freshie Resources",
        // NOTE: this URL is currently identical to the Hackathon Guide
        // (same Google Doc ID). Likely a copy-paste mistake — keeping
        // the existing behaviour but flagging for follow-up.
        href: "https://docs.google.com/document/d/1cJya3Zb2ck9vkxIKc1LQjJomQS_LFtBOABlzHrb7Z5s/",
        icon: "/logos/freshie.png",
        alt: "Freshie Resources - A guide for incoming students",
      },
    ],
  },
  {
    label: "Play",
    links: [
      {
        name: "Strava",
        href: "https://www.strava.com/athletes/129023200",
        icon: "/logos/strava.png",
        alt: "Strava - Fitness and running activities",
      },
      {
        name: "osu!",
        href: "https://osu.ppy.sh/users/14900686",
        icon: "/logos/osu.svg",
        alt: "osu! - Rhythm game profile",
      },
    ],
  },
];
