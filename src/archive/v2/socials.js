// Frozen v2.0 social icons — logos under public/archive/v2/logos/.

import { ARCHIVE_V2 } from "@/lib/archive-paths";

const icon = (file) => `${ARCHIVE_V2}/logos/${file}`;

export const socialCategories = [
  {
    label: "Connect",
    links: [
      { name: "LinkedIn", href: "https://www.linkedin.com/in/stimmie", icon: icon("linkedin.png") },
      { name: "Email", href: "mailto:semariquit@gmail.com", icon: icon("email.png") },
      { name: "Instagram", href: "https://www.instagram.com/friedicecrm", icon: icon("instagram.png") },
      { name: "Facebook", href: "https://www.facebook.com/stimmieuwu/", icon: icon("facebook.svg") },
      { name: "Discord: @pataponz", href: null, icon: icon("discord.svg"), alt: "Discord - @pataponz" },
    ],
  },
  {
    label: "Code",
    links: [
      { name: "GitHub", href: "https://www.github.com/smmariquit", icon: icon("github.png") },
      { name: "LeetCode", href: "https://leetcode.com/u/stimmers/", icon: icon("leetcode.png") },
      { name: "Kattis", href: "https://open.kattis.com/users/simonee", icon: icon("kattis.png") },
      { name: "Kaggle", href: "https://www.kaggle.com/stimmie", icon: icon("kaggle.svg") },
    ],
  },
  {
    label: "Watching · Reading · Listening",
    links: [
      { name: "Letterboxd", href: "http://letterboxd.com/stimmieuwu", icon: icon("letterboxd.png") },
      { name: "IMDb", href: "https://www.imdb.com/name/nm12149035/", icon: icon("imdb.png") },
      { name: "MyAnimeList", href: "https://myanimelist.net/profile/amorgosposter", icon: icon("myanimelist.png") },
      { name: "Goodreads", href: "https://goodreads.com/stimmie", icon: icon("goodreads.png") },
      { name: "Spotify", href: "https://open.spotify.com/user/opzo90f4votlfqmg9rl94qrra", icon: icon("spotify.png") },
      { name: "Last.fm", href: "https://www.last.fm/user/mistakenpog", icon: icon("lastfm.png") },
    ],
  },
  {
    label: "Writing & Resources",
    links: [
      { name: "Medium", href: "https://medium.com/@semariquit", icon: icon("medium.png") },
      { name: "Pizza and Friends", href: "https://pizza-and-friends.webflow.io", icon: icon("pizza.png") },
      { name: "Hackathon Guide", href: "https://guide.stimmie.dev/hackathons", icon: icon("hackathon.png") },
      { name: "Freshie Resources", href: "https://guide.stimmie.dev/freshie", icon: icon("freshie.png") },
    ],
  },
  {
    label: "Play",
    links: [
      { name: "Strava", href: "https://www.strava.com/athletes/129023200", icon: icon("strava.png") },
      { name: "osu!", href: "https://osu.ppy.sh/users/14900686", icon: icon("osu.svg") },
    ],
  },
];
