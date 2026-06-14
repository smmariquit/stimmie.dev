// src/app/blog/juneyour/page.js

import Link from "next/link";
import PageShell from "@/components/neo/PageShell";

export const metadata = {
  title: "Parkour",
  description:
    "A full breakdown of what Stimmie achieved this semester: speaking, building, leadership, academics, and life.",
};

const CATEGORIES = {
  life: { label: "Life & Side Quests", emoji: "✨" },
  speaking: { label: "Speaking & Workshops", emoji: "🎤" },
  building: { label: "Building & Shipping", emoji: "🛠️" },
  leadership: { label: "Leadership & Community", emoji: "🤝" },
  academics: { label: "Academics & Research", emoji: "📚" },
};

const achievements = [
  {
    cat: "speaking",
    text: "Hosted Google Developer Groups: Build with AI Manila, in collaboration with Data Engineering Pilipinas",
  },
  {
    cat: "speaking",
    text: "Delivered an ML workshop on Iris classification for our org's applicants",
    link: "https://workshops.stimmie.dev/ml-iris-slides",
    linkLabel: "View slides →",
  },
  {
    cat: "speaking",
    text: "Gave a talk on Python in research at UST",
    link: "https://workshops.stimmie.dev/python",
    linkLabel: "View slides →",
  },
  {
    cat: "speaking",
    text: "Got invited as a guest speaker twice to talk about Minecraft and engineering lessons",
    link: "https://workshops.stimmie.dev/minecraft",
    linkLabel: "Workshop slides →",
    link2: "https://www.youtube.com/watch?v=Ko2HDZOv5PY&t=182s",
    linkLabel2: "Watch the talk →",
  },
  {
    cat: "speaking",
    text: "Guest speaker at Batangas Eastern College, crash course on presenting insights better",
    link: "https://workshops.stimmie.dev/present-insights-better",
    linkLabel: "View slides →",
  },
  {
    cat: "speaking",
    text: "Spent 40+ hours preparing and facilitating UPLB Data Science Guild's Python & R workshop for professionals",
    link: "https://www.facebook.com/share/p/1EBviYF4vK/",
    linkLabel: "See the post →",
  },
  {
    cat: "speaking",
    text: "Held a mock review for CMSC 21 (programming fundamentals)",
    link: "https://workshops.stimmie.dev/cmsc21",
    linkLabel: "View materials →",
  },
  {
    cat: "speaking",
    text: "Went back to my alma mater SMCL to give a workshop on Machine Learning :)",
  },
  {
    cat: "building",
    text: "Built Room TBA, a search engine for rooms, dorms, buildings, colleges, and divisions at UPLB",
    link: "https://room-tba.stimmie.dev",
    linkLabel: "Try Room TBA →",
  },
  {
    cat: "building",
    text: "Created Elbi GradeSim, a GWA calculator and Latin honor simulator for UPLB students",
    link: "https://gradesim.stimmie.dev",
    linkLabel: "Try GradeSim →",
  },
  {
    cat: "building",
    text: "Launched my Minecraft server portfolio, 10+ years of server admin, plugin dev, and community building",
    link: "https://minecraft.stimmie.dev",
    linkLabel: "Explore the portfolio →",
  },
  {
    cat: "building",
    text: "Experimented with Three.js to tell the story of my Minecraft journey, kinda revived the writer in me",
    link: "https://scaffolding.stimmie.dev",
    linkLabel: "Read Scaffolding →",
  },
  {
    cat: "building",
    text: "Contributed to OEIS sequence A322054",
    link: "https://oeis.org/A322054",
    linkLabel: "View sequence →",
  },
  {
    cat: "building",
    text: "Collaborated to make upsked.com/uplb, a class schedule tool for UPLB students",
    link: "https://upsked.com/uplb",
    linkLabel: "Try it →",
  },
  { cat: "building", text: "Did freelance software engineering work" },
  {
    cat: "building",
    text: "Hit 10,000 contributions on GitHub (turns out private repos count, they're just anonymized)",
  },
  {
    cat: "leadership",
    text: "Led one of the biggest projects in an Elbi CS degree, a section of 20, teaching the team advanced Git, JS/React, CI/CD, and Agile",
  },
  { cat: "leadership", text: "Led two meetups for Sip & Scale in UPLB!" },
  {
    cat: "leadership",
    text: "Organized and mentored The Innovation Lab 2026, the inaugural hackathon of my CS org",
  },
  {
    cat: "leadership",
    text: "Mentored and judged the DataCamp × DOST-SEI data hackathon via START-DOST",
  },
  {
    cat: "leadership",
    text: "Partnered Pizza & Friends with GitHub to host a non-serious hackathon and slides party, a break from all the sigma vibes of tech",
    link: "https://joinpizza.fun",
    linkLabel: "Join Pizza & Friends →",
  },
  {
    cat: "leadership",
    text: "Established a student-led Discord server for my Institute",
  },
  {
    cat: "leadership",
    text: "Hosted a Cubao karaoke session with people in tech",
    link: "https://www.facebook.com/share/p/1EFyPpVhpb/",
    linkLabel: "See the chaos →",
  },
  {
    cat: "leadership",
    text: "Served as photographer and mentor in Devcon Laguna's code camp",
  },
  { cat: "leadership", text: "Started a Minecraft server for UPLB Batch 2026" },
  {
    cat: "academics",
    text: "Nominated for one of the best CMSC 173 projects this sem!",
    link: "https://kimiroutes.stimmie.dev",
    linkLabel: "Check out KimiRoutes →",
  },
  {
    cat: "academics",
    text: "Placed 11th out of ~50+ teams nationwide in Algolympics 2026, actually learned C++ and competitive programming",
  },
  {
    cat: "academics",
    text: "Got a proper Codeforces rating",
    link: "https://codeforces.com/profile/manilaboi?graphType=all",
    linkLabel: "See profile →",
  },
  {
    cat: "academics",
    text: "Won UPLB COSS' Web Design Competition (early-sem side quest)",
  },
  {
    cat: "academics",
    text: "Finalist in the Ateneo Blue Case Competition out of 600 participants???",
    link: "https://www.facebook.com/share/p/18upyhm524/",
    linkLabel: "We were shocked too →",
  },
  {
    cat: "academics",
    text: "Attended PythonAsia 2026 with my Data Engineering Pilipinas fam!",
  },
  {
    cat: "academics",
    text: "Tutored via TutorHub and personal tutoring in math, programming, and CS",
  },
  {
    cat: "academics",
    text: "Completed the DOST Scholars' Leadership Camp, a 5-day bootcamp on nation-building, civic duty, and leadership",
  },
  {
    cat: "academics",
    text: "Somehow attended research group sessions of a physics professor, listening to and presenting technical papers on physics-adjacent data analytics",
  },
  { cat: "life", text: "Colored my hair brown" },
  { cat: "life", text: "Got my 5K run time down to 28m 16s" },
  {
    cat: "life",
    text: "Officially visited all 16 cities in Metro Manila + Pateros",
  },
  {
    cat: "life",
    text: "Got the largest library fine in UPLB history (sorry!!!)",
  },
  {
    cat: "life",
    text: "Replaced the LED screen of my friend's phone, guess this can be my backup job",
  },
  { cat: "life", text: "Volunteered for Pahinungod" },
  {
    cat: "life",
    text: "Sat in on random classes out of curiosity, COST 10, ABME 10, CMSC 173, CMSC 191, and more",
  },
  { cat: "life", text: "Did my first half-marathon" },
  {
    cat: "life",
    text: "Somehow hit 5,000 LinkedIn followers despite hating LinkedIn lmao",
  },
  { cat: "life", text: "Laid down drunk in the middle of Lopez Ave" },
  {
    cat: "life",
    text: "Demonstrated to my CS professor how to use Claude Code",
  },
  { cat: "life", text: "Ran Doom on an OrangePi" },
  {
    cat: "life",
    text: "Presented my final project on Computer Networking with a monumental hangover and got a perfect score",
  },
];

function AchievementItem({ item }) {
  return (
    <li>
      {item.text}
      {item.link && (
        <>
          {" "}
          <Link href={item.link} target="_blank" rel="noopener noreferrer">
            {item.linkLabel}
          </Link>
        </>
      )}
      {item.link2 && (
        <>
          {" · "}
          <Link href={item.link2} target="_blank" rel="noopener noreferrer">
            {item.linkLabel2}
          </Link>
        </>
      )}
    </li>
  );
}

export default function JuneYourPage() {
  const grouped = Object.entries(CATEGORIES)
    .map(([key, config]) => ({
      key,
      ...config,
      items: achievements.filter((a) => a.cat === key),
    }))
    .filter((group) => group.items.length > 0);

  return (
    <PageShell title="Parkour 🤸" current="/blog" maxWidth="52rem">
      <p className="mb-4 text-base">
        <Link href="/blog">◄ back to blog</Link>
      </p>

      <article className="neo-prose">
        <p className="text-base neo-muted font-mono">June 1, 2026</p>
        <p>
          If you&apos;re curious about what I achieved this semester (roughly
          from January until now), here&apos;s the full breakdown. Included are
          all the slides, links, and resources so you can hopefully grab the
          same opportunities for yourself.
        </p>

        {grouped.map((group) => (
          <section key={group.key}>
            <h2>
              <span aria-hidden="true">{group.emoji}</span> {group.label}{" "}
              <span className="neo-tag align-middle">{group.items.length}</span>
            </h2>
            <ul>
              {group.items.map((item) => (
                <AchievementItem key={`${item.cat}-${item.text}`} item={item} />
              ))}
            </ul>
          </section>
        ))}
      </article>
    </PageShell>
  );
}
