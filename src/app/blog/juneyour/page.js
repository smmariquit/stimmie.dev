"use client";
import Link from "next/link";
import MosaicBackground from "@/components/MosaicBackground";

// ── Category config ──────────────────────────────────────────────
const CATEGORIES = {
  life: {
    label: "Life & Side Quests",
    emoji: "✨",
    color: "from-rose-500/20 to-pink-600/10",
    border: "border-rose-500/30",
    badge: "bg-rose-500/20 text-rose-300",
  },
  speaking: {
    label: "Speaking & Workshops",
    emoji: "🎤",
    color: "from-violet-500/20 to-purple-600/10",
    border: "border-violet-500/30",
    badge: "bg-violet-500/20 text-violet-300",
  },
  building: {
    label: "Building & Shipping",
    emoji: "🛠️",
    color: "from-sky-500/20 to-cyan-600/10",
    border: "border-sky-500/30",
    badge: "bg-sky-500/20 text-sky-300",
  },
  leadership: {
    label: "Leadership & Community",
    emoji: "🤝",
    color: "from-amber-500/20 to-orange-600/10",
    border: "border-amber-500/30",
    badge: "bg-amber-500/20 text-amber-300",
  },
  academics: {
    label: "Academics & Research",
    emoji: "📚",
    color: "from-emerald-500/20 to-green-600/10",
    border: "border-emerald-500/30",
    badge: "bg-emerald-500/20 text-emerald-300",
  },
};

// ── Achievements data ────────────────────────────────────────────
const achievements = [
  // Speaking & Workshops
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

  // Building & Shipping
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
  {
    cat: "building",
    text: "Did freelance software engineering work",
  },
  {
    cat: "building",
    text: "Hit 10,000 contributions on GitHub (turns out private repos count, they're just anonymized)",
  },

  // Leadership & Community
  {
    cat: "leadership",
    text: "Led one of the biggest projects in an Elbi CS degree, a section of 20, teaching the team advanced Git, JS/React, CI/CD, and Agile",
  },
  {
    cat: "leadership",
    text: "Led two meetups for Sip & Scale in UPLB!",
  },
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
  {
    cat: "leadership",
    text: "Started a Minecraft server for UPLB Batch 2026",
  },

  // Academics & Research
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

  // Life & Side Quests
  {
    cat: "life",
    text: "Colored my hair brown",
  },
  {
    cat: "life",
    text: "Got my 5K run time down to 28m 16s",
  },
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
  {
    cat: "life",
    text: "Volunteered for Pahinungod",
  },
  {
    cat: "life",
    text: "Sat in on random classes out of curiosity, COST 10, ABME 10, CMSC 173, CMSC 191, and more",
  },
  {
    cat: "life",
    text: "Did my first half-marathon",
  },
  {
    cat: "life",
    text: "Somehow hit 5,000 LinkedIn followers despite hating LinkedIn lmao",
  },
  {
    cat: "life",
    text: "Laid down drunk in the middle of Lopez Ave",
  },
  {
    cat: "life",
    text: "Demonstrated to my CS professor how to use Claude Code",
  },
  {
    cat: "life",
    text: "Ran Doom on an OrangePi",
  },
  {
    cat: "life",
    text: "Presented my final project on Computer Networking with a monumental hangover and got a perfect score",
  },
];

// ── Component ────────────────────────────────────────────────────
function AchievementItem({ item }) {
  return (
    <li className="text-white/95 leading-relaxed">
      {item.text}
      {item.link && (
        <>
          {" "}
          <a
            href={item.link}
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-400 hover:text-blue-300 transition-colors"
          >
            {item.linkLabel}
          </a>
        </>
      )}
      {item.link2 && (
        <>
          {" "}
          <a
            href={item.link2}
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-400 hover:text-blue-300 transition-colors"
          >
            {item.linkLabel2}
          </a>
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
      items: achievements.filter((achievement) => achievement.cat === key),
    }))
    .filter((group) => group.items.length > 0);

  return (
    <div className="min-h-screen bg-black w-full overflow-hidden">
      <MosaicBackground />

      {/* Blog post content overlay */}
      <div
        className="fixed inset-0 flex items-center justify-center pointer-events-none"
        style={{ zIndex: 50 }}
      >
      <div className="fixed inset-0 flex items-center justify-center pointer-events-none" style={{ zIndex: 50 }}>
        <div className="pointer-events-auto bg-gray-800/70 text-white p-6 rounded backdrop-blur-sm max-h-[83vh] max-w-3xl w-full mx-4 overflow-y-auto no-scrollbar">
          <div className="flex flex-col">
            {/* Navigation */}
              <Link href="/blog" className="text-white/70 hover:text-white transition-colors flex items-center gap-2">
                <span>←</span>
                <span>Blog</span>
              </Link>
              <span className="text-white/30">|</span>
              <Link href="/" className="text-white/70 hover:text-white transition-colors">
                Home
              </Link>
            </div>

            {/* Article header */}
            <article>
              <header className="mb-8">
                <h1 className="font-sans text-4xl sm:text-5xl font-black text-white mb-2">
                  Parkour 🤸‍♂️
                </h1>
                <p className="text-white/60">June 1, 2026</p>
              </header>

              {/* Achievements by category */}
                <h2 className="text-2xl font-bold text-white mb-2">
                  The Breakdown
                </h2>
                <p className="text-white/85 mb-8 leading-relaxed">
                  If you&apos;re curious about what I achieved this semester (roughly
                  from January until now), here&apos;s the full breakdown. Included
                  are all the slides, links, and resources so you can hopefully
                  grab the same opportunities for yourself.
                </p>

                <div className="space-y-10">
                  {grouped.map((group) => (
                    <div key={group.key}>
                      <div className="flex items-center gap-2 mb-4">
                        <span className="text-xl">{group.emoji}</span>
                        <h3 className="text-lg font-bold text-white">
                          {group.label}
                        </h3>
                        <span
                          className={`text-xs px-2 py-0.5 rounded-full ${group.badge}`}
                        >
                          {group.items.length}
                        </span>
                      </div>
                      <ul className="list-disc list-inside space-y-2 pl-1">
                        {group.items.map((item) => (
                          <AchievementItem key={`${item.cat}-${item.text}`} item={item} />
                        ))}
                      </ul>
                    </div>
                  ))}
                </div>
            </article>
          </div>
        </div>
      </div>
    </div>
  );
}
