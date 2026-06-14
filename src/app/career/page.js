// src/app/career/page.js

import Link from "next/link";
import PageShell from "@/components/neo/PageShell";

export const metadata = {
  title: "Career",
  description:
    "Simonee Ezekiel Mariquit (Stimmie): software engineer, community builder, and lifelong tinkerer. Currently open to part-time remote dev work and one-time software consultancy.",
};

const ORGANIZATIONS = [
  { role: "Academic Affairs", org: "Alliance of Computer Science Students, UPLB" },
  { role: "Education & Research", org: "UP Data Science Society" },
  { role: "Internals", org: "UPLB Data Science Guild" },
  {
    role: "Finance",
    org: "UPLB Gavel Club, an affiliate of Toastmasters International",
  },
  { role: "Comp Prog", org: "UP Algorithms Plus Plus" },
  {
    role: "Member",
    org: "UPLB Eliens, a competitive programming group under UPLB",
  },
  {
    role: "Member",
    org: "START - DOST, for DOST-SEI scholars in the field of tech",
  },
  { role: "Volunteer", org: "Data Engineering Pilipinas" },
  { role: "Volunteer", org: "UPLB Ugnayan ng Pahinungod" },
  { role: "Member", org: "ASES Manila" },
];

const INITIATIVES = [
  { role: "Founder", org: "UPLB DX Student Volunteer Group" },
  { role: "Founder", org: "UPLB ICS Students' Discord" },
  {
    role: "Tech & Administration",
    org: "UPLB Batch 2024, 2025, and 2026 Freshies' Servers",
  },
];

function RoleList({ items, label }) {
  return (
    <ul className="neo-facts list-none p-0 m-0 mt-2 space-y-1" aria-label={label}>
      {items.map((item) => (
        <li key={`${item.role}-${item.org}`}>
          <strong>{item.role}</strong> @ {item.org}
        </li>
      ))}
    </ul>
  );
}

export default function CareerPage() {
  return (
    <PageShell
      title="~ my career ~"
      intro="A website version of my resume. Who I am, what I've built, and what I'm up to."
      current="/career"
      maxWidth="56rem"
    >
      <p>
        Hi! My name is <strong>Simonee Ezekiel Mariquit</strong>, though if you
        come from UP, you might know me better as <strong>Stimmie</strong>.
      </p>

      <p className="mt-3">
        Right now, I own and run a Minecraft server called{" "}
        <Link
          href="https://crib.stimmie.dev"
          target="_blank"
          rel="noopener noreferrer"
        >
          The Crib
        </Link>
        . Before that, I was the lead at HearthCraft, a survival Minecraft
        server that ran for six years (including straight through the pandemic)
        and grew into a safe space for over 10,000 players. Running it for that
        long taught me how to keep real infrastructure alive, support a
        community, and stick with something for the long haul. I&apos;m also a
        co-founder of Pizza &amp; Friends, a not-so-serious tech community.
      </p>

      <p className="mt-3">
        On the engineering side, I&apos;m currently doing engineering work at{" "}
        <strong>Navegante</strong> (Jun 2026 to present). Before that I was at{" "}
        <strong>E-Konsulta Medical Clinic</strong> (Apr 2025 to Jun 2026), where
        I built tooling and shipped fixes for software that serves real
        patients.
      </p>

      <p className="mt-3">
        I&apos;ve joined and organized 30+ hackathons, game jams, and design
        challenges. Along the way I&apos;ve met tons of people, heard brilliant
        (and not-so-brilliant) ideas, and gotten to explore new tech. I write
        down what I learn in my{" "}
        <Link
          href="https://guide.stimmie.dev/hackathons"
          target="_blank"
          rel="noopener noreferrer"
        >
          hackathon guide
        </Link>
        . A few of those rooms have been kind to me, too: I&apos;ve won the
        Meralco IDOL Hackathon and the UPLB CPAf Data 2 Decisions challenge,
        among other placements.
      </p>

      <p className="mt-3">
        I spend a good chunk of my time finishing my Bachelor of Science degree
        at the University of the Philippines.
      </p>

      <p className="mt-3">
        The rest of my time goes to my college organizations, volunteering,
        meeting new people, running, watching films, and generally living life.
        Here are the organizations I&apos;m involved with:
      </p>
      <RoleList items={ORGANIZATIONS} label="Organizations" />

      <p className="mt-4">
        And some other initiatives that aren&apos;t necessarily orgs :)
      </p>
      <RoleList items={INITIATIVES} label="Other initiatives" />

      <p className="mt-4">
        I also get invited to give{" "}
        <Link href="/talks">talks and workshops</Link> every now and then.
      </p>

      <p className="mt-3">
        These days I&apos;m especially interested in advanced web development,
        data science, and game development. I also love building stupid projects
        purely for the fun and the memes.
      </p>

      <p className="mt-3">
        If there&apos;s a throughline to all of this, it&apos;s that I like
        building things that help people, whether that&apos;s a server, a clinic
        portal, a community, or a hackathon team, and then sticking around to
        keep them running.
      </p>

      <div
        className="mt-8 p-5"
        style={{ border: "3px double #d6008f", background: "#fff7fb" }}
      >
        <h2 className="neo-section-title">let&apos;s work together</h2>
        <p className="mt-1">I&apos;m currently open to:</p>
        <ul className="neo-facts list-none p-0 m-0 mt-2 space-y-1">
          <li>💻 Part-time, remote, and paid dev work</li>
          <li>🛠 One-time software consultancy</li>
        </ul>
        <div className="mt-4 flex flex-wrap gap-3">
          <Link
            href="https://cv.stimmie.dev"
            target="_blank"
            rel="noopener noreferrer"
            className="neo-link-card inline-block font-bold"
          >
            📄 here&apos;s my CV ↗
          </Link>
          <Link
            href="https://cal.stimmie.dev"
            target="_blank"
            rel="noopener noreferrer"
            className="neo-link-card inline-block font-bold"
          >
            ☎ book a call
          </Link>
          <Link
            href="mailto:semariquit@gmail.com?subject=Work%20Inquiry"
            className="neo-link-card inline-block font-bold"
          >
            ✉ email me
          </Link>
        </div>
      </div>
    </PageShell>
  );
}
