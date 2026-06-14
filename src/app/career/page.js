// src/app/career/page.js

import Link from "next/link";
import PageShell from "@/components/neo/PageShell";

export const metadata = {
  title: "Career",
  description:
    "Simonee Ezekiel Mariquit (Stimmie): software engineer, community builder, teacher, and lifelong tinkerer. Currently open to part-time remote dev work and one-time software consultancy.",
};

const SOFTWARE_ENGINEERING = [
  {
    role: "Software Engineer",
    org: "Navegante",
    url: "https://navegante.app",
    period: "Jun 2026 – present",
  },
  {
    role: "Software Engineer",
    org: "E-Konsulta Medical Clinic",
    url: "https://ekonsultaclinic.ph",
    period: "Apr 2025 – Jun 2026",
    note: "Built tooling and shipped fixes for a TypeScript clinic portal used by real medical staff and patients.",
  },
  {
    role: "Freelance Software & Minecraft Developer",
    org: "Self-employed",
    period: "Nov 2022 – present",
    note: "A bit of everything: POS systems running on Sunmi hardware, educational tools, websites, and full-stack apps, plus Minecraft server setup and plugin work.",
  },
  {
    role: "Technical Support Specialist",
    org: "OrbitNode",
    period: "Oct 2021 – Jan 2022",
    note: "Support for a Minecraft server hosting company.",
  },
];

const TEACHING = [
  {
    role: "Tutor",
    org: "Tutor Hub PH",
    period: "present",
    note: "Math, computer science, and programming.",
  },
  {
    role: "Teaching Partner",
    org: "Aralin",
    period: "Nov 2024 – Jun 2025",
  },
];

const COMMUNITY = [
  {
    role: "Co-founder",
    org: "Pizza & Friends",
    url: "https://joinpizza.fun",
    note: "A not-so-serious tech community.",
  },
  {
    role: "Founder",
    org: "UPLB ICS Students' Discord",
    note: "A home base for Institute of Computer Science students.",
  },
  {
    role: "Tech & Administration",
    org: "UPLB Batch 2024, 2025, and 2026 Freshies' Discords",
  },
  {
    role: "Founder",
    org: "UPLB DX Student Volunteer Group",
  },
  {
    role: "Volunteer",
    org: "Data Engineering Pilipinas",
    url: "https://dataengineering.ph/",
  },
  {
    role: "Volunteer",
    org: "DevCon Laguna",
  },
  {
    role: "Volunteer",
    org: "UPLB Ugnayan ng Pahinungod",
    url: "https://pahinungod.up.edu.ph/",
  },
];

const SCHOOL_ORGS = [
  {
    role: "Academic Affairs",
    org: "Alliance of Computer Science Students, UPLB",
    url: "https://v2.acssuplb.org/",
  },
  {
    role: "Education & Research",
    org: "UP Data Science Society",
    url: "https://www.facebook.com/updatasciencesociety/",
  },
  { role: "Internals", org: "UPLB Data Science Guild" },
  {
    role: "Finance",
    org: "UPLB Gavel Club",
    url: "https://www.toastmasters.org/",
    note: "An affiliate of Toastmasters International.",
  },
  {
    role: "Comp Prog",
    org: "UP Algorithms Plus Plus",
    url: "https://github.com/UP-Algorithm-Plus-Plus",
  },
  {
    role: "Member",
    org: "UPLB Eliens",
    note: "A competitive programming group under UPLB.",
  },
  {
    role: "Member",
    org: "START - DOST",
    url: "https://sei.dost.gov.ph/",
    note: "For DOST-SEI scholars in the field of tech.",
  },
  {
    role: "Member",
    org: "ASES Manila",
    url: "https://www.facebook.com/asesmnl/",
  },
];

function RoleList({ items, label }) {
  return (
    <ul
      className="neo-facts list-none p-0 m-0 mt-2 space-y-2"
      aria-label={label}
    >
      {items.map((item) => (
        <li key={`${item.role}-${item.org}`}>
          <strong>{item.role}</strong> @{" "}
          {item.url ? (
            <Link href={item.url} target="_blank" rel="noopener noreferrer">
              {item.org}
            </Link>
          ) : (
            item.org
          )}
          {item.period ? (
            <span className="neo-muted"> ({item.period})</span>
          ) : null}
          {item.note ? (
            <span className="block text-base neo-muted mt-0.5">{item.note}</span>
          ) : null}
        </li>
      ))}
    </ul>
  );
}

export default function CareerPage() {
  return (
    <PageShell
      title="~ my career ~"
      intro="Never lose your child-like sense of wonder! A website version of my resume: who I am, what I've built, and what I'm up to."
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
        . Before that, I was the lead at{" "}
        <Link
          href="https://hearthcraft.stimmie.dev"
          target="_blank"
          rel="noopener noreferrer"
        >
          HearthCraft
        </Link>
        , a survival Minecraft server that ran for six years (including straight
        through the pandemic) and grew into a safe space for over 10,000
        players. I&apos;ve also joined and organized 30+ hackathons, game jams,
        and design challenges, where I met tons of people, heard brilliant (and
        not-so-brilliant) ideas, and explored new tech. I write down what I
        learn in{" "}
        <Link
          href="https://workshops.stimmie.dev/hackathon-guide"
          target="_blank"
          rel="noopener noreferrer"
        >
          my hackathon guide
        </Link>
        .
      </p>

      <p className="mt-3">
        I spend some of my time finishing my Bachelor of Science degree at the{" "}
        <Link
          href="https://up.edu.ph/"
          target="_blank"
          rel="noopener noreferrer"
        >
          University of the Philippines
        </Link>
        . The rest goes to building things, my organizations, volunteering,
        meeting new people, running, watching films, and generally living life.
      </p>

      <h2 className="neo-section-title mt-8">💻 software engineering</h2>
      <RoleList items={SOFTWARE_ENGINEERING} label="Software engineering roles" />

      <h2 className="neo-section-title mt-8">📚 teaching</h2>
      <RoleList items={TEACHING} label="Teaching roles" />

      <h2 className="neo-section-title mt-8">🤝 community &amp; volunteering</h2>
      <RoleList items={COMMUNITY} label="Community and volunteering" />

      <h2 className="neo-section-title mt-8">🎓 school organizations</h2>
      <RoleList items={SCHOOL_ORGS} label="School organizations" />

      <p className="mt-8">
        I also get invited to give{" "}
        <Link href="/talks">talks and workshops</Link> every now and then.
        Right now I&apos;m especially interested in advanced web development,
        data science, and game development. I also love building stupid projects
        purely for the fun and the memes.
      </p>

      <p className="mt-3">
        <strong>Languages I work with:</strong> HTML &amp; CSS, JavaScript &amp;
        TypeScript, C, C++, SQL, and Python.
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
            href="https://www.linkedin.com/in/stimmie/"
            target="_blank"
            rel="noopener noreferrer"
            className="neo-link-card inline-block font-bold"
          >
            in/stimmie ↗
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
