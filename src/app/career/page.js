// src/app/career/page.js

import Link from "next/link";
import PageShell from "@/components/neo/PageShell";

export const metadata = {
  title: "Career",
  description:
    "Simonee Ezekiel Mariquit (Stimmie): Minecraft dev turned software engineer, community builder, teacher, and certified LinkedIn shitposter. Currently open to part-time remote dev work and one-time software consultancy.",
};

const SOFTWARE_ENGINEERING = [
  {
    role: "Software Engineer",
    org: "Navegante",
    url: "https://navegante.app",
    period: "Jun 2026 – present",
    note: "Remote from Metro Manila.",
  },
  {
    role: "Software Engineer",
    org: "E-Konsulta Medical Clinic",
    url: "https://ekonsultaclinic.ph",
    period: "Apr 2025 – Jan 2026",
    note: "Shipped tooling and web work for a clinic startup. Asked questions, took feedback, and failed fast. Also explored Agile workflows and value-adding initiatives beyond the codebase.",
  },
  {
    role: "Fundraising Lead",
    org: "E-Konsulta Medical Clinic",
    url: "https://ekonsultaclinic.ph",
    period: "Aug 2025 – Sep 2025",
    note: "Short stint securing funding opportunities with government agencies, VCs, and angel investors.",
  },
  {
    role: "Project Lead",
    org: "HearthCraft",
    url: "https://hearthcraft.stimmie.dev",
    period: "Apr 2018 – Apr 2025",
    note: "A non-pay-to-win survival Minecraft server that grew into a safe space for 50,000+ players, with around $5k annual recurring revenue that helped fund college and charitable givebacks.",
  },
  {
    role: "Freelance Software & Minecraft Developer",
    org: "Self-employed",
    period: "Nov 2022 – present",
    note: "Minecraft server setup and plugin work via BuiltByBit, HelpChat, and forums. Lately also MERN, Flutter, Firebase, POS systems on Sunmi hardware, educational tools, and full-stack apps.",
  },
  {
    role: "Minecraft Developer",
    org: "Spark Services",
    period: "Jan 2022 – May 2022",
  },
  {
    role: "Technical Support Specialist",
    org: "OrbitNode",
    period: "Oct 2021 – Dec 2021",
    note: "Remote support for a Minecraft server hosting company.",
  },
];

const UNIVERSITY_WORK = [
  {
    role: "Office Assistant",
    org: "UPLB Institute of Physics",
    period: "Feb 2025 – Apr 2025",
    note: "Officework for a newly formed institute. Learned a bit about the academic side of physics along the way.",
  },
  {
    role: "Office Assistant",
    org: "UPLB Office of Alumni Relations",
    period: "Sep 2024 – Jan 2025",
    note: "Administrative work, data entry, and computer maintenance. Got a closer look at alumni initiatives and how a Philippine government office runs.",
  },
];

const TEACHING = [
  {
    role: "Private Tutor",
    org: "tutor.stimmie.dev",
    url: "https://tutor.stimmie.dev",
    period: "present",
    note: "Math, computer science, and programming through my own tutoring practice.",
  },
  {
    role: "Freelance Tutor",
    org: "Tutor Hub PH",
    period: "Nov 2025 – present",
    note: "Math, computer science, and programming.",
  },
  {
    role: "Teaching Partner",
    org: "Aralin",
    period: "Nov 2024 – Jul 2025",
    note: "Created educational materials and test questions for college entrance test (CET) review on an edtech platform connecting mentors with students.",
  },
];

const COMMUNITY = [
  {
    role: "City Lead",
    org: "Sip & Scale",
    period: "Mar 2026 – present",
    note: "Curated monthly dinners in Los Baños for people in startups and tech to unwind, connect, and talk shop.",
  },
  {
    role: "Founder",
    org: "UX Elbi",
    period: "Dec 2025 – present",
    note: "A community of UX enthusiasts in Los Baños who believe design is a form of public service. Built from scratch.",
  },
  {
    role: "Co-founder",
    org: "Pizza & Friends",
    url: "https://joinpizza.fun",
    period: "Jul 2025 – present",
    note: "A not-so-serious tech community co-founded with John Yumul. Slides parties, hackathons, dinners, Discord calls, and a server home to 300+ members.",
  },
  {
    role: "Volunteer",
    org: "Data Engineering Pilipinas",
    url: "https://dataengineering.ph/",
    period: "Aug 2025 – present",
    note: "Speaking engagements on data science, plus community initiatives in their Discord server.",
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
    period: "May 2025 – present",
    note: "Develop member and public capacity in tech. Keep a steady flow of hackathon and project opportunities for orgmates.",
  },
  {
    role: "Fellow",
    org: "UP Data Science Society",
    url: "https://www.facebook.com/updatasciencesociety/",
    period: "Sep 2025 – present",
  },
  {
    role: "Internal Affairs",
    org: "UPLB Data Science Guild",
    period: "Dec 2024 – present",
    note: "Member records, team building, workshops, and training sessions for members and partner orgs.",
  },
  {
    role: "Executive Officer",
    org: "UPLB Gavel Club",
    url: "https://www.toastmasters.org/",
    period: "Feb 2025 – present",
    note: "An affiliate of Toastmasters International. Accessible public speaking training in UPLB.",
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
    role: "Officer",
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

const EDUCATION = [
  {
    degree: "Bachelor of Science, Computer Science",
    org: "University of the Philippines Los Baños",
    url: "https://up.edu.ph/",
    period: "Aug 2023 – Aug 2027",
    note: "DOST-SEI undergraduate scholar.",
  },
];

const CERTIFICATIONS = [
  "Google AI Essentials (DTI x Google Career Certificates)",
  "DataCamp x DEP Scholarship (50+ courses completed)",
  "Six Sigma: Green Belt",
];

const HONORS = [
  "DOST-SEI Undergraduate Scholar",
  "Champion, Meralco IDOL Hackathon (2025)",
  "Champion, UPLB CPAf Data 2 Decisions (2025)",
];

function RoleList({ items, label }) {
  return (
    <ul
      className="neo-facts neo-role-list list-none p-0 m-0 mt-2 space-y-2"
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

function EducationList({ items }) {
  return (
    <ul
      className="neo-facts neo-role-list list-none p-0 m-0 mt-2 space-y-2"
      aria-label="Education"
    >
      {items.map((item) => (
        <li key={`${item.org}-${item.degree}`}>
          <strong>{item.degree}</strong> @{" "}
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

function SimpleList({ items, label }) {
  return (
    <ul
      className="neo-facts neo-role-list list-none p-0 m-0 mt-2 space-y-1"
      aria-label={label}
    >
      {items.map((item) => (
        <li key={item}>{item}</li>
      ))}
    </ul>
  );
}

export default function CareerPage() {
  return (
    <PageShell
      title="~ my career ~"
      intro="Never lose your child-like sense of wonder! A website version of my resume and LinkedIn: who I am, what I've built, and what I'm up to."
      current="/career"
      maxWidth="56rem"
    >
      <div className="flex flex-wrap gap-3 mb-5">
        <Link
          href="https://www.linkedin.com/in/stimmie/"
          target="_blank"
          rel="noopener noreferrer"
          className="neo-link-card inline-block font-bold"
        >
          in/stimmie on LinkedIn ↗
        </Link>
        <Link
          href="https://github.com/smmariquit"
          target="_blank"
          rel="noopener noreferrer"
          className="neo-link-card inline-block font-bold"
        >
          smmariquit on GitHub ↗
        </Link>
      </div>

      <p>
        Hi! My name is <strong>Simonee Ezekiel Mariquit</strong>, though if you
        come from UP, you might know me better as <strong>Stimmie</strong>.
        Minecraft dev turned software engineer. Certified LinkedIn shitposter.
      </p>

      <p className="mt-3">
        In 2018 I turned my love for Minecraft into{" "}
        <Link
          href="https://hearthcraft.stimmie.dev"
          target="_blank"
          rel="noopener noreferrer"
        >
          HearthCraft
        </Link>
        , a popular non-pay-to-win survival server that became a safe space for
        50,000+ players. At around $5,000 in annual recurring revenue, it helped
        fund much of my college life, gave back to charity, and gradually pushed
        me toward a career in software engineering. These days I run{" "}
        <Link
          href="https://crib.stimmie.dev"
          target="_blank"
          rel="noopener noreferrer"
        >
          The Crib
        </Link>
        .
      </p>

      <p className="mt-3">
        I&apos;ve joined and organized 30+ hackathons, game jams, and design
        challenges. I write down what I learn in{" "}
        <Link
          href="https://workshops.stimmie.dev/hackathon-guide"
          target="_blank"
          rel="noopener noreferrer"
        >
          my hackathon guide
        </Link>
        . I like hearing ideas. If you want to collab on something, send me a
        message.
      </p>

      <p className="mt-3">
        I spend some of my time finishing my BS in Computer Science at{" "}
        <Link
          href="https://up.edu.ph/"
          target="_blank"
          rel="noopener noreferrer"
        >
          UP Los Baños
        </Link>
        . The rest goes to building things, my organizations, volunteering,
        meeting new people, running, watching films, and generally living life.
      </p>

      <h2 className="neo-section-title mt-8">💻 software engineering</h2>
      <RoleList items={SOFTWARE_ENGINEERING} label="Software engineering roles" />

      <h2 className="neo-section-title mt-8">🏛 university work</h2>
      <RoleList items={UNIVERSITY_WORK} label="University work" />

      <h2 className="neo-section-title mt-8">📚 teaching</h2>
      <RoleList items={TEACHING} label="Teaching roles" />

      <h2 className="neo-section-title mt-8">🤝 community &amp; volunteering</h2>
      <RoleList items={COMMUNITY} label="Community and volunteering" />

      <h2 className="neo-section-title mt-8">🏫 school organizations</h2>
      <RoleList items={SCHOOL_ORGS} label="School organizations" />

      <h2 className="neo-section-title mt-8">🎓 education</h2>
      <EducationList items={EDUCATION} />

      <h2 className="neo-section-title mt-8">📜 certifications</h2>
      <SimpleList items={CERTIFICATIONS} label="Certifications" />

      <h2 className="neo-section-title mt-8">🏅 honors &amp; awards</h2>
      <SimpleList items={HONORS} label="Honors and awards" />

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
      <p className="mt-2">
        <strong>Languages I speak:</strong> Filipino and English.
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
        <ul className="neo-facts neo-role-list list-none p-0 m-0 mt-2 space-y-1">
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
