// src/app/resume-workshop/page.js

import Link from "next/link";
import PageShell from "@/components/neo/PageShell";
import WorkshopForm from "./WorkshopForm";

const DESCRIPTION =
  "Extra eyes on your resume, portfolio, or interview prep. Send it over, get a review, pay as you can. 20+ people helped and counting!";

export const metadata = {
  title: "Resume Workshop",
  description: DESCRIPTION,
  alternates: { canonical: "/resume-workshop" },
  openGraph: {
    title: "Stimmie's Resume & Portfolio Workshop",
    description: DESCRIPTION,
    url: "https://stimmie.dev/resume-workshop",
    siteName: "Stimmie",
    locale: "en_US",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "Stimmie's Resume & Portfolio Workshop",
    description: DESCRIPTION,
  },
};

const GUIDE_URL = "https://guide.stimmie.dev/resume";
const KAPE_URL = "https://kape.stimmie.dev";

const TEMPLATES = [
  {
    name: "Overleaf (LaTeX)",
    url: "https://www.overleaf.com/gallery/tagged/cv",
  },
  { name: "Canva", url: "https://www.canva.com/resumes/templates/" },
  {
    name: "Microsoft",
    url: "https://create.microsoft.com/en-us/templates/resumes",
  },
  { name: "Novoresume", url: "https://novoresume.com/resume-templates" },
];

const RESOURCES = [
  {
    name: "STAR Method (r/LifeProTips)",
    url: "https://www.reddit.com/r/LifeProTips/comments/9gctz1/lpt_when_writing_your_resume_use_the_star_method/",
  },
  {
    name: "Google's X-Y-Z Formula (Inc.)",
    url: "https://www.inc.com/bill-murphy-jr/google-recruiters-say-these-5-resume-tips-including-x-y-z-formula-will-improve-your-odds-of-getting-hired-at-google.html",
  },
  {
    name: "Harvard's guide to a strong resume",
    url: "https://careerservices.fas.harvard.edu/resources/create-a-strong-resume/",
  },
  {
    name: "YouTube: how to write a great resume",
    url: "https://www.youtube.com/watch?v=Tt08KmFfIYQ",
  },
  {
    name: "r/EngineeringResumes wiki",
    url: "https://www.reddit.com/r/EngineeringResumes/wiki/index/",
  },
];

const CHANNELS = [
  { label: "☎ book a call", url: "https://cal.stimmie.dev" },
  {
    label: "✉ email me",
    url: "mailto:semariquit@gmail.com?subject=Resume%20Workshop",
  },
  { label: "💬 Messenger ↗", url: "https://m.me/stimmieuwu" },
  { label: "in/stimmie ↗", url: "https://www.linkedin.com/in/stimmie/" },
];

function ExternalList({ items, label }) {
  return (
    <ul
      className="neo-facts neo-role-list list-none p-0 m-0 mt-2 space-y-1"
      aria-label={label}
    >
      {items.map((item) => (
        <li key={item.url}>
          <Link href={item.url} target="_blank" rel="noopener noreferrer">
            {item.name}
          </Link>
        </li>
      ))}
    </ul>
  );
}

function ChannelLinks() {
  return (
    <div className="mt-4 flex flex-wrap gap-3">
      {CHANNELS.map((c) => (
        <Link
          key={c.url}
          href={c.url}
          target={c.url.startsWith("mailto:") ? undefined : "_blank"}
          rel="noopener noreferrer"
          className="neo-link-card inline-block font-bold"
        >
          {c.label}
        </Link>
      ))}
      <span className="neo-link-card inline-block font-bold">
        🎮 pataponz on Discord
      </span>
    </div>
  );
}

export default function ResumeWorkshopPage() {
  return (
    <PageShell
      title="~ resume workshop ~"
      intro="Stimmie's resume and portfolio workshop. Extra eyes on your resume, portfolio, or interview prep. Pay-as-you-can."
      maxWidth="56rem"
    >
      <p>
        Applying for an internship, an org, a scholarship, or a student
        assistant post? You might need some extra eyes on your resume,
        portfolio, or interview prep. That&apos;s what this is for.{" "}
        <strong>20+ people helped and counting!</strong>
      </p>

      <h2 className="neo-section-title mt-8">🛠 how it works</h2>
      <ol className="neo-facts list-decimal pl-6 m-0 mt-2 space-y-2">
        <li>
          Have a resume or portfolio draft ready. Starting from scratch? Grab a
          template from the FAQ below, and{" "}
          <Link href={GUIDE_URL} target="_blank" rel="noopener noreferrer">
            my resume guide
          </Link>{" "}
          is there if you want a head start.
        </li>
        <li>Fill out the form below with a link to it.</li>
        <li>
          I&apos;ll get back to you on your preferred channel: a call, email,
          Messenger, Discord, whatever you&apos;re comfortable with.
        </li>
      </ol>

      <h2 className="neo-section-title mt-8">📨 send me your resume</h2>
      <WorkshopForm />

      <p className="mt-6">
        Prefer to skip the form? Message me directly, that works too:
      </p>
      <ChannelLinks />

      <div
        className="mt-8 p-5"
        style={{ border: "3px double #d6008f", background: "#fff7fb" }}
      >
        <h2 className="neo-section-title">☕ pay-as-you-can</h2>
        <p className="mt-1">
          There&apos;s no fixed price. If the workshop helps you and you can
          spare something,{" "}
          <Link href={KAPE_URL} target="_blank" rel="noopener noreferrer">
            kape.stimmie.dev
          </Link>{" "}
          is where you can chip in. If you can&apos;t, that&apos;s completely
          fine. No one gets turned away.
        </p>
      </div>

      <h2 className="neo-section-title mt-8">❓ faq</h2>

      <h3 className="font-bold mt-4">1. What is a resume?</h3>
      <p className="mt-1">
        A resume is a one-page document showcasing your professional background.
        Here are some templates to get you started:
      </p>
      <ExternalList items={TEMPLATES} label="Resume templates" />

      <h3 className="font-bold mt-4">2. Where do I need a resume?</h3>
      <p className="mt-1">
        You might need a resume or a portfolio for the following:
      </p>
      <ul className="neo-facts neo-role-list list-none p-0 m-0 mt-2 space-y-1">
        <li>✅ Applying as a student assistant</li>
        <li>✅ Joining an organization</li>
        <li>✅ Applying for a side job or internship</li>
        <li>
          ✅ Other college opportunities: exchange programs, workshops, job
          fairs, and more
        </li>
      </ul>

      <h3 className="font-bold mt-4">3. Where can I learn how to make one?</h3>
      <p className="mt-1">
        Start with{" "}
        <Link href={GUIDE_URL} target="_blank" rel="noopener noreferrer">
          my guide
        </Link>
        , then dig into these:
      </p>
      <ExternalList items={RESOURCES} label="Resume writing resources" />

      <h3 className="font-bold mt-4">
        4. Can I just message you directly with a specific question?
      </h3>
      <p className="mt-1">
        Yes, of course! Reach me on{" "}
        <Link
          href="https://www.facebook.com/stimmieuwu/"
          target="_blank"
          rel="noopener noreferrer"
        >
          Facebook
        </Link>
        , Discord (<strong>pataponz</strong>), or{" "}
        <Link
          href="https://www.linkedin.com/in/stimmie/"
          target="_blank"
          rel="noopener noreferrer"
        >
          LinkedIn
        </Link>
        .
      </p>
    </PageShell>
  );
}
