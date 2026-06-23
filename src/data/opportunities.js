// src/data/opportunities.js
//
// Curated newsletter issues for /opportunities. Add a new issue object when
// publishing an edition; bump issueNumber; never reuse slugs. Images go in
// public/opportunities/.

const MANILA_TZ = "Asia/Manila";

export const OPPORTUNITY_TYPES = {
  hackathon: { label: "Hackathon", badge: "neo-badge-hackathon" },
  internship: { label: "Internship", badge: "neo-badge-internship" },
  event: { label: "Event", badge: "neo-badge-event" },
  certificate: { label: "Certificate", badge: "neo-badge-certificate" },
  program: { label: "Program", badge: "neo-badge-program" },
};

export const opportunityIssues = [
  {
    slug: "june-2026",
    issueNumber: 1,
    title: "June 2026",
    published: "2026-06-01",
    intro:
      "Hackathons, internships, and learning programs worth a look this month. Deadlines are Manila time unless noted.",
    items: [
      {
        title: "NextStep Hacks 2026",
        type: "hackathon",
        url: "https://joinpizza.fun",
        image: "/opportunities/hackathon.png",
        imageAlt: "Students collaborating at a hackathon",
        org: "Pizza & Friends",
        location: "Metro Manila",
        dates: [
          {
            label: "Registration closes",
            date: "2026-06-25T23:59:00+08:00",
          },
          { label: "Hackathon", date: "2026-07-05", endDate: "2026-07-06" },
        ],
        blurb:
          "Beginner-friendly hackathon with mentors, free food, and a focus on shipping something real in 24 hours.",
      },
      {
        title: "Google Summer of Code",
        type: "internship",
        url: "https://summerofcode.withgoogle.com",
        org: "Google",
        location: "Remote",
        dates: [
          { label: "Contributor applications", date: "2026-04-08" },
          {
            label: "Program period",
            date: "2026-05-26",
            endDate: "2026-08-25",
          },
        ],
        blurb:
          "Paid open-source internship. Pick an org, write a proposal, code all summer.",
      },
      {
        title: "Google Career Certificates",
        type: "certificate",
        url: "https://grow.google/certificates",
        org: "Google",
        location: "Online",
        blurb:
          "Self-paced certs in IT support, data analytics, UX, and more. Scholarships pop up — check the site.",
      },
      {
        title: "UPLB Data Science Guild Apps' Workshop",
        type: "event",
        url: "https://stimmie.dev/talks/ml-workshop-iris-classification",
        image: "/opportunities/program.jpg",
        org: "UPLB Data Science Guild",
        location: "UPLB",
        dates: [{ label: "Workshop day", date: "2026-03-02T09:00:00+08:00" }],
        blurb:
          "Hands-on ML workshop from the guild — good template if you're running something similar on campus.",
      },
    ],
  },
];

export function getIssueBySlug(slug) {
  return opportunityIssues.find((issue) => issue.slug === slug);
}

export function getSortedIssues() {
  return [...opportunityIssues].sort(
    (a, b) => new Date(b.published).getTime() - new Date(a.published).getTime(),
  );
}

export function getOpportunityType(type) {
  return OPPORTUNITY_TYPES[type] ?? {
    label: type,
    badge: "neo-badge-program",
  };
}

export function formatOpportunityDate(date, endDate) {
  const dateOpts = {
    month: "short",
    day: "numeric",
    year: "numeric",
    timeZone: MANILA_TZ,
  };
  const timeOpts = {
    ...dateOpts,
    hour: "numeric",
    minute: "2-digit",
    timeZone: MANILA_TZ,
  };

  const showTime = date.length > 10;
  const start = showTime
    ? new Date(date).toLocaleString("en-US", timeOpts)
    : new Date(date).toLocaleDateString("en-US", dateOpts);

  if (!endDate) {
    return start;
  }

  const end = new Date(endDate).toLocaleDateString("en-US", dateOpts);

  if (showTime) {
    const startDateOnly = new Date(date).toLocaleDateString("en-US", dateOpts);
    return `${startDateOnly} – ${end}`;
  }

  return `${start} – ${end}`;
}

export function isDatePast(date) {
  return new Date(date).getTime() < Date.now();
}
