// src/data/opportunities.js
//
// Curated newsletter issues for /opportunities. Add a new issue object when
// publishing an edition; bump issueNumber; never reuse slugs.
//
// Research pipeline: research/opportunities/ (prompts → responses → here).
// Optional `image` overrides everything. Optional `imageUrl` is used only by
// `npm run opportunities:images` when the apply page blocks bots/screenshots.

import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const MANILA_TZ = "Asia/Manila";
const manifestPath = join(
  dirname(fileURLToPath(import.meta.url)),
  "opportunity-images.json",
);

function loadOpportunityImages() {
  if (!existsSync(manifestPath)) {
    return { images: {} };
  }
  return JSON.parse(readFileSync(manifestPath, "utf8"));
}

export const OPPORTUNITY_TYPE_DEFAULT_IMAGES = {
  hackathon: "/opportunities/defaults/hackathon.png",
  internship: "/opportunities/defaults/internship.png",
  event: "/opportunities/defaults/event.png",
  certificate: "/opportunities/defaults/certificate.svg",
  program: "/opportunities/defaults/program.jpg",
};

export const OPPORTUNITY_TYPES = {
  hackathon: { label: "Hackathon", badge: "neo-badge-hackathon" },
  internship: { label: "Internship", badge: "neo-badge-internship" },
  event: { label: "Event", badge: "neo-badge-event" },
  certificate: { label: "Certificate", badge: "neo-badge-certificate" },
  program: { label: "Program", badge: "neo-badge-program" },
};

/** Display order for grouped issue pages. */
export const OPPORTUNITY_TYPE_ORDER = [
  "hackathon",
  "internship",
  "program",
  "event",
  "certificate",
];

export const opportunityIssues = [
  {
    slug: "q3-2026",
    issueNumber: 2,
    title: "Q3 2026 Ecosystem Report",
    published: "2026-06-24",
    intro:
      "The Philippine opportunity matrix for Q3 2026: scholarships, agentic-AI hackathons, hybrid FMCG internships, ASEAN civic programs, and certs. Deadlines are Manila time unless noted. Verify on the official site before you apply.",
    items: [
      {
        title: "AWS AI & ML Scholars Program",
        type: "certificate",
        url: "https://aws.amazon.com/about-aws/our-impact/scholars/",
        org: "Amazon Web Services / Udacity",
        location: "Online",
        dates: [
          { label: "Application deadline", date: "2026-06-24" },
          {
            label: "Program period",
            date: "2026-08-04",
            endDate: "2026-11-04",
          },
        ],
        blurb:
          "Fully funded generative AI scholarship via Udacity. Amazon Bedrock, Amazon Q — no prior ML experience required.",
      },
      {
        title: "TheFirst Spark 2026: Youth Symposium & Build Sprint",
        type: "hackathon",
        url: "https://www.startupnetworks.co.uk/links/category/26-tenders/",
        org: "Startup Networks / TheFirst Spark",
        location: "Online / Singapore",
        dates: [
          { label: "Submission deadline", date: "2026-06-25" },
          { label: "Symposium day", date: "2026-06-27" },
        ],
        blurb:
          "7-day remote AI build sprint. Top projects pitch live to founders and investors.",
      },
      {
        title: "Girlathon 4.0",
        type: "hackathon",
        url: "https://girlathon26.devfolio.co/",
        org: "GDG on Campus MACE",
        location: "Online",
        dates: [
          { label: "Registration closes", date: "2026-06-25" },
          {
            label: "Development phase",
            date: "2026-07-12",
            endDate: "2026-09-12",
          },
        ],
        blurb:
          "Women-centric hybrid hackathon on AI, IoT, and future mobility. Teams of 2–4, beginner-friendly.",
      },
      {
        title: "Friends of Figma Config Watch Party 2026",
        type: "event",
        url: "https://friends.figma.com/events/details/figma-philippines-presents-figma-slides-party-2026/",
        org: "Friends of Figma PH / UXPH / PWDO",
        location: "Makati / Metro Manila",
        dates: [{ label: "Event", date: "2026-06-26" }],
        blurb:
          "Watch the Figma Config keynote with the local design community. Walk-ins subject to capacity.",
      },
      {
        title: "DDB Saliksik 2026: National Research Conference Plenary",
        type: "event",
        url: "https://ddb.gov.ph/announcement-join-us-live-for-ddb-saliksik-2026/",
        org: "Dangerous Drugs Board",
        location: "Online livestream",
        dates: [{ label: "Event", date: "2026-06-26" }],
        blurb:
          "National plenary on policy, public health, and drug abuse prevention research. Livestream open.",
      },
      {
        title: "AUN Internship Programme 2026 (Batch 3)",
        type: "internship",
        url: "https://www.aunsec.org/",
        org: "ASEAN University Network Secretariat",
        location: "Remote / ASEAN",
        dates: [
          { label: "Application deadline", date: "2026-06-29" },
          {
            label: "Program period",
            date: "2026-09-01",
            endDate: "2026-12-31",
          },
        ],
        blurb:
          "Project management, research, and media production across ASEAN higher education cooperation.",
      },
      {
        title: "Bagong Pilipinas Merit Scholarship Program 2026",
        type: "program",
        url: "https://bpms.ched.gov.ph/",
        imageUrl: "https://www.ched.gov.ph/",
        org: "Commission on Higher Education (CHED)",
        location: "Philippines",
        dates: [{ label: "Application deadline", date: "2026-06-30" }],
        blurb:
          "Up to ₱104,000/semester for high-achieving freshmen in priority courses. GWA 95%+ required.",
      },
      {
        title: "ADB-Japan Scholarship Program",
        type: "program",
        url: "https://www.sih.m.u-tokyo.ac.jp/admission-en/",
        org: "Asian Development Bank / University of Tokyo",
        location: "Japan",
        dates: [{ label: "Application deadline", date: "2026-06-30" }],
        blurb:
          "Fully funded two-year Master's in health, economics, engineering, or public policy.",
      },
      {
        title: "Wireless Engineering Foundations & LTE Networks",
        type: "event",
        url: "https://www.terrahertz.net/",
        org: "Terrahertz",
        location: "Online",
        dates: [{ label: "Event", date: "2026-07-04" }],
        blurb:
          "CPD program on RF fundamentals, LTE drive testing, and GIS mapping for telco pros.",
      },
      {
        title: "FlutterFlow Champions League Hackathon 2026",
        type: "hackathon",
        url: "https://flutterflow-champions-league-hackathon.devfolio.co/",
        org: "The Innovation League / Devfolio",
        location: "Online / Mumbai, India",
        dates: [
          { label: "Registration closes", date: "2026-07-06" },
          { label: "Pitch & event", date: "2026-07-18" },
        ],
        blurb:
          "Low-code hackathon for EdTech, FinTech, or open innovation. Teams of 2–3.",
      },
      {
        title: "YSEALI Summit 2026",
        type: "program",
        url: "https://ph.usembassy.gov/notice-of-funding-opportunity-nofo-young-southeast-asian-leaders-initiative-yseali-summit-2026/",
        imageUrl: "https://www.yseali.state.gov/",
        org: "U.S. Embassy in the Philippines",
        location: "Philippines",
        dates: [
          { label: "Application deadline", date: "2026-07-06" },
          { label: "Program start", date: "2026-09-01" },
        ],
        blurb:
          "Regional youth leadership summit with micro-grants for community action plans.",
      },
      {
        title: "UP Diliman Graduate Programs Admission (1st Sem)",
        type: "program",
        url: "https://science.upd.edu.ph/application-to-graduate-programs-for-2nd-semester-ay-2025-2026/",
        org: "UP Diliman — College of Science",
        location: "Quezon City",
        dates: [{ label: "Application deadline", date: "2026-07-08" }],
        blurb:
          "Graduate programs from data science and math to environmental meteorology.",
      },
      {
        title: "Agentic AI Build Week",
        type: "hackathon",
        url: "https://aabw.genaifund.ai/",
        org: "GenAI Fund",
        location: "Online / Ho Chi Minh City, Vietnam",
        dates: [
          { label: "Event", date: "2026-07-08", endDate: "2026-07-12" },
        ],
        blurb:
          "5-day agentic AI build sprint and workshop series with enterprise partners.",
      },
      {
        title: "Reddit's Games with a Hook Hackathon",
        type: "hackathon",
        url: "https://www.reddit.com/r/Devvit/comments/1u8f6r4/announcing_our_reddits_games_with_a_hook_virtual/",
        imageUrl: "https://developers.reddit.com/docs/",
        org: "Reddit / Phaser",
        location: "Online",
        dates: [
          {
            label: "Event period",
            date: "2026-06-17",
            endDate: "2026-07-15",
          },
          { label: "Submission deadline", date: "2026-07-15" },
        ],
        blurb:
          "Build daily games for Reddit communities with Devvit, React, and Phaser. $40k prize pool.",
      },
      {
        title: "Github Readme Generation Hackathon",
        type: "hackathon",
        url: "https://digitomize.com/hackathons",
        org: "Digitomize",
        location: "Online",
        dates: [{ label: "Application deadline", date: "2026-07-16" }],
        blurb:
          "Virtual sprint on automating and improving open-source repo documentation.",
      },
      {
        title: "FutureForge Hackathon 2026",
        type: "hackathon",
        url: "https://srm-builds4.devfolio.co/hackathons/open",
        org: "Devfolio",
        location: "Online",
        dates: [
          { label: "Application deadline", date: "2026-07-17" },
          { label: "Event start", date: "2026-07-20" },
        ],
        blurb:
          "Buildathon across blockchain, FinTech, and HealthTech infrastructure.",
      },
      {
        title: "Grab Scholarship 2026",
        type: "program",
        url: "https://governmentph.com/grab-scholarship/",
        org: "Grab Philippines",
        location: "Philippines",
        dates: [{ label: "Registration closes", date: "2026-07-17" }],
        blurb:
          "Full tuition, ₱8,000/month allowance, and internship pathways for STEM and business freshmen.",
      },
      {
        title: "Low Earth Orbit Communication Satellites",
        type: "certificate",
        url: "https://academy.itu.int/",
        org: "ITU Academy",
        location: "Online",
        dates: [
          {
            label: "Program period",
            date: "2026-07-20",
            endDate: "2026-07-31",
          },
        ],
        blurb:
          "Instructor-led credential on LEO satellite infrastructure and policy. ~$150.",
      },
      {
        title: "CHED Merit Scholarship Program (CMSP)",
        type: "program",
        url: "https://caraga.ched.gov.ph/ched-scholarship-program-csp/",
        imageUrl: "https://www.ched.gov.ph/",
        org: "Commission on Higher Education (CHED)",
        location: "Philippines",
        dates: [{ label: "Application deadline", date: "2026-07-31" }],
        blurb:
          "State aid for psychology, PT, OT, and social sciences. GWA 93%+, income cap applies.",
      },
      {
        title: "Aurecon Student Engineer Programme",
        type: "internship",
        url: "https://ph.prosple.com/graduate-employers/aurecon-ph/jobs-internships/student-engineer",
        org: "Aurecon Philippines",
        location: "Pasig City",
        dates: [{ label: "Application deadline", date: "2026-07-31" }],
        blurb:
          "12-week paid civil engineering internship in transport, water, and land infrastructure.",
      },
      {
        title: "DTI Hackathon 2026",
        type: "hackathon",
        url: "https://www.startupnetworks.co.uk/links/link/30310-dti-hackathon-2026/",
        imageUrl: "https://www.dti.gov.ph/",
        org: "Department of Trade and Industry",
        location: "Online / Philippines",
        dates: [{ label: "Registration closes", date: "2026-07-31" }],
        blurb:
          "National hackathon for product development tackling Philippine ecosystem needs.",
      },
      {
        title: "ASEAN AI Hackathon 2026 Awarding & Closing Ceremony",
        type: "event",
        url: "https://aifest.ph/ai-hackathon-2026/",
        org: "AI Fest PH",
        location: "Iloilo City",
        dates: [{ label: "Event", date: "2026-08-05" }],
        blurb:
          "Final pitches for top teams building AI for blue economy, tourism, and renewable energy.",
      },
      {
        title: "Unilever HR Internship '26",
        type: "internship",
        url: "https://careers.unilever.com/en/job/rotterdam/hr-internship-26-generalist-and-business-partnering/34155/95252046048",
        imageUrl: "https://www.unilever.com/careers/",
        org: "Unilever Philippines",
        location: "Hybrid",
        dates: [
          {
            label: "Program period",
            date: "2026-08-01",
            endDate: "2026-09-30",
          },
        ],
        blurb:
          "Recruitment, onboarding, employee relations, and HR analytics at a top FMCG.",
      },
      {
        title: "Unilever Customer Strategy & Planning Internship 2026",
        type: "internship",
        url: "https://careers.unilever.com/en/job/cidde/customer-strategy-and-planning-internship-2026/34155/96153764192",
        imageUrl: "https://www.unilever.com/careers/",
        org: "Unilever Philippines",
        location: "Hybrid",
        dates: [
          {
            label: "Program period",
            date: "2026-08-01",
            endDate: "2026-09-30",
          },
        ],
        blurb:
          "Category, channel, and shopper planning with analytics and commercial skills.",
      },
      {
        title: "5G-Advanced Mobile Broadband",
        type: "certificate",
        url: "https://academy.itu.int/",
        org: "ITU Academy",
        location: "Online",
        dates: [
          {
            label: "Program period",
            date: "2026-08-17",
            endDate: "2026-08-24",
          },
        ],
        blurb:
          "Professional course on 5G-Advanced services, tech foundations, and regulation. ~$150.",
      },
      {
        title: "Level Up Your Prototypes in Figma",
        type: "event",
        url: "https://friends.figma.com/events/details/figma-philippines-presents-level-up-your-prototypes-in-figma/",
        org: "Friends of Figma / UX Davao",
        location: "Davao City",
        dates: [{ label: "Event", date: "2026-08-24" }],
        blurb:
          "Advanced Figma prototyping for digital and physical interfaces. Mindanao UX community.",
      },
      {
        title: "MARS Internship Program (September 2026 Intake)",
        type: "internship",
        url: "https://ph.talent.com/view?id=619434479436246943",
        org: "Mars Philippines",
        location: "Taguig City / Hybrid",
        dates: [{ label: "Program start", date: "2026-09-01" }],
        blurb:
          "4-month roles in sales, marketing, supply chain, design, or finance. Rolling intake.",
      },
      {
        title: "L'Oréal PH Management Trainee (September Intake)",
        type: "internship",
        url: "https://careers.loreal.com/",
        imageUrl: "https://www.loreal.com/en/philippines/",
        org: "L'Oréal Philippines",
        location: "Pasig",
        dates: [{ label: "Program start", date: "2026-09-01" }],
        blurb:
          "Leadership pipeline in data, marketing, and commercial strategy. Fresh grads welcome.",
      },
      {
        title: "DSU DevHack 3.0",
        type: "hackathon",
        url: "https://www.startupgrantsindia.com/competitions/dsu-devhack-30",
        org: "Dayananda Sagar University / Devfolio",
        location: "Online / India",
        dates: [
          { label: "Event", date: "2026-09-18", endDate: "2026-09-19" },
        ],
        blurb:
          "36-hour arena for production-ready AI apps and hardware-software integrations.",
      },
      {
        title: "MunichTech EXPO AI Innovation Hackathon 2026",
        type: "hackathon",
        url: "https://dorahacks.io/hackathon/2019",
        imageUrl: "https://dorahacks.io/hackathon/2019",
        org: "MunichTech EXPO",
        location: "Hybrid / Online",
        dates: [{ label: "Submission deadline", date: "2026-09-20" }],
        blurb:
          "Applied AI for industrial automation, mobility, and enterprise — corporate pilot tracks.",
      },
      {
        title: "IATSS Forum Leadership Development Program 2027",
        type: "program",
        url: "https://www.iatssforum.jp/en/applications/",
        org: "International Association of Traffic and Safety Sciences",
        location: "Japan",
        dates: [{ label: "Application deadline", date: "2026-09-30" }],
        blurb:
          "8-week fully funded leadership training in Japan on urban design and disaster resilience.",
      },
      {
        title: "LUXASIA 2026 Internships",
        type: "internship",
        url: "https://www.luxasia.com/",
        imageUrl: "https://www.luxasia.com/",
        org: "LUXASIA Philippines",
        location: "Taguig / Alabang",
        dates: [
          {
            label: "Program period",
            date: "2026-07-01",
            endDate: "2026-12-31",
          },
        ],
        blurb:
          "E-commerce, data analytics, digital graphics, and sales roles for luxury brands. Rolling applications.",
      },
      {
        title: "Phoenix Petroleum Operational Excellence Intern",
        type: "internship",
        url: "https://www.phoenixfuels.ph/careers",
        imageUrl: "https://www.phoenixfuels.ph/",
        org: "Phoenix Petroleum Philippines",
        location: "Taguig",
        dates: [{ label: "Program start", date: "2026-07-01" }],
        blurb:
          "Supply chain and commercial process improvement. Rolling applications.",
      },
      {
        title: "Monde Nissin MondeXplore Internship Program",
        type: "internship",
        url: "https://www.mondenissin.com/careers/",
        imageUrl: "https://www.mondenissin.com/",
        org: "Monde Nissin Corporation",
        location: "Nationwide",
        blurb:
          "FMCG exposure in business, engineering, and product dev. Paid (~₱400–450/day). Rolling applications.",
      },
      {
        title: "Globe Telecom Technology & Innovation Internship",
        type: "internship",
        url: "https://www.globe.com.ph/about-us/careers.html",
        imageUrl: "https://www.globe.com.ph/student-program",
        org: "Globe Telecom",
        location: "Hybrid / Taguig / Mandaluyong",
        blurb:
          "Digital solutions across Globe's tech ecosystem. IT, CS, or business undergrads. Rolling applications.",
      },
      {
        title: "Google Career Certificates via DICT",
        type: "certificate",
        url: "https://dict.gov.ph/dict-opens-access-to-free-google-career-certificates",
        imageUrl: "https://grow.google/intl/en_ph/certificates/",
        org: "Google / DICT / Globe",
        location: "Online",
        blurb:
          "Free scholarships for data analytics, UX design, project management, and IT support certs.",
      },
      {
        title: "UP Open University Free MOOCs (MODeL)",
        type: "certificate",
        url: "https://model.upou.edu.ph/",
        org: "University of the Philippines Open University",
        location: "Online",
        blurb:
          "Self-paced free courses with e-certificates — disaster resilience, data annotation, entrepreneurship, and more.",
      },
    ],
  },
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

export function slugifyOpportunityTitle(title) {
  return title
    .toLowerCase()
    .normalize("NFD")
    .replace(/\p{M}/gu, "")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 48);
}

/** Stable id for image filenames and manifest keys. */
export function getOpportunityId(issueSlug, item) {
  if (item.id) {
    return item.id;
  }
  return `${issueSlug}-${slugifyOpportunityTitle(item.title)}`;
}

/** Manual `image` wins, then fetched og:image, then type default. */
export function resolveOpportunityImage(issueSlug, item) {
  if (item.image) {
    return item.image;
  }

  const id = getOpportunityId(issueSlug, item);
  const entry = loadOpportunityImages().images[id];
  if (
    (entry?.status === "fetched" || entry?.status === "screenshot") &&
    entry.path
  ) {
    return entry.path;
  }

  return (
    OPPORTUNITY_TYPE_DEFAULT_IMAGES[item.type] ??
    OPPORTUNITY_TYPE_DEFAULT_IMAGES.program
  );
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

export function groupIssueItemsByType(items) {
  const groups = new Map();
  for (const item of items) {
    if (!groups.has(item.type)) {
      groups.set(item.type, []);
    }
    groups.get(item.type).push(item);
  }

  return OPPORTUNITY_TYPE_ORDER.filter((type) => groups.has(type)).map((type) => ({
    type,
    items: groups.get(type),
  }));
}

/** Nearest upcoming date, or first listed if all are past. */
export function getPrimaryOpportunityDate(dates) {
  if (!dates?.length) {
    return null;
  }
  const upcoming = dates.find((entry) => !isDatePast(entry.date));
  return upcoming ?? dates[0];
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
