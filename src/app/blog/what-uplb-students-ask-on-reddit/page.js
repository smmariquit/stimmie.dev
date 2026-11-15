import Link from "next/link";
import PageShell from "@/components/neo/PageShell";

const desc =
  "I read two years of r/peyups, r/uplb and r/upelbi to see what UPLB students keep asking and which questions get no reply.";

export const metadata = {
  title: "What UPLB students ask on Reddit",
  description: desc,
  openGraph: {
    title: "What UPLB students ask on Reddit",
    description: desc,
    url: "https://stimmie.dev/blog/what-uplb-students-ask-on-reddit",
    type: "article",
    images: [
      {
        url: "/blog/what-uplb-students-ask-on-reddit/cover.png",
        width: 1200,
        height: 630,
        alt: "Dot chart of UPLB posts per month on Reddit, with the title What UPLB students ask on Reddit",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    images: ["/blog/what-uplb-students-ask-on-reddit/cover.png"],
  },
};

const BASE = "/blog/what-uplb-students-ask-on-reddit";

const NO_REPLY = [
  ["r/peyups, about UPLB", "5,463", "30%"],
  ["r/peyups, other campuses", "36,726", "25%"],
  ["r/uplb", "596", "43%"],
  ["r/upelbi", "554", "33%"],
  ["All posts about UPLB", "6,613", "32%"],
];

const COURSES = [
  ["CHEM 18", "University Chemistry", "122"],
  ["MATH 27", "Analytic Geometry and Calculus II", "83"],
  ["CHEM 40", "Basic Organic Chemistry", "70"],
  ["MATH 25", "Fundamental Calculus", "46"],
  ["ECON 11", "General Economics", "43"],
];

function Figure({ src, alt, caption }) {
  return (
    <figure className="my-6">
      <img
        src={src}
        alt={alt}
        className="w-full border-2 border-current"
        loading="lazy"
      />
      <figcaption className="mt-2 text-sm neo-muted font-mono">
        {caption}
      </figcaption>
    </figure>
  );
}

function Table({ head, rows, rightCols = [] }) {
  return (
    <table className="my-6 w-full text-sm border-collapse font-mono">
      <thead>
        <tr className="border-b-2 border-current text-left">
          {head.map((h, i) => (
            <th
              key={h}
              className={`py-1 pr-3 ${rightCols.includes(i) ? "text-right" : ""}`}
            >
              {h}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row[0]} className="border-t border-current">
            {row.map((cell, i) => (
              <td
                key={cell}
                className={`py-1 pr-3 align-top ${rightCols.includes(i) ? "text-right" : ""} ${i === 0 ? "whitespace-nowrap" : ""}`}
              >
                {cell}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default function UplbRedditPage() {
  return (
    <PageShell
      title="What UPLB students ask on Reddit"
      current="/blog"
      maxWidth="52rem"
    >
      <p className="mb-4 text-base">
        <Link href="/blog">◄ back to blog</Link>
      </p>

      <article className="neo-prose">
        <p className="text-base neo-muted font-mono">November 2026</p>

        <p>
          I&apos;m writing the UPLB Freshie Guide, and I wanted its pages to
          answer what students actually ask, not what I assume they ask. Reddit
          is where a lot of those questions end up, so I read two years of it.
        </p>

        <p>
          I pulled every post and comment from r/peyups, r/uplb and r/upelbi
          between 16 November 2024 and 16 November 2026. The data comes from{" "}
          <a href="https://arctic-shift.photon-reddit.com">Arctic Shift</a>, a
          public archive of Reddit with an API. My script asks it for one
          subreddit at a time, oldest first, and saves everything to a file.
          That came to 53,117 posts and 168,707 comments.
        </p>

        <p>
          r/peyups covers every UP campus, so I counted a post as being about
          UPLB when it was on r/uplb or r/upelbi, or when its title or text said
          UPLB, Elbi, Los Baños or LB. That gave 7,353 posts. I dropped deleted
          comments and AutoModerator. About 10% of the UPLB posts had been
          removed or deleted, and 87% of those have no replies, so I left them
          out of every reply number. Otherwise a moderator&apos;s decision would
          look like nobody answering. That leaves 6,613 posts. The method has
          two weak spots. A post that never names the campus gets missed, and
          r/uplb&apos;s first post in the data is from March 2026
          (r/upelbi&apos;s is from July 2025), so most of the UPLB talk is still
          on r/peyups.
        </p>

        <p>
          The first thing I checked was how many posts get no reply at all.
          It&apos;s 32%, or 2,089 of the 6,613 (95% interval 30.5% to 32.7%).
          Most posts get a comment or two. The median is one, and the top 10% of
          posts hold 49% of all the comments. Many of the ten busiest threads
          are stories and opinions, like a thread about how cheap the food in UPLB
          is (95 comments) or one about a deleted UP mail account (67).
        </p>

        <Table
          head={["Where", "Posts", "No reply"]}
          rows={NO_REPLY}
          rightCols={[1, 2]}
        />

        <p>
          On r/peyups, a post about UPLB goes unanswered more often than a post
          about another campus. The gap is five points, and the risk ratio is
          1.20 (95% interval 1.15 to 1.26). The smaller subreddits don&apos;t
          fix that. r/uplb has the highest rate of the four, and its median
          first reply takes 2.1 hours against 0.8 on r/peyups.
        </p>

        <p>
          The topic matters too. I grouped the 7,353 posts into 12 topics with a
          topic model (NMF) and named each group from its top words and sample
          titles. The no-reply rate differs between topics by more than chance
          (chi-square test, p &lt; 0.001), from 39% for manual appeals to 23% for
          UPCAT results. The difference is real but small, since Cramér&apos;s V
          is only 0.08. SLAS and DOST questions come second at 37%.
        </p>

        <Figure
          src={`${BASE}/topics.png`}
          alt="Dot and line chart of the no-reply rate for each of 12 topics, from 39% for manual appeals to 23% for UPCAT results, with 95% intervals"
          caption="No-reply rate by topic. The lines are 95% intervals and the gold line is the 32% average."
        />

        <p>
          Then I asked what goes with getting a reply. I fit a logistic
          regression on the 6,613 posts, where the outcome is whether a post got
          any reply. A title with a question mark has odds of a reply 1.45 times
          higher (95% interval 1.28 to 1.65). A post twice as long has odds 1.33
          times higher (1.27 to 1.39). A post made from May to July has odds 0.85
          times as high (0.76 to 0.94). Time of day, weekends, naming a course
          code and which subreddit the post is on all have intervals that cross
          1, so I can&apos;t tell them apart from no effect. This is correlation.
          It shows what goes with a reply and doesn&apos;t show what causes one.
          A longer post may simply give people more to work with.
        </p>

        <Figure
          src={`${BASE}/replies.png`}
          alt="Chart of the odds ratios from the regression, with a question mark in the title at 1.45 and a post twice as long at 1.33 clearly above 1"
          caption="Odds ratios from the regression. The faded rows have intervals that cross 1."
        />

        <p>
          Replies come fast when they come. Among the 4,524 posts that got a
          reply, half got their first one within 56 minutes. 77% got one within
          6 hours and 91% within 24 hours. After a silent day the odds drop. Of
          all 6,613 posts, 68% got a reply, but 2,491 were still silent after 24
          hours, and only 402 of those (16%) got a comment later.
        </p>

        <p>
          Posts also come in waves. They peak in June, with 824 UPLB posts in
          June 2025 and 519 in June 2026, and the monthly pattern of the two
          years is similar (Spearman&apos;s rho is 0.89 for January to October).
          That&apos;s the season of UPCAT results, appeals and enrollment, and
          it&apos;s the same stretch where the regression gave posts lower odds
          of a reply.
        </p>

        <Figure
          src={`${BASE}/months.png`}
          alt="Dot chart of UPLB posts per month from December 2024 to October 2026, with peaks of 824 in June 2025 and 519 in June 2026"
          caption="UPLB posts per month. One dot is 20 posts."
        />

        <p>
          741 threads name a UPLB course. Chemistry and math take the top four
          spots.
        </p>

        <Table
          head={["Course", "Name", "Threads"]}
          rows={COURSES}
          rightCols={[2]}
        />

        <p>
          Transferring comes up in 1,023 threads and shifting in 844, which is
          more than any other subject I counted by keyword. I used all of this
          to decide what to add to the Freshie Guide for v4.0. It now has pages
          for appeals and waitlists, the enrollment checklist, the medical exam,
          scholarships (DOST and SLAS), shifting, GE courses, the feared
          courses, dorms and mental health.
        </p>

        <p>
          There are limits to all of this. A post with no comments isn&apos;t
          always an unanswered question, since some are stories and some get
          answered in a DM. A comment isn&apos;t always an answer. And Reddit
          isn&apos;t every UPLB student, only the ones who post there. Still,
          6,613 posts is enough to see what people keep asking, and the monthly
          pattern repeats from one year to the next.
        </p>

        <hr />

        <p className="text-sm neo-muted">
          The guide is at{" "}
          <a href="https://guide.stimmie.dev/freshie/">guide.stimmie.dev</a>.
          Its source is on{" "}
          <a href="https://github.com/smmariquit/freshie-guide">GitHub</a>.
        </p>
      </article>
    </PageShell>
  );
}
