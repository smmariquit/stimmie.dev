import Link from "next/link";
import PageShell from "@/components/neo/PageShell";

const desc =
  "A Gemini API key I pushed to GitHub, a $25 budget alert that only warned me, a terminated billing account, and a debt collector still asking for $64.39 a year later.";

export const metadata = {
  title: "How a leaked API key sent me to collections",
  description: desc,
  openGraph: {
    title: "How a leaked API key sent me to collections",
    description: desc,
    url: "https://stimmie.dev/blog/how-a-leaked-key-sent-me-to-collections",
    type: "article",
    images: [{ url: "/blog/default-cover.jpg" }],
  },
  twitter: { card: "summary_large_image", images: ["/blog/default-cover.jpg"] },
};

const TIMELINE = [
  ["May 18, 2025", "Google Cloud free trial starts"],
  ["May 19 to Jun 8, 2025", "Google warns me about publicly visible API keys in four projects"],
  ["Jun to Sep 2025", "Monthly invoices, charged to my card automatically"],
  ["Aug 14 to 15, 2025", "Budget alerts at 50%, 90% and 100% of $25"],
  ["Aug 24, 2025", "Trial billing account closes, credits used up"],
  ["Sep 1, 2025", "Card payment declined for insufficient funds"],
  ["Sep 8, 2025", "Billing account suspended"],
  ["Sep 15, 2025", "Billing account and its projects terminated"],
  ["Oct 2025 to Jan 2026", "Weekly overdue notices from Google Collections"],
  ["Jan 13 to 17, 2026", "Three projects suspended for hijacked resources"],
  ["Jan 28, 2026", "I appeal and admit I pushed the key to GitHub"],
  ["Feb 19, 2026", "Warning that the debt may go to a recovery agency"],
  ["Apr 14, 2026", "First collection notice, $64.39"],
  ["Jun 11, 2026", "One suspended project reinstated"],
  ["Jul 4, 2026", "I open a billing support case"],
  ["Jul 18, 2026", "Google waives $32.19 of it"],
  ["Oct 6, 2026", "The collector still asks for $64.39"],
  ["Oct 8, 2026", "I pay the $32.20 through the Cloud console"],
];

export default function LeakedKeyPage() {
  return (
    <PageShell title="How a leaked API key sent me to collections" current="/blog" maxWidth="52rem">
      <p className="mb-4 text-base">
        <Link href="/blog">◄ back to blog</Link>
      </p>

      <article className="neo-prose">
        <p className="text-base neo-muted font-mono">October 2026</p>

        <p>
          For the past six months I&apos;ve gotten a weekly an email from a debt collector in Buffalo, New York.
          The subject line has my Google Cloud billing account number and the words &quot;Balance Due Reminder&quot;.
          The amount is $64.39.
        </p>

        <p>
          I finally sat down and read every Google email I&apos;d received since late 2024 to figure out how I got
          here. This is that, in order.
        </p>

        <h2>The trial</h2>

        <p>
          In May 2025 I started a Google Cloud free trial. I was building student projects, a travel app, a few
          things with the Gemini API, and Firebase was already pushing me onto the pay-as-you-go plan. A trial
          comes with credits, so it felt free.
        </p>

        <p>
          The next day Google started emailing me about API keys in my projects that were publicly visible. Four
          projects over three weeks. I read those emails the way most people read cookie banners.
        </p>

        <h2>The key</h2>

        <p>
          The short version is that I pushed a Gemini API key to a public GitHub repository, and someone found it
          and used it. I don&apos;t know who. Bots scan GitHub for keys around the clock, and an exposed key can be
          picked up within minutes of a push.
        </p>

        <p>
          In August 2025 Google also disabled a Firebase service-account key I had committed to a different public
          repo. So it wasn&apos;t a one-off. I had a habit of putting secrets in files that went to GitHub.
        </p>

        <h2>The budget alert that only alerted</h2>

        <p>
          I had done one thing right, or so I thought. I had a $25 monthly budget on the account, named
          &quot;Death Alert&quot;. On August 14 and 15, 2025, it fired at 50%, then 90%, then 100%.
        </p>

        <p>
          What I didn&apos;t understand is that a Google Cloud budget doesn&apos;t stop anything. It sends an email.
          Usage keeps running and the bill keeps growing. To actually cut spending off you have to wire the alert to
          something that disables billing, like a Pub/Sub topic and a Cloud Function, and I hadn&apos;t.
        </p>

        <h2>Declined, suspended, terminated</h2>

        <p>
          On August 24 the trial account closed because the credits were gone. On September 1 the card on the paid
          account was declined for insufficient funds. Then it moved fast. Suspended on September 8. Terminated on
          September 15, along with every project linked to it.
        </p>

        <p>
          For the next four months I got a weekly email from Google Collections. None of them listed an amount. In
          January 2026 three of my projects were suspended for &quot;abusive activity consistent with hijacked
          resources&quot;, which is the formal way of saying someone else was using them.
        </p>

        <h2>Appealing</h2>

        <p>
          I appealed. My message to Google was one sentence: I accidentally pushed the key to GitHub and it was used.
          One project came back in June.
        </p>

        <p>
          In February Google warned me the debt might go to a recovery agency. In April it did, to ABC-Amega, which
          later renamed itself Cadex Receivables. The weekly emails started, with an actual number this time, $64.39.
        </p>

        <p>
          In July I opened a billing support case. Support wrote back that they understood I had exposed the key,
          secured the repo, revoked the key and deleted the affected resources, and passed my request to the billing
          team. Six days later billing approved an adjustment of $32.19. Not the full amount. The case closed itself a
          week after that.
        </p>

        <h2>Why the numbers don&apos;t match</h2>

        <p>
          The collector still asks for $64.39. The Cloud console says I owe $32.20. Subtract the $32.19 credit from
          $64.39 and you get $32.20, so the console is right and the agency never picked up the adjustment. Their own
          emails say to pay only through the Cloud console, never through them, so that&apos;s where I paid it. On
          October 8, 2026 I paid the $32.20, a little over a year after the first declined charge.
        </p>

        <h2>The whole thing</h2>

        <table className="my-6 w-full text-sm border-collapse font-mono">
          <thead>
            <tr className="border-b-2 border-current text-left">
              <th className="py-1 pr-3">When</th>
              <th className="py-1">What happened</th>
            </tr>
          </thead>
          <tbody>
            {TIMELINE.map(([when, what]) => (
              <tr key={when} className="border-t border-current">
                <td className="py-1 pr-3 whitespace-nowrap align-top">{when}</td>
                <td className="py-1">{what}</td>
              </tr>
            ))}
          </tbody>
        </table>

        <h2>What I&apos;d tell myself in May 2025</h2>

        <p>
          Put keys in a <code>.env</code> file and put <code>.env</code> in <code>.gitignore</code> before the first
          commit, not after. If a key ever reaches GitHub, deleting the file isn&apos;t enough, because it stays in
          the history. Revoke it and make a new one.
        </p>

        <p>
          Restrict every API key to the APIs it needs. A Gemini key that can only call Gemini still costs money when
          stolen, but less of it.
        </p>

        <p>
          Read the &quot;publicly visible key&quot; emails. Google told me what was wrong four times before it cost
          anything.
        </p>

        <p>
          A budget is an alarm, not a brake. If you want a hard cap, build one, or don&apos;t attach a card you
          can&apos;t afford to see drained.
        </p>

        <p>
          And ask support early. Half of this was waived with one polite email. I waited ten months to send it.
        </p>

        <hr />

        <p className="text-sm">
          Further reading. Google&apos;s guide to{" "}
          <a href="https://cloud.google.com/billing/docs/how-to/budgets">budgets and budget alerts</a>, which says
          plainly that budgets don&apos;t cap usage, and the{" "}
          <a href="https://cloud.google.com/billing/docs/how-to/notify#cap_disable_billing_to_stop_usage">
            example of disabling billing from a budget notification
          </a>
          . GitHub&apos;s page on{" "}
          <a href="https://docs.github.com/en/code-security/secret-scanning/introduction/about-secret-scanning">
            secret scanning
          </a>
          , and Google&apos;s{" "}
          <a href="https://cloud.google.com/docs/authentication/api-keys#securing">best practices for API keys</a>.
        </p>
      </article>
    </PageShell>
  );
}
