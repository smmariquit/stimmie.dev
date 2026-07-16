import Link from "next/link";

export const metadata = {
  title: "Privacy Policy",
  description: "How stimmie.dev handles analytics and visitor data.",
};

export default function PrivacyPage() {
  return (
    <main className="max-w-2xl mx-auto px-4 py-10 pb-16">
      <Link
        href="/"
        className="text-sm opacity-60 hover:underline mb-8 inline-block"
      >
        ← back home
      </Link>

      <header className="mb-8">
        <p className="text-xs uppercase tracking-widest opacity-50">legal</p>
        <h1 className="text-2xl font-bold mt-1">Privacy Policy</h1>
        <p className="text-sm opacity-60 mt-2">Last updated: July 4, 2026</p>
      </header>

      <article className="space-y-5 text-sm leading-relaxed opacity-85">
        <section>
          <h2 className="text-base font-semibold mb-1">What this site is</h2>
          <p>
            stimmie.dev is a personal portfolio showcasing projects, talks, and
            writing. It is operated by Stimmie as an individual, not a company.
          </p>
        </section>

        <section>
          <h2 className="text-base font-semibold mb-1">What we collect</h2>
          <ul className="list-disc pl-5 space-y-1">
            <li>
              <strong>Vercel Analytics</strong> — anonymous page views and referrers to
              understand traffic. No intentional collection of names or emails through
              analytics.
            </li>
            <li>
              <strong>Visitor counter</strong> — an aggregate visit count stored for
              display on the homepage (not tied to your identity).
            </li>
            <li>
              <strong>Hosting logs</strong> — Vercel may retain standard request logs
              (IP, user agent) for security and operations.
            </li>
          </ul>
        </section>

        <section>
          <h2 className="text-base font-semibold mb-1">What we do not collect</h2>
          <p>
            This site has no user accounts, no contact forms that store submissions on
            a server, and no advertising trackers. External links (GitHub, social
            profiles, project sites) have their own policies.
          </p>
        </section>

        <section>
          <h2 className="text-base font-semibold mb-1">Cookies</h2>
          <p>
            Analytics may use minimal cookies or similar technologies as described in{" "}
            <a
              href="https://vercel.com/docs/analytics/privacy-policy"
              target="_blank"
              rel="noopener noreferrer"
              className="underline"
            >
              Vercel&apos;s documentation
            </a>
            . The site itself does not set login cookies.
          </p>
        </section>

        <section>
          <h2 className="text-base font-semibold mb-1">Changes</h2>
          <p>The date above reflects the latest revision.</p>
        </section>

        <section>
          <h2 className="text-base font-semibold mb-1">Contact</h2>
          <p>
            Questions? Find contact links on the homepage or reach out via GitHub.
          </p>
        </section>
      </article>
    </main>
  );
}
