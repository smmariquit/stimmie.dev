import Link from "next/link";

export const metadata = {
  title: "Terms of Service",
  description: "Terms for using stimmie.dev.",
};

export default function TermsPage() {
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
        <h1 className="text-2xl font-bold mt-1">Terms of Service</h1>
        <p className="text-sm opacity-60 mt-2">Last updated: July 4, 2026</p>
      </header>

      <article className="space-y-5 text-sm leading-relaxed opacity-85">
        <section>
          <h2 className="text-base font-semibold mb-1">Use of the site</h2>
          <p>
            By browsing stimmie.dev, you agree to use it lawfully and respectfully.
            Content (text, images, project descriptions) is for personal viewing unless
            otherwise noted.
          </p>
        </section>

        <section>
          <h2 className="text-base font-semibold mb-1">Content</h2>
          <p>
            Portfolio materials are provided &quot;as is.&quot; Project links may point
            to third-party sites with their own terms. Opinions expressed are personal
            and not those of any employer or institution unless stated.
          </p>
        </section>

        <section>
          <h2 className="text-base font-semibold mb-1">No warranties</h2>
          <p>
            The site is offered without warranties. It may change, break, or go offline
            without notice.
          </p>
        </section>

        <section>
          <h2 className="text-base font-semibold mb-1">Limitation of liability</h2>
          <p>
            To the extent permitted by law, the site operator is not liable for damages
            arising from use of this website or linked external projects.
          </p>
        </section>

        <section>
          <h2 className="text-base font-semibold mb-1">Changes</h2>
          <p>These Terms may be updated occasionally. Check the date above.</p>
        </section>
      </article>
    </main>
  );
}
