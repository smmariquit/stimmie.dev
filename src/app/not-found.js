import Link from "next/link";
import PageShell from "@/components/neo/PageShell";

export const metadata = {
  title: "You wandered off the map",
};

export default function NotFound() {
  return (
    <PageShell title="~ you wandered off the map ~">
      <div className="text-center py-4">
        <p className="text-6xl mb-4" aria-hidden="true">
          🛸✨
        </p>
        <p className="text-lg mb-2">
          Whoops! There&apos;s nothing here. This page either drifted off into
          the void, never existed, or is hiding from you on purpose.
        </p>
        <p
          className="neo-muted mb-6"
          style={{ fontFamily: "var(--neo-ui)" }}
        >
          Either way, let&apos;s get you back to solid ground.
        </p>
        <ul className="space-y-2 list-none p-0 m-0 inline-block text-left">
          <li>
            <Link href="/">► beam me back home</Link>
          </li>
          <li>
            <Link href="/projects">► poke around my projects</Link>
          </li>
          <li>
            <Link href="/talks">► talks &amp; workshops</Link>
          </li>
          <li>
            <Link href="/blog">► read my writing</Link>
          </li>
          <li>
            <Link href="/links">► see literally everything</Link>
          </li>
        </ul>
      </div>
    </PageShell>
  );
}
