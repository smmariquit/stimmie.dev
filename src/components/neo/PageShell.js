import Link from "next/link";
import { NAV_LINKS } from "@/components/home/constants";
import NeoFooter from "./NeoFooter";
import SkipLink from "./SkipLink";

/**
 * Shared Neocities chrome for every inner tab.
 * Provides the skip link, a primary nav landmark, the <main> landmark
 * (focus target for the skip link), and the footer with the visitor counter.
 */
export default function PageShell({
  title,
  intro,
  current,
  children,
  maxWidth = "60rem",
}) {
  return (
    <div className="neo-page" lang="en">
      <SkipLink />
      <div className="neo-shell" style={{ maxWidth }}>
        <header className="neo-box mb-3">
          <Link
            href="/"
            className="neo-title"
            style={{ fontSize: "1.8rem", display: "inline-block" }}
          >
            ~ stimmie.dev ~
          </Link>
          <nav className="neo-topnav mt-2" aria-label="Primary">
            {NAV_LINKS.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                aria-current={current === item.href ? "page" : undefined}
              >
                {item.label}
              </Link>
            ))}
          </nav>
        </header>

        <main id="main-content" tabIndex={-1} className="neo-box min-w-0">
          <h1
            className="neo-title"
            style={{ textAlign: "left", fontSize: "2rem" }}
          >
            {title}
          </h1>
          {intro && (
            <p
              className="neo-muted mt-1 mb-4"
              style={{ fontFamily: "var(--neo-ui)" }}
            >
              {intro}
            </p>
          )}
          {children}
        </main>

        <NeoFooter />
      </div>
    </div>
  );
}
