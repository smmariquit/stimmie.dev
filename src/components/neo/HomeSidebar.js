import Link from "next/link";
import {
  STIMMIEVERSE_CATEGORIES,
} from "@/components/home/constants";
import { SECTION_LINKS } from "@/components/home/sectionLinks";
import CribStatus from "@/components/neo/CribStatus";
import PizzaFriendsPlug from "@/components/neo/PizzaFriendsPlug";
import VisitorCounter from "@/components/neo/VisitorCounter";
import { socialCategories } from "@/data/socials";

function SocialLink({ link }) {
  const needsInvert = link.name === "GitHub" || link.name === "Kattis";
  const isExternal = link.href?.startsWith("http");

  if (!link.href) {
    return (
      <li>
        <span
          className="neo-social-link neo-muted cursor-default"
          title={link.alt || link.name}
        >
          {link.icon && (
            <img
              src={link.icon}
              alt=""
              className={`neo-social-icon ${needsInvert ? "brightness-0" : ""}`}
            />
          )}
          <span>{link.name}</span>
        </span>
      </li>
    );
  }

  return (
    <li>
      <Link
        href={link.href}
        target={isExternal ? "_blank" : undefined}
        rel={isExternal ? "noopener noreferrer" : undefined}
        className="neo-social-link"
        title={link.alt || link.name}
      >
        {link.icon && (
          <img
            src={link.icon}
            alt=""
            className={`neo-social-icon ${needsInvert ? "brightness-0" : ""}`}
          />
        )}
        <span>{link.name}</span>
      </Link>
    </li>
  );
}

function OnThisPageNav() {
  return (
    <nav className="neo-sidebar-box" aria-label="On this page">
      <h2 className="neo-sidebar-heading neo-accent-red">~ on this page ~</h2>
      <ul className="neo-nav-list">
        {SECTION_LINKS.map((item) => (
          <li key={item.href}>
            <a href={item.href}>{item.label}</a>
          </li>
        ))}
      </ul>
    </nav>
  );
}

function SidebarHubContent({ version }) {
  return (
    <>
      <div className="neo-sidebar-box">
        <h2 className="neo-sidebar-heading neo-accent-blue">~ find me ~</h2>
        {socialCategories.map((category) => (
          <div key={category.label}>
            <p className="neo-social-category">{category.label}</p>
            <ul className="neo-social-list">
              {category.links.map((link) => (
                <SocialLink key={link.name} link={link} />
              ))}
            </ul>
          </div>
        ))}
      </div>

      <div className="neo-sidebar-box">
        <h2 className="neo-sidebar-heading neo-accent-purple">
          ~ stimmieverse ~
        </h2>
        {STIMMIEVERSE_CATEGORIES.map((category) => (
          <div key={category.label}>
            <p className="neo-social-category">{category.label}</p>
            {category.blurb && (
              <p className="neo-social-desc">{category.blurb}</p>
            )}
            <ul className="neo-nav-list">
              {category.links.map((site) => (
                <li key={site.name}>
                  <Link
                    href={site.url}
                    target={site.local ? "_self" : "_blank"}
                    rel={site.local ? "" : "noopener noreferrer"}
                  >
                    {site.name}
                    {!site.local && " ↗"}
                  </Link>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      <div className="neo-sidebar-box">
        <CribStatus />
      </div>

      <div className="neo-sidebar-box">
        <PizzaFriendsPlug />
      </div>

      <div className="neo-sidebar-box">
        <VisitorCounter />
      </div>

      <div className="neo-sidebar-box">
        <p
          className="text-xl mb-2 text-center"
          style={{ fontFamily: "var(--neo-pixel)", color: "#1a1a1a" }}
        >
          site version
        </p>
        <p className="text-center text-lg font-bold neo-accent-red">
          <Link href="/changelog">v{version}</Link>
        </p>
      </div>
    </>
  );
}

export function DesktopHomeSidebar({ version }) {
  return (
    <aside className="hidden md:block space-y-3 min-w-0">
      <OnThisPageNav />
      <SidebarHubContent version={version} />
    </aside>
  );
}

export function MobileHomeSidebar({ version }) {
  return (
    <aside className="md:hidden space-y-3 min-w-0">
      <details className="neo-site-hub">
        <summary className="neo-sidebar-box neo-site-hub-summary">
          <span className="neo-sidebar-heading neo-accent-purple m-0">
            ~ site hub ~
          </span>
          <span className="neo-site-hub-hint">
            socials, stimmieverse, communities &amp; more
          </span>
        </summary>
        <div className="neo-site-hub-panel space-y-3 mt-3">
          <SidebarHubContent version={version} />
        </div>
      </details>
    </aside>
  );
}
