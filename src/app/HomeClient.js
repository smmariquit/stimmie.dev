// src/app/HomeClient.js

"use client";

import Image from "next/image";
import Link from "next/link";
import { GitHubCalendar } from "react-github-calendar";
import {
  FRIENDS,
  MARQUEE_TEXT,
  NAV_LINKS,
  STIMMIEVERSE_CATEGORIES,
} from "@/components/home/constants";
import CribStatus from "@/components/neo/CribStatus";
import SkipLink from "@/components/neo/SkipLink";
import VisitorCounter from "@/components/neo/VisitorCounter";
import { blogPosts } from "@/data/blogs";
import { projects } from "@/data/projects";
import { socialCategories } from "@/data/socials";
import { talks } from "@/data/talks";
import { body, display } from "./fonts";

function StarDivider() {
  return <p className="neo-divider">✧･ﾟ: *✧･ﾟ:* ✧･ﾟ: *✧･ﾟ:* ✧･ﾟ: *✧･ﾟ:*</p>;
}

function NeoSection({ title, children, id }) {
  return (
    <section id={id} className="mb-4">
      <h2 className="neo-section-title">{title}</h2>
      {children}
    </section>
  );
}

// In-page section anchors for the sidebar "on this page" menu.
const SECTION_LINKS = [
  { label: "about me", href: "#about" },
  { label: "projects", href: "#projects" },
  { label: "talks", href: "#talks" },
  { label: "writing", href: "#writing" },
  { label: "currently into", href: "#currently" },
  { label: "friends", href: "#friends" },
  { label: "book a call", href: "#book" },
  { label: "support me", href: "#support" },
];

function getSortedTalks() {
  return [...talks].sort(
    (a, b) => new Date(b.date).getTime() - new Date(a.date).getTime(),
  );
}

// Fixed time zone keeps the rendered date identical on server and client
// (the talk dates are authored in +08:00), avoiding hydration mismatches.
function formatTalkDate(date) {
  return new Date(date).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    timeZone: "Asia/Manila",
  });
}

function SocialLink({ link }) {
  const needsInvert = link.name === "GitHub" || link.name === "Kattis";

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
        target="_blank"
        rel="noopener noreferrer"
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

export default function HomeClient({ mediaData, version }) {
  const { film, book, music } = mediaData || {};
  const sortedTalks = getSortedTalks();
  const HOME_PROJECT_LIMIT = 4;
  const featuredProjects = projects.slice(0, HOME_PROJECT_LIMIT);
  const remainingProjects = Math.max(0, projects.length - HOME_PROJECT_LIMIT);
  const HOME_TALK_LIMIT = 4;
  const remainingTalks = Math.max(0, sortedTalks.length - HOME_TALK_LIMIT);

  return (
    <div className={`neo-page ${display.variable} ${body.variable}`} lang="en">
      <SkipLink />
      <div className="neo-shell">
        {/* Header */}
        <header className="neo-box mb-3 text-center">
          <h1 className="neo-title">~* stimmie&apos;s homepage *~</h1>
          <p
            className="mt-2 text-lg neo-muted"
            style={{ fontFamily: "var(--neo-ui)" }}
          >
            Hi! You found my website on the{" "}
            <strong className="text-[#cc0000]">internet</strong>.
          </p>
          <nav
            className="neo-topnav mt-3"
            style={{ justifyContent: "center" }}
            aria-label="Primary"
          >
            {NAV_LINKS.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                aria-current={item.href === "/" ? "page" : undefined}
              >
                {item.label}
              </Link>
            ))}
          </nav>
        </header>

        {/* Marquee */}
        <div className="neo-marquee-wrap mb-3" aria-hidden="true">
          <div className="neo-marquee-track">
            {MARQUEE_TEXT}
            {MARQUEE_TEXT}
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-[14rem_1fr] gap-3">
          {/* Sidebar */}
          <aside className="space-y-3">
            <nav className="neo-sidebar-box" aria-label="On this page">
              <h2 className="neo-sidebar-heading neo-accent-red">
                ~ on this page ~
              </h2>
              <ul className="neo-nav-list">
                {SECTION_LINKS.map((item) => (
                  <li key={item.href}>
                    <a href={item.href}>{item.label}</a>
                  </li>
                ))}
              </ul>
            </nav>

            <div className="neo-sidebar-box">
              <h2 className="neo-sidebar-heading neo-accent-blue">
                ~ find me ~
              </h2>
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
          </aside>

          {/* Main content */}
          <main id="main-content" tabIndex={-1} className="neo-box min-w-0">
            <NeoSection title="about me" id="about">
              <p>
                I&apos;m <strong>Stimmie</strong>, a creator, tinkerer, and
                builder of digital experiences. Whether it&apos;s writing code,
                designing interfaces, or crafting data stories, I love bringing
                wild ideas to life on the internet.
              </p>
              <p className="mt-3">
                Away from the keyboard I&apos;m usually out exploring Metro
                Manila, lifting at the gym, taking photowalks, or watching and
                over-analyzing films. I&apos;m always happy to meet new people,
                so don&apos;t be a stranger!
              </p>
              <p className="mt-3 font-bold">~ fun facts ~</p>
              <ul className="neo-facts list-none p-0 m-0 mt-1 space-y-1">
                <li>
                  💻 I built the PC I still use today, by myself, when I was 16.
                </li>
                <li>👒 Fedora Linux is my daily driver.</li>
                <li>
                  🚌 I&apos;ve alighted at and explored all 51 LRT and MRT
                  stations in Metro Manila.
                </li>
                <li>🏋️ Gym regular (SBD: 100 / 60 / 150).</li>
                <li>📃 Big on photowalks and writing.</li>
                <li>
                  🐛 My first ever technical failure: I left a VM on its default
                  password and it got drafted into a botnet attacking a Russian
                  IP.
                </li>
                <li>
                  🎬 I watch and analyze films, and yes, I have an{" "}
                  <Link
                    href="https://www.imdb.com/name/nm12149035/"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    IMDb page
                  </Link>
                  .
                </li>
              </ul>
            </NeoSection>

            <StarDivider />

            <NeoSection title="my projects" id="projects">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {featuredProjects.map((p) => (
                  <Link
                    key={p.slug}
                    href={`/projects/${p.slug}`}
                    className="neo-media-card group block"
                  >
                    <Image
                      src={p.src}
                      alt={p.title}
                      width={800}
                      height={450}
                      quality={90}
                      sizes="(max-width: 640px) 100vw, 360px"
                      className="neo-thumb-lg w-full aspect-video object-cover"
                    />
                    <h3 className="font-bold mt-2 group-hover:text-[#cc0066]">
                      {p.title}
                    </h3>
                    {p.date && (
                      <p className="text-base neo-muted mt-0.5 font-mono">
                        {p.date}
                      </p>
                    )}
                    {p.tags && (
                      <p className="text-base neo-muted mt-0.5">
                        [{p.tags.join(" · ")}]
                      </p>
                    )}
                  </Link>
                ))}
              </div>
              <p className="mt-2 text-base">
                <Link href="/projects">
                  ►{" "}
                  {remainingProjects > 0
                    ? `see ${remainingProjects} more project${remainingProjects === 1 ? "" : "s"}`
                    : "see all projects"}
                </Link>
              </p>
            </NeoSection>

            <StarDivider />

            <NeoSection title="talks & workshops" id="talks">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {sortedTalks.slice(0, HOME_TALK_LIMIT).map((t) => {
                  const displayImage = t.actionPhoto || t.slidesThumbnail;
                  return (
                    <Link
                      key={t.slug}
                      href={`/talks/${t.slug}`}
                      className="neo-media-card group block"
                    >
                      {displayImage && (
                        <Image
                          src={displayImage}
                          alt={`Photo from ${t.title}`}
                          width={800}
                          height={450}
                          quality={90}
                          sizes="(max-width: 640px) 100vw, 360px"
                          className="neo-thumb-lg w-full aspect-video object-cover"
                        />
                      )}
                      <h3 className="font-bold mt-2 group-hover:text-[#cc0066]">
                        {t.title}
                      </h3>
                      <p className="text-base neo-muted mt-0.5 font-mono">
                        {formatTalkDate(t.date)}
                      </p>
                      <p className="text-base neo-muted mt-0.5">
                        {t.type} · {t.event}
                      </p>
                    </Link>
                  );
                })}
              </div>
              <p className="mt-2 text-base">
                <Link href="/talks">
                  ►{" "}
                  {remainingTalks > 0
                    ? `see ${remainingTalks} more talk${remainingTalks === 1 ? "" : "s"}`
                    : "see all talks"}
                </Link>
              </p>
            </NeoSection>

            <StarDivider />

            <NeoSection title="writing" id="writing">
              <ul className="space-y-3 list-none p-0 m-0">
                {blogPosts.map((post) => (
                  <li key={post.slug}>
                    <span className="text-base neo-muted font-mono">
                      {post.date}
                    </span>
                    {" · "}
                    <Link href={`/blog/${post.slug}`}>{post.title}</Link>
                  </li>
                ))}
              </ul>
            </NeoSection>

            <StarDivider />

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <NeoSection title="github activity">
                <div className="overflow-x-auto border-2 inset border-gray-400 bg-white p-2">
                  <GitHubCalendar
                    username="smmariquit"
                    colorScheme="light"
                    blockSize={11}
                    blockMargin={3}
                    fontSize={13}
                  />
                </div>
              </NeoSection>

              <NeoSection title="currently into..." id="currently">
                <ul className="space-y-4 list-none p-0 m-0">
                  {film && (
                    <li className="flex gap-4 items-start">
                      {film.posterUrl && (
                        <img
                          src={film.posterUrl}
                          alt={film.title}
                          className="neo-thumb-lg w-28 h-40 object-cover flex-shrink-0"
                        />
                      )}
                      <div className="pt-1">
                        <strong>film:</strong> {film.title}
                        {film.rating && (
                          <p className="neo-muted mt-1">
                            {"★".repeat(Math.round(film.rating))}
                          </p>
                        )}
                      </div>
                    </li>
                  )}
                  {book && (
                    <li className="flex gap-4 items-start">
                      {book.coverUrl && (
                        <img
                          src={book.coverUrl}
                          alt={book.title}
                          className="neo-thumb-lg w-28 h-40 object-cover flex-shrink-0"
                        />
                      )}
                      <div className="pt-1">
                        <strong>book:</strong> {book.title}
                        <p className="neo-muted mt-1">{book.author}</p>
                      </div>
                    </li>
                  )}
                  {music && (
                    <li>
                      <div className="pt-1">
                        <strong>music:</strong> top albums ·{" "}
                        <Link
                          href={music.profileUrl}
                          target="_blank"
                          rel="noopener noreferrer"
                        >
                          last.fm
                        </Link>
                        <p className="neo-muted mt-1 mb-2">{music.period}</p>
                      </div>
                      <Link
                        href={music.profileUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="block"
                      >
                        <img
                          src={music.collageUrl}
                          alt={`${music.username}'s top albums on Last.fm over the ${music.period.toLowerCase()}`}
                          width={260}
                          height={260}
                          loading="lazy"
                          className="neo-thumb-lg w-full max-w-[260px] aspect-square object-cover"
                        />
                      </Link>
                    </li>
                  )}
                </ul>
              </NeoSection>
            </div>

            <StarDivider />

            <NeoSection title="friends" id="friends">
              <p className="mb-2 text-base neo-muted">
                cool people &amp; neighbors from around the web:
              </p>
              <ul className="neo-friends-list">
                {FRIENDS.map((friend) => (
                  <li key={friend.url}>
                    <Link
                      href={friend.url}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      {friend.name} ↗
                    </Link>
                    {friend.blurb && (
                      <span className="neo-friend-blurb">· {friend.blurb}</span>
                    )}
                  </li>
                ))}
              </ul>
            </NeoSection>

            <StarDivider />

            <NeoSection title="let's chat!" id="book">
              <div
                className="text-center p-5"
                style={{ border: "3px double #d6008f", background: "#fff7fb" }}
              >
                <p className="mx-auto mb-3">
                  Got an idea, a question, or just want to say hi? Grab a slot
                  and let&apos;s hop on a quick call or chat.
                </p>
                <Link
                  href="https://cal.stimmie.dev"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="neo-link-card inline-block font-bold"
                >
                  ☎ book a call or quick chat ↗
                </Link>
              </div>
            </NeoSection>

            <StarDivider />

            <NeoSection title="buy me a kape" id="support">
              <div
                className="text-center p-5"
                style={{ border: "3px double #d6008f", background: "#fff7fb" }}
              >
                <p className="mx-auto mb-3">
                  Found something here useful? Kape is my little
                  &ldquo;buy me a coffee&rdquo; corner. No pressure at all, but
                  if you&apos;d like to support my work, it genuinely means a
                  lot.
                </p>
                <Link
                  href="https://kape.stimmie.dev"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="neo-link-card inline-block font-bold"
                >
                  ☕ buy me a kape ↗
                </Link>
              </div>
            </NeoSection>

            <StarDivider />

            <div className="text-center py-2">
              <p
                className="text-base mb-3"
                style={{ fontFamily: "var(--neo-pixel)" }}
              >
                ~ link to me ~
              </p>
              <img
                src="/stimmie_88x31_67.png"
                srcSet="/stimmie_88x31_67.png 1x, /stimmie_88x31_67@2x.png 2x"
                alt="stimmie.dev 88x31 button with a goofy meme face"
                width={88}
                height={31}
                className="neo-pixel-btn inline-block"
              />
            </div>
          </main>
        </div>

        <footer className="neo-footer mt-3">
          <p>
            made with ♥ · <Link href="/changelog">v{version}</Link> ·{" "}
            <Link href="/archive">site history</Link>
          </p>
          <p className="text-[#ff00aa]">thanks 4 visiting!!!</p>
        </footer>
      </div>
    </div>
  );
}
