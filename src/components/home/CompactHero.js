import Link from "next/link";
import { socialCategories } from "@/data/socials";
import { ABOUT_TEXT, MOBILE_TAGLINE } from "./constants";

const socialLinks = socialCategories
  .flatMap((cat) => cat.links)
  .filter(
    (link) => !["Hackathon Guide", "Freshie Resources"].includes(link.name),
  );

const resourceLinks = socialCategories
  .flatMap((cat) => cat.links)
  .filter((link) =>
    ["Hackathon Guide", "Freshie Resources"].includes(link.name),
  );

export function DesktopHero() {
  return (
    <header className="pt-28 pb-8 px-6 md:px-12 max-w-7xl mx-auto relative z-10">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 lg:gap-12 items-start">
        <div>
          <h1 className="text-5xl md:text-6xl lg:text-7xl font-black uppercase leading-[0.9] tracking-tighter bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400 text-transparent bg-clip-text drop-shadow-lg">
            CRAFTING
            <br />
            DIGITAL
            <br />
            WORLDS.
          </h1>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 border-t lg:border-t-0 lg:border-l border-purple-900/30 pt-6 lg:pt-0 lg:pl-8">
          <div className="sm:col-span-2">
            <h2 className="text-xs font-bold uppercase tracking-widest text-indigo-400 mb-3">
              [ ABOUT ]
            </h2>
            <p className="text-sm leading-relaxed text-indigo-100/70">
              {ABOUT_TEXT}
            </p>
          </div>
          <div>
            <h2 className="text-xs font-bold uppercase tracking-widest text-pink-400 mb-3">
              [ SOCIALS ]
            </h2>
            <div className="flex flex-wrap gap-x-4 gap-y-3">
              {socialLinks.map((link) => {
                const Wrapper = link.href ? Link : "div";
                return (
                  <Wrapper
                    key={link.name}
                    href={link.href || undefined}
                    className={`flex items-center gap-2 text-sm font-medium transition-colors ${link.href ? "hover:text-pink-400" : "text-gray-400"}`}
                    title={link.alt || link.name}
                  >
                    {link.icon && (
                      <img
                        src={link.icon}
                        alt=""
                        className="w-4 h-4 object-contain hover:scale-110 transition-transform"
                      />
                    )}
                    {link.name}
                  </Wrapper>
                );
              })}
            </div>
          </div>
          <div>
            <h2 className="text-xs font-bold uppercase tracking-widest text-cyan-400 mb-3">
              [ RESOURCES ]
            </h2>
            <div className="flex flex-col gap-4">
              {resourceLinks.map((link) => (
                <Link
                  key={link.name}
                  href={link.href || "#"}
                  className="group flex items-start gap-3"
                  title={link.alt || link.name}
                >
                  {link.icon && (
                    <img
                      src={link.icon}
                      alt=""
                      className="w-5 h-5 object-contain mt-0.5 group-hover:scale-110 transition-transform"
                    />
                  )}
                  <div>
                    <span className="text-sm font-bold text-gray-200 group-hover:text-cyan-400 transition-colors block leading-tight">
                      {link.name}
                    </span>
                    <span className="text-xs text-cyan-600/70 group-hover:text-cyan-400 font-mono mt-1 block">
                      ↗ Read
                    </span>
                  </div>
                </Link>
              ))}
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}

export function MobileHero() {
  return (
    <header className="pt-20 pb-4 px-6 max-w-7xl mx-auto relative z-10">
      <h1 className="text-3xl font-black uppercase tracking-tighter bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400 text-transparent bg-clip-text mb-2">
        STIMMIE
      </h1>
      <p className="text-sm text-indigo-100/70 leading-relaxed mb-4 max-w-sm">
        {MOBILE_TAGLINE}
      </p>
      <div className="flex flex-wrap gap-3">
        {socialLinks
          .filter((link) => link.href && link.icon)
          .map((link) => (
            <Link
              key={link.name}
              href={link.href}
              title={link.alt || link.name}
              aria-label={link.alt || link.name}
              className="p-2 rounded-lg border border-purple-900/30 bg-indigo-950/20 hover:border-pink-500/40 transition-colors"
            >
              <img src={link.icon} alt="" className="w-5 h-5 object-contain" />
            </Link>
          ))}
      </div>
    </header>
  );
}
