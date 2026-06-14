import Link from "next/link";
import { STIMMIEVERSE_LINKS } from "./constants";

export default function SiteNav() {
  return (
    <nav className="fixed top-0 w-full px-6 py-4 md:px-8 md:py-5 flex justify-between items-center bg-[#050014]/80 backdrop-blur-lg z-50 border-b border-purple-900/30">
      <div className="font-bold tracking-widest text-sm uppercase flex-shrink-0 text-transparent bg-gradient-to-r from-pink-500 to-indigo-400 bg-clip-text drop-shadow-[0_0_8px_rgba(236,72,153,0.3)]">
        STIMMIE // CREATOR
      </div>
      <div className="hidden md:flex gap-6 font-medium text-xs tracking-widest uppercase items-center">
        <span className="text-purple-500/50 hidden lg:inline">
          THE STIMMIEVERSE //
        </span>
        {STIMMIEVERSE_LINKS.map((site) => (
          <Link
            key={site.name}
            href={site.url}
            target={site.local ? "_self" : "_blank"}
            rel={site.local ? "" : "noopener noreferrer"}
            className="hover:text-cyan-400 text-gray-400 transition-colors flex-shrink-0"
          >
            {site.name} {!site.local && "↗"}
          </Link>
        ))}
      </div>
    </nav>
  );
}
