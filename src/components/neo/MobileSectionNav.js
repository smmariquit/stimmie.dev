import Link from "next/link";
import { SECTION_LINKS } from "@/components/home/sectionLinks";

export default function MobileSectionNav() {
  return (
    <nav
      className="neo-mobile-section-nav md:hidden mb-3"
      aria-label="On this page"
    >
      <p className="neo-mobile-section-nav-label">~ on this page ~</p>
      <ul className="neo-mobile-section-nav-list">
        {SECTION_LINKS.map((item) => (
          <li key={item.href}>
            <a href={item.href} className="neo-mobile-section-pill">
              {item.label}
            </a>
          </li>
        ))}
      </ul>
    </nav>
  );
}
