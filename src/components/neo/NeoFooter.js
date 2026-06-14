import Link from "next/link";
import { getSiteVersion } from "@/lib/changelog";
import VisitorCounter from "./VisitorCounter";

export default function NeoFooter() {
  const version = getSiteVersion();
  return (
    <footer className="neo-footer mt-3">
      <div className="mx-auto mb-3 max-w-[16rem]">
        <VisitorCounter />
      </div>
      <p>
        made with ♥ · <Link href="/changelog">v{version}</Link> ·{" "}
        <Link href="/archive">site history</Link>
      </p>
      <p className="text-[#ff00aa]">thanks 4 visiting!!!</p>
    </footer>
  );
}
