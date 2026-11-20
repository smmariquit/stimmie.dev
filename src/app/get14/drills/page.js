// src/app/get14/drills/page.js

import Link from "next/link";
import PageShell from "@/components/neo/PageShell";
import Drills from "./Drills";

const DESCRIPTION =
  "Unofficial 60 second drills for GET ¼!, the pen and paper programming contest by YSES. Small arithmetic, tracing skills, facts to memorize and language traps in Python, JavaScript and C.";

export const metadata = {
  title: "GET ¼ Drills",
  description: DESCRIPTION,
  alternates: { canonical: "/get14/drills" },
  openGraph: {
    title: "GET ¼ Drills",
    description: DESCRIPTION,
    url: "https://stimmie.dev/get14/drills",
    siteName: "Stimmie",
    locale: "en_US",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "GET ¼ Drills",
    description: DESCRIPTION,
  },
};

export default function Get14DrillsPage() {
  return (
    <PageShell
      title="~ get ¼ drills ~"
      intro="Unofficial 60 second drills for GET ¼!, the pen and paper programming contest by YSES."
      maxWidth="56rem"
    >
      <div className="g14-page">
        <p>
          These drills train the small steps inside the{" "}
          <Link href="/get14">practice problems</Link>, one line at a time. They
          aren&apos;t from YSES.
        </p>

        <Drills />
      </div>
    </PageShell>
  );
}
