// src/app/get14/cheatsheet/page.js

import Link from "next/link";
import PageShell from "@/components/neo/PageShell";
import { compare, sheets } from "@/data/get14-cheatsheet";

const DESCRIPTION =
  "Unofficial cheat sheet for GET ¼!, the pen and paper programming contest by YSES. The numbers, rules and language traps that come up when tracing Python, JavaScript and C by hand.";

export const metadata = {
  title: "GET ¼ Cheat Sheet",
  description: DESCRIPTION,
  alternates: { canonical: "/get14/cheatsheet" },
  openGraph: {
    title: "GET ¼ Cheat Sheet",
    description: DESCRIPTION,
    url: "https://stimmie.dev/get14/cheatsheet",
    siteName: "Stimmie",
    locale: "en_US",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "GET ¼ Cheat Sheet",
    description: DESCRIPTION,
  },
};

const POWERS = Array.from({ length: 11 }, (_, n) => 2 ** n);

const JUMPS = [
  ["three languages", "#three-languages"],
  ...sheets.map((sheet) => [sheet.title, `#${sheet.id}`]),
  ["precedence", "#precedence"],
  ["habits", "#habits"],
];

// Tightest first. Each entry is one level.
const PRECEDENCE = [
  "( )  [ ]  .  and function calls",
  "**  (Python and JavaScript, right to left)",
  "-x  !x  ~x  ++x  --x",
  "*  /  //  %",
  "+  -",
  "<<  >>",
  "<  <=  >  >=",
  "==  !=",
  "&  then  ^  then  |",
  "&&  (and)",
  "||  (or)",
  "?:  then  =  +=  -=",
];

function Cell({ cell }) {
  if (!cell) return <td className="g14c-none">no built-in</td>;
  return (
    <td>
      <code>{cell[0]}</code>
      <strong>{cell[1]}</strong>
    </td>
  );
}

export default function Get14CheatSheetPage() {
  return (
    <PageShell
      title="~ get ¼ cheat sheet ~"
      intro="Unofficial cheat sheet for GET ¼!, the pen and paper programming contest by YSES."
      maxWidth="56rem"
    >
      <div className="g14-page g14 g14c">
        <p>
          These are the numbers, rules and traps that come up when you trace
          code by hand. Every result here was checked against a real run. Read
          it, then test yourself with the{" "}
          <Link href="/get14/drills">drills</Link> and the{" "}
          <Link href="/get14">practice problems</Link>.
        </p>

        <nav className="g14c-jumps" aria-label="Sections">
          {JUMPS.map(([label, href]) => (
            <a key={href} href={href}>
              {label}
            </a>
          ))}
        </nav>

        <section aria-labelledby="three-languages">
          <h2 id="three-languages" className="neo-section-title">
            same idea, three languages
          </h2>
          <div className="g14c-scroll">
            <table className="g14c-table">
              <thead>
                <tr>
                  <td />
                  <th scope="col">Python</th>
                  <th scope="col">JavaScript</th>
                  <th scope="col">C</th>
                </tr>
              </thead>
              <tbody>
                {compare.map((row) => (
                  <tr key={row.what}>
                    <th scope="row">{row.what}</th>
                    <Cell cell={row.python} />
                    <Cell cell={row.js} />
                    <Cell cell={row.c} />
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        {sheets.map((sheet) => (
          <section key={sheet.id} aria-labelledby={sheet.id}>
            <h2 id={sheet.id} className="neo-section-title">
              {sheet.title}
            </h2>
            {sheet.id === "numbers" && (
              <p className="g14c-powers">
                The powers of 2 are{" "}
                {POWERS.map((power) => (
                  <code key={power}>{power}</code>
                ))}
                .
              </p>
            )}
            {sheet.groups.map((group) => (
              <div key={group.title}>
                <h3 className="g14c-group">{group.title}</h3>
                <ul className="g14c-rows">
                  {group.rows.map((row) => (
                    <li key={row.code}>
                      <code className="g14c-code">{row.code}</code>
                      <strong className="g14c-out">{row.out}</strong>
                      <span className="g14c-why">{row.why}</span>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </section>
        ))}

        <section aria-labelledby="precedence">
          <h2 id="precedence" className="neo-section-title">
            what runs first
          </h2>
          <p>
            Operators higher on this list run before the ones below. The order
            is nearly the same in all three languages.
          </p>
          <ol className="g14c-order">
            {PRECEDENCE.map((level) => (
              <li key={level}>
                <code>{level}</code>
              </li>
            ))}
          </ol>
          <p className="mt-2">
            Python has two differences. Its bitwise operators run before
            comparisons, so <code>6 &amp; 3 == 2</code> is True in Python and 0
            in C and JavaScript. Its <code>**</code> also runs before a minus
            sign, so <code>-2 ** 2</code> is -4.
          </p>
        </section>

        <section aria-labelledby="habits">
          <h2 id="habits" className="neo-section-title">
            tracing habits
          </h2>
          <ol className="neo-facts list-decimal pl-6 m-0 mt-2 space-y-1">
            <li>Check which language it is before reading anything else.</li>
            <li>
              Write one column for each variable and cross out old values.
            </li>
            <li>Run one statement at a time, even when the line looks easy.</li>
            <li>
              List every value a loop variable takes, and stop before the end.
            </li>
            <li>
              Copy the output exactly, with its spaces, line breaks and capital
              letters.
            </li>
          </ol>
        </section>
      </div>
    </PageShell>
  );
}
