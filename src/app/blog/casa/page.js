// src/app/blog/casa/page.js

import Link from "next/link";
import PageShell from "@/components/neo/PageShell";

export const metadata = {
  title: "11 things I learned from my software engineering class",
  description:
    "Lessons from leading a full enterprise app build for a software engineering course at UPLB.",
};

export default function CasaPage() {
  return (
    <PageShell
      title="11 things I learned from my software engineering class"
      current="/blog"
      maxWidth="52rem"
    >
      <p className="mb-4 text-base">
        <Link href="/blog">◄ back to blog</Link>
      </p>

      <article className="neo-prose">
        <p className="text-base neo-muted font-mono">May 22, 2026</p>

        <p>
          This sem, I had the pleasure to be the project manager, basically the
          big boss leader, of our software engineering course in UPLB. Over four
          months, we had to create a full enterprise app for student housing. It
          turned out to be one of the most chaotic but growth-heavy experience
          in my engineering career :)
        </p>
        <p>
          Visit our app at <Link href="https://uplb.casa">uplb.casa</Link>
        </p>
        <p>
          See our repository at{" "}
          <Link href="https://github.com/smmariquit/comsci-128">
            github.com/smmariquit/comsci-128
          </Link>
        </p>
        <p>
          Learn how to test it by reading our{" "}
          <Link href="https://github.com/smmariquit/comsci-128/blob/main/betaTesting.md">
            beta testing guide
          </Link>
        </p>

        <h2>How I was chosen</h2>
        <p>
          While (I think) all other classes had their leaders chosen by the
          instructor, our instructor opted to let the rest of my class choose.
          For some reason, my classmates didn&apos;t even have to stop and
          think. They just pointed at me. Perhaps that&apos;s because my
          batchmates in UPLB recognize me as some sort of expert. But for
          whatever reason, they trusted me with one of the biggest projects of
          their BSCS degree.
        </p>

        <h2>Non-technical lessons</h2>
        <p>
          1. This is *the* biggest one. You cannot mask ineffective leadership
          by overcontribution. Yes, I was the one with the most commits, most
          hours worked. I could give myself a medal. But I realized that&apos;s
          not a good approach because I wasn&apos;t going out of my comfort
          zone. Coding for 14 hours, while difficult, is a difficulty I&apos;m
          personally accustomed to. But confronting people for their
          shortcomings, motivating them, and keeping team morale is not
          something I&apos;m naturally good at.
        </p>
        <p>
          2. Make people *own* things instead of just handing out tasks. People
          will end up more passionate about their work if they fully understand
          why they are doing it and that it is their brainchild so to speak
        </p>
        <p>
          3. The power of passive aggressive corpo-speak messages insanely
          underrated and works well to get people to do their shit LMAO
        </p>
        <p>
          4. Since the last company I worked at was a startup (E-Konsulta),
          there&apos;s sort of a mindset where you should be quick, document as
          little as possible, and just do the work. When you&apos;re leading a
          bunch of people of which some have underdeveloped communication skills
          and work ethic, this is less applicable. You should consistently track
          the tasks of the team.
        </p>
        <p>
          5. The beginning of any project is crucial in identifying who among in
          your team are the GOATs, the responsible people who take ownership,
          and those who tend to lag behind or procrastinate. Use this
          information wisely throughout the rest of the project to inform
          decisions on delegation.
        </p>
        <p>
          6. Democracy is unreliable, and you shouldn&apos;t form a subteam
          without a leader.
        </p>

        <h2>Technical lessons</h2>
        <p>
          1. Clearly communicate to people that if you code something, you
          should be the one to ensure that it gets to production.
        </p>
        <p>
          2. Clearly communicate the concept of code being stale and how a PR
          only has a life span before siya maging incompatiible with the current
          production code
        </p>
        <p>
          3. GitHub Dependabot and Security code scanning should definitely be
          used for production projects
        </p>
        <p>
          4. Keep GitHub PR templates simple. Don&apos;t add a huge checklist of
          tests that don&apos;t apply to the project.
        </p>
        <p>
          5. It is NEVER too late to resolve technical debt. I had the choice of
          whether or not to resolve something 4 weeks before the final deadline.
          I chose not too because I thought it&apos;s too late, but that
          specific thing was a big headache for those four weeks.
        </p>
      </article>
    </PageShell>
  );
}
