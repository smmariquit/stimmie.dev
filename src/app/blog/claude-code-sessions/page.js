import Link from "next/link";
import PageShell from "@/components/neo/PageShell";

const title = "I have 27 Claude Code sessions open right now";
const desc =
  "One session per project, a front desk for everything else, and a calendar only one script can touch. Almost every rule in it has a date.";

export const metadata = {
  title,
  description: desc,
  openGraph: {
    title,
    description: desc,
    url: "https://stimmie.dev/blog/claude-code-sessions",
    type: "article",
    images: [{ url: "/blog/claude-code-sessions/cover.png" }],
  },
  twitter: {
    card: "summary_large_image",
    images: ["/blog/claude-code-sessions/cover.png"],
  },
};

export default function ClaudeCodeSessionsPage() {
  return (
    <PageShell title={title} current="/blog" maxWidth="52rem">
      <p className="mb-4 text-base">
        <Link href="/blog">◄ back to blog</Link>
      </p>

      <article className="neo-prose">
        <p className="text-base neo-muted font-mono">November 2026</p>

        <p>
          As I write this, there are 27 Claude Code sessions open on my laptop.
          One handles my org UXLB, one handles school, one handles job
          applications, and so on down a list of 30 projects. Most of them are sitting idle, waiting for me to say
          something. A few are in the middle of a task.
        </p>

        <p>
          I didn&apos;t plan any of this. It grew one complaint at a time, and I
          think the complaints are the interesting part, so this is the setup
          and how it got that way.
        </p>

        <p>
          I&apos;m in a lot of orgs, I have classes, and I take on paid work.
          Each of those has its own group chats, docs, repos and deadlines. When
          I used one long Claude Code session for everything, it grew until
          compacting it took ages, and every new topic meant explaining the last
          one again.
        </p>

        <p>
          So each project now gets its own folder and its own session. A small
          script called <code>cs</code> holds the list. <code>cs uxlb</code>{" "}
          reopens the UXLB session exactly where it left off.{" "}
          <code>cs board</code> lists every running session, whose turn it is
          and what each one still has open. <code>cs bg &lt;name&gt; &lt;task&gt;</code>{" "}
          starts a project&apos;s agent on a job in the background while I do
          something else.
        </p>

        <p>
          This is the top line of the board right now. I left out the rows
          under it because they&apos;re full of other people&apos;s names.
        </p>

        <pre className="text-sm"><code>{`Fri 12:27   27 sessions   2218 MB of memory free`}</code></pre>

        <p>
          Each project keeps its state in the CLAUDE.md file in its folder,
          under a section called State. When one project&apos;s work changes
          something another project relies on, the agent adds one dated line to
          the other project&apos;s State. That way I don&apos;t have to
          re-explain anything when two projects meet.
        </p>

        <p>
          One session is different. I call it the front desk. Anything that
          doesn&apos;t belong to a project goes there, like a random form, an
          email, or an ask that touches three orgs at once. It hands project
          work to that project&apos;s agent, keeps only the summary, and ends
          every reply with one &quot;Still open&quot; list built from{" "}
          <code>cs board</code>.
        </p>

        <p>
          The sessions can also talk to each other over Unix sockets. A project
          session can ask the front desk something directly instead of waiting
          for me to pass the message along.
        </p>

        <p>
          With this many sessions, dates get messy. If every session could write
          to my Google Calendar, two of them finding the same deadline would add
          it twice, and they&apos;d add dates I hadn&apos;t confirmed.
        </p>

        <p>
          Now no agent writes to my calendar. When an agent finds a date I have
          to act on, it appends one tab-separated line to a queue file for its
          project. These are the columns.
        </p>

        <pre className="text-sm"><code>{`date  start  end  title  location  kind  source  sure`}</code></pre>

        <p>
          The last column says whether the agent checked the date against its
          source. A script called <code>cal-sync</code> runs every 30 minutes
          from 7 AM to 11 PM. It adds the confirmed dates. Everything it
          isn&apos;t sure about goes into one all-day event called &quot;Waiting
          on you&quot;, which also lists clashes in the next 7 days. An hour or
          two before a project event, it starts that project&apos;s agent to
          write a prep note into the event, as long as the laptop has at least 3
          GB of memory free.
        </p>

        <p>
          My global CLAUDE.md is mostly rules, and almost every rule has the
          date I wrote it and the thing that went wrong. These are a few of
          them.
        </p>

        <p>
          An agent read a Messenger chat by pulling out the text, which drops
          who sent each message, and it reported my own messages as someone
          else&apos;s. Now senders have to be confirmed from a screenshot, and
          every chat gets a log file so the next session doesn&apos;t reread it.
          There are 130 of those logs.
        </p>

        <p>
          Agents sent my messages back to back in the same second. Now they wait
          at least 10 seconds between sends and check each message after it goes
          out.
        </p>

        <p>
          Office tasks left me with a pile of LibreOffice windows. Now documents
          are converted with no window and opened for me once, when they&apos;re
          finished.
        </p>

        <p>
          Agents wrote &quot;this weekend&quot; for work that takes them a few
          minutes. Now they estimate in agent time and do it in the same
          session.
        </p>

        <p>
          Then there&apos;s how the agents write. I got tired of telling every session
          not to use colons, em dashes and sentence fragments, so a hook adds
          those rules to every single message I send.
        </p>

        <p>
          There are also things the agents can&apos;t do. Every project agent
          has the tools for deleting email, trashing Drive
          files and deleting calendar events taken away. Anything that sends,
          posts or pays needs me to say so in the chat, every time. Logins,
          passwords, 2FA codes and CAPTCHAs are always mine.
        </p>

        <p>
          Some of the tools exist only because something kept failing.
          Viber&apos;s desktop app took about 830 tool calls over six weeks
          before an agent found one way to drive it that works, and that method
          now lives in a skill so nobody has to find it again.{" "}
          <code>sec</code> reads a secret from 1Password once and keeps it in
          the login keyring, because a 1Password prompt in the middle of a job
          would block it until I came back.
        </p>

        <p>
          It still breaks plenty. Both Gmail connectors were signed out today.
          Hacker News wouldn&apos;t take a link to this blog from my account. A
          browser agent removed five of my calendars today by clicking at fixed
          positions, and only four came back.
        </p>

        <p>
          Most of the setup is a record of the ways it hasn&apos;t worked, with
          dates, so the same thing doesn&apos;t break twice.
        </p>
      </article>
    </PageShell>
  );
}
