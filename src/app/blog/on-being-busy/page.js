import Link from "next/link";
import PageShell from "@/components/neo/PageShell";

export const metadata = {
  title: "On Being Busy",
  description: "Being busy is a means to an end. So what is the end?",
  openGraph: {
    title: "On Being Busy",
    description: "Being busy is a means to an end. So what is the end?",
    url: "https://stimmie.dev/blog/on-being-busy",
    type: "article",
    images: [{ url: "/blog/on-being-busy/cover.png", width: 1200, height: 630, alt: "A week as a Tetris board, pieces labelled standup, thesis, org mtg, stacked almost to the top with holes buried under them, one more piece falling" }],
  },
  twitter: { card: "summary_large_image", images: ["/blog/on-being-busy/cover.png"] },
};

export default function OnBeingBusyPage() {
  return (
    <PageShell title="On Being Busy" current="/blog" maxWidth="52rem">
      <p className="mb-4 text-base">
        <Link href="/blog">◄ back to blog</Link>
      </p>

      <article className="neo-prose">
        <p className="text-base neo-muted font-mono">September 2026</p>

        <p>I do not understand people who take pride in being busy. Being busy is a means to an end. It is the thing you do so that some other thing can happen, and on its own it tells you nothing about whether that other thing was worth it.</p>

        <p>So I cringe a little, quietly, when someone posts a screenshot of their Google Calendar with every slot coloured in, captioned like it is a medal. What are you busy for? Is any of it meaningful? Are you helping someone? Getting better at something? Having fun, at least? Or did you say yes to too many things, and now the calendar is the only proof you have that the week happened.</p>

        <p>A full calendar looks the same whether the hours went to something you care about or to a meeting that could have been a text. The screenshot cannot tell the difference. Neither can the people liking it. That is exactly why it is a strange thing to be proud of.</p>

        <hr />

        <p>I think the honest version of the flex is usually one of two things. Either the person has not learned to say no, and the calendar is the record of that. Or the busyness is doing a job for them. It is hard to sit with the question of whether your life is going anywhere when there is no gap in the day to sit in. A packed week is a very good place to hide.</p>

        <p>That is the &quot;perhaps&quot; I kept reaching for. Perhaps you are not managing your time badly. Perhaps you are managing it very well, for the purpose of never having any.</p>

        <p>The people I admire are the ones who have fun, who keep doing things with no finish line alongside their responsibilities. Philosophers call these atelic activities. A walk that goes nowhere is one. So is a game you are bad at and keep playing anyway. I admire that far more than someone who is always busy and has lost all sense of whimsy, because the first person has kept the part of themselves that the busyness was supposed to be for.</p>

        <p>I would rather see an empty afternoon on someone&apos;s calendar and hear what they did with it.</p>

        <hr />

        <p className="text-sm neo-muted font-mono">
          Also on <a href="https://medium.com/@stimmieuwu/on-being-busy-0ed65a9d77a6">Medium</a>.
        </p>
      </article>
    </PageShell>
  );
}
