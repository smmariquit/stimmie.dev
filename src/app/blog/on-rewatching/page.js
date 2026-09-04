import Link from "next/link";
import PageShell from "@/components/neo/PageShell";

export const metadata = {
  title: "On Rewatching",
  description: "I have watched Your Name five times. I am not entirely sure about the number, but five is what I tell people and five is what I believe.",
};

export default function OnRewatchingPage() {
  return (
    <PageShell title="On Rewatching" current="/blog" maxWidth="52rem">
      <p className="mb-4 text-base">
        <Link href="/blog">◄ back to blog</Link>
      </p>

      <article className="neo-prose">
        <p className="text-base neo-muted font-mono">January 2025</p>

        <p>I have watched Your Name five times. I am not entirely sure about the number, because the first two were close together and may have been one long evening, but five is what I tell people and five is what I believe.</p>

        <p>There is a strain of thinking that says rewatching is a waste. You already know the comet falls. You already know they forget each other. Why sit through it again when there are so many films you haven&apos;t seen? I understand the argument, and I think it has the same energy as asking why you would eat dinner when you already ate lunch. The point was never the surprise. The point is that it feeds you, and you are hungry again.</p>

        <hr />

        <p>The film does not change. I do. That is the whole mechanism, and I keep being surprised that it works every time.</p>

        <p>The first time I watched it I was in high school, and I thought it was about a boy and a girl swapping bodies, which it is, and I thought the comet was the twist, which it isn&apos;t. The second time, a year or so later, I noticed that half the film is about forgetting, and that the swapping is almost a distraction from the fact that these two people spend most of the runtime trying to hold on to a name. The fourth time, which was last year, I sat through the scene at twilight on the crater rim, the one where they finally stand in front of each other and have a few minutes before it goes, and I could not tell you why my chest hurt, only that it hadn&apos;t hurt like that at seventeen.</p>

        <p>Same film, same runtime. I brought a different person to it each time, and each time it had something ready.</p>

        <p>None of this is deep. It is what happens when you let a thing sit and come back to it later. The frames stay put. You are the variable.</p>

        <hr />

        <p>Some rewatches are for comfort. I know where the songs land. I know the exact cut where Sparkle starts, and I know I will get chills at it regardless of what kind of week it has been, which is a strange thing to be able to rely on and I am grateful for it. This is the film equivalent of going home. You know where everything is. You know how the light comes in.</p>

        <p>Other rewatches are not comfortable at all. The film was smarter than me the first time and I am still catching up. It knew something about time, about how you can miss someone you cannot name, and it said it plainly and I wasn&apos;t ready. Rewatching it feels like being told the same thing again, more slowly, by something that has been patient with me.</p>

        <p>I think you need both. You need the films that feel like home and you need the films that feel like a place you should know better than you do.</p>

        <hr />

        <p>I keep a list of things to rewatch. It is not organised by director or by year. It is organised by something I can only describe as weather: films for restlessness, films for the specific kind of Sunday evening when you are not unhappy but not quite anything else either, and you need someone else&apos;s story to tell you what the feeling is called. Your Name is on it more than once, under different weather.</p>

        <p>A new film is a gamble. You might love it. You might not. You will not know until forty minutes in, by which point the evening is spent. A rewatch is a sure thing. You know what it costs and you know what it gives back, and the deal is honest.</p>

        <p>I don&apos;t apologise for it. There are too many new films and not enough evenings, same as everyone, and I choose to spend some of those evenings on the one that has already proven it is worth it. Perhaps that is not waste at all. Perhaps that is just knowing, for once, what I want.</p>
      </article>
    </PageShell>
  );
}
