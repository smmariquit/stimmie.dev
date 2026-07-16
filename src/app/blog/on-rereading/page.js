import Link from "next/link";
import PageShell from "@/components/neo/PageShell";

export const metadata = {
  title: "On Rereading",
  description: "I reread books the way some people revisit restaurants. Not because I've forgotten what's on the menu.",
};

export default function OnRereadingPage() {
  return (
    <PageShell title="On Rereading" current="/blog" maxWidth="52rem">
      <p className="mb-4 text-base">
        <Link href="/blog">◄ back to blog</Link>
      </p>

      <article className="neo-prose">
        <p className="text-base neo-muted font-mono">January 2025</p>

        <p>I reread books the way some people revisit restaurants. Not because I&apos;ve forgotten what&apos;s on the menu. Because I know exactly what&apos;s on the menu, and I want it again.</p>

        <p>There is a strain of thinking that says rereading is waste. You already know what happens. The mystery is gone. Why not read something new? This argument has the same energy as asking why you&apos;d eat dinner when you already ate lunch. The point is not novelty. The point is sustenance.</p>

        <h2>The Same River</h2>

        <p>The book does not change. You change. This is the entire mechanism of rereading and it never stops working.</p>

        <p>I first read &quot;The Great Gatsby&quot; when I was seventeen and thought it was about a guy who throws great parties. I read it again at twenty-five and realized it was about wanting something so badly that the wanting becomes the thing, and the actual object of desire is beside the point. I read it a third time last year and saw it was about the way we narrate other people&apos;s lives to avoid narrating our own. Same book. Three readings. Three books.</p>

        <p>This is not deep literary analysis. This is just what happens when you bring a different self to the same text. The words stay put. You are the variable.</p>

        <h2>Comfort and Confrontation</h2>

        <p>Some books I reread for comfort. I know what is going to happen. I know the sentences. I know which paragraphs will make me stop and look up. This is not laziness. This is the reading equivalent of going home, you know where everything is, you know how the light falls in the afternoon, and the familiarity is the point.</p>

        <p>Other books I reread because they were smarter than I was the first time. They said things I didn&apos;t catch. They made arguments I wasn&apos;t ready to hear. Rereading them is not comfort. It is confrontation. The book has been sitting on the shelf, patient and unchanged, waiting for me to catch up.</p>

        <p>Both kinds of rereading are necessary. You need the books that feel like home and you need the books that feel like being lost in a neighborhood you should know better by now.</p>

        <hr />

        <p>I keep a shelf of books I intend to reread. It is not organized by author or genre. It is organized by something I can only describe as emotional weather: books for restlessness, books for sadness, books for the specific kind of afternoon where you are not unhappy but not quite happy either and you need someone else&apos;s words to tell you what the feeling is called.</p>

        <p>New books are a gamble. You might love them. You might not. You won&apos;t know until you&apos;re forty pages in, by which point you&apos;ve committed the evening. Rereading is a sure thing. You know the return. You know the cost. The transaction is honest.</p>

        <p>I do not apologize for rereading. I have too many new books on the pile and not enough time, same as everyone, and I choose to spend some of that time on books that have already proven they are worth it. That is not waste. That is judgment.</p>
      </article>
    </PageShell>
  );
}
