import Link from "next/link";
import PageShell from "@/components/neo/PageShell";

export const metadata = {
  title: "On Walking",
  description: "There is a particular hour in the late afternoon when the campus empties out and the acacia shadows stretch long across the oval...",
};

export default function OnWalkingPage() {
  return (
    <PageShell title="On Walking" current="/blog" maxWidth="52rem">
      <p className="mb-4 text-base">
        <Link href="/blog">◄ back to blog</Link>
      </p>

      <article className="neo-prose">
        <p className="text-base neo-muted font-mono">June 2025</p>

        <p>There is a particular hour in the late afternoon when the campus empties out and the acacia shadows stretch long across the oval. I have always preferred walking at that hour. Not for exercise, not to get somewhere, but because walking is the only activity I know that asks nothing of you and gives everything back.</p>

        <p>You put one foot in front of the other and the world rearranges itself. Problems that sat like stones in your chest begin to loosen. Not because you solved them, you didn't, but because your body remembered that it is an animal, and animals do not hold committee meetings about their anxieties.</p>

        <p>The best walks have no destination. A destination turns a walk into an errand. I want the opposite of an errand. I want to be unpurposed, unmapped, moving through space with the dumb happiness of a dog let off the leash.</p>

        <hr />

        <p>I distrust people who never walk. Not in a moral way, I'm not that sort, but in a diagnostic way. Something in them has gone rigid. They have forgotten that the body is not a vehicle for the head. It is the other way around.</p>

        <p>When I cannot write, I walk. When I cannot think, I walk. When I am angry at someone and composing devastating replies in my head, I walk, and by the third kilometer the replies have dissolved and I am watching a mynah bird harass a cat and the anger seems very far away and very silly.</p>

        <p>Walking does not solve your problems. It puts them at the correct scale.</p>
      </article>
    </PageShell>
  );
}
