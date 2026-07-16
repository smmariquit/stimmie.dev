import Link from "next/link";
import PageShell from "@/components/neo/PageShell";

export const metadata = {
  title: "The Province of Doing Nothing",
  description: "I grew up in the province, where doing nothing is an art form practiced by experts.",
};

export default function ProvinceOfDoingNothingPage() {
  return (
    <PageShell title="The Province of Doing Nothing" current="/blog" maxWidth="52rem">
      <p className="mb-4 text-base">
        <Link href="/blog">◄ back to blog</Link>
      </p>

      <article className="neo-prose">
        <p className="text-base neo-muted font-mono">April 2025</p>

        <p>I grew up in the province, where doing nothing is an art form practiced by experts. The old men at the sari-sari store have perfected it. They sit on plastic chairs that have molded to the exact shape of their bodies over decades. They watch the road. They comment on the road. They are the road&apos;s critics, its most devoted audience.</p>

        <p>&quot;May dumaan,&quot; one will say. Someone passed by. This is news. This is the dispatch.</p>

        <p>In the city, doing nothing is a moral failure. You are wasting time. You are unproductive. You are, worst of all, not optimizing. Every minute must produce something: a task completed, a skill acquired, a connection monetized. The city has decided that the purpose of human life is throughput.</p>

        <h2>The Chair</h2>

        <p>My lolo had a chair on the porch, a wooden thing, dark with age and arm-oil, that creaked when you sat in it but held firm. He sat in that chair every afternoon from about two o&apos;clock until merienda. He watched the mango tree. He watched the chickens. He watched the clouds, which, in the province, are worth watching because they are enormous and slow and have nowhere to be, same as him.</p>

        <p>If you asked him what he was doing, he would say &quot;nag-iisip-isip.&quot; Thinking-thinking. The reduplication implies something less than real thinking. Not working out a problem, not planning. Just letting the mind go where it goes, the way you let a carabao wade into a river.</p>

        <p>He was not idle. He was doing the most important thing a person can do, which is being present for his own life. He was letting the afternoon happen to him. This is a skill. Most people have lost it.</p>

        <h2>Against Productivity</h2>

        <p>I am suspicious of productivity. Not of work, work is fine, work is often good, but of the ideology that has grown up around it like kudzu on a telephone pole. The ideology says: every hour is a resource. To leave a resource unexploited is waste. Waste is sin.</p>

        <p>This is the logic of a factory applied to a human being. It works about as well as you&apos;d expect. The factory produces output. The human being produces anxiety, guilt, and a chronic sense that something has gone wrong, although by every external metric things are going fine.</p>

        <p>My lolo did not have this problem. He had other problems, money was often tight, the roof leaked, the local politics were a soap opera he could not change the channel on, but he did not have the specifically modern problem of feeling guilty for sitting still.</p>

        <hr />

        <p>When I go home to the province now, I try to sit in the chair. I am bad at it. Within five minutes I have reached for my phone. The phone is a portal to a universe of things that need doing, or at least things that present themselves as needing doing. None of them need doing. The mango tree is still there. The clouds are still enormous.</p>

        <p>I put the phone face-down. I watch the chickens. The chickens are unbothered by my existential crisis. They have grain to peck and dirt to scratch and this is sufficient. I aspire to the contentment of the chicken, which does not need to justify its existence through quarterly reviews.</p>

        <h2>Merienda</h2>

        <p>At four o&apos;clock, someone brings out the merienda. This is the reward for the labor of sitting, although it is not framed as a reward because no one considers the sitting to be labor. The merienda arrives because it is four o&apos;clock and four o&apos;clock is when merienda arrives. That is the full extent of the reasoning.</p>

        <p>There are kakanin, or there are biscuits, or there is champorado if someone felt ambitious. The food is eaten slowly. There is no rush. The evening will come on its own schedule, same as always.</p>

        <p>I have tried to explain the province&apos;s relationship to time to my city friends, and they nod politely the way you nod at someone describing a country you have no intention of visiting. They understand the concept. They do not understand the practice. The practice requires something the city has trained out of them, which is the ability to sit with time rather than against it.</p>

        <p>The province is not paradise. It has its own cruelties and tediums. But it has this one thing the city does not: permission to be still. That permission costs nothing, requires no app, no retreat, no self-help book. You just sit down. You watch what happens. Usually, nothing happens. That is the whole point.</p>
      </article>
    </PageShell>
  );
}
