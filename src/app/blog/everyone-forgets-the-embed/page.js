import Link from "next/link";
import PageShell from "@/components/neo/PageShell";

const desc =
  "People design every pixel of their website and forget the card that shows up when someone pastes the link. I forgot too.";

export const metadata = {
  title: "Everyone forgets the embed",
  description: desc,
  openGraph: {
    title: "Everyone forgets the embed",
    description: desc,
    url: "https://stimmie.dev/blog/everyone-forgets-the-embed",
    type: "article",
    images: [{ url: "/blog/everyone-forgets-the-embed/cover.png" }],
  },
  twitter: {
    card: "summary_large_image",
    images: ["/blog/everyone-forgets-the-embed/cover.png"],
  },
};

function Figure({ src, alt, caption }) {
  return (
    <figure className="my-6">
      <img
        src={src}
        alt={alt}
        className="w-full border-2 border-current"
        loading="lazy"
      />
      <figcaption className="mt-2 text-sm neo-muted font-mono">
        {caption}
      </figcaption>
    </figure>
  );
}

export default function EveryoneForgetsTheEmbedPage() {
  return (
    <PageShell
      title="Everyone forgets the embed"
      current="/blog"
      maxWidth="52rem"
    >
      <p className="mb-4 text-base">
        <Link href="/blog">◄ back to blog</Link>
      </p>

      <article className="neo-prose">
        <p className="text-base neo-muted font-mono">November 2026</p>

        <p>
          People put a lot of work into making their websites look nice. They
          pick the fonts, tune the spacing and argue about the shade of a
          button. Then they forget about the embed.
        </p>

        <p>
          By the embed I mean the small card that shows up under a link when you
          paste it into Discord, Messenger or Slack. It has a title, a line of
          description, sometimes a coloured bar down the left side and a big
          image. It&apos;s usually the first thing anyone sees of your website,
          and it&apos;s the part almost nobody remembers to design.
        </p>

        <p>
          I forgot too. When I started writing this, I checked what my own essay
          &quot;On Walking&quot; handed to Discord when someone pasted it. The
          title said &quot;Stimmie&quot;. The description said I&apos;m a
          software engineer with a lifelong passion for technology, and the
          picture was a screenshot of my homepage. Nothing on the card mentioned
          walking. Two more of my older essays previewed the exact same way,
          because I never gave them their own tags and they fell back to my
          homepage&apos;s. Someone scrolling past would have had no idea the
          link was an essay at all. I&apos;ve fixed all three now.
        </p>

        <Figure
          src="/blog/everyone-forgets-the-embed/on-walking-before.png"
          alt="Discord preview of stimmie.dev/blog/on-walking with the title Stimmie, the homepage description and a screenshot of the homepage"
          caption="On Walking, pasted into Discord before the fix."
        />

        <p>
          I think it&apos;s easy to forget because you never see it while
          you&apos;re building. The embed doesn&apos;t live on your page. You
          write a few meta tags in the head of the page (<code>og:title</code>,{" "}
          <code>og:description</code>, <code>og:image</code>, and{" "}
          <code>theme-color</code> if you want Discord to tint the bar). Then
          you hand them to an app you don&apos;t own, and it draws your card
          however it likes. If you skip them, nothing breaks and nothing warns
          you. You only find out when someone else pastes your link.
        </p>

        <p>
          That would be fine if people clicked through, but most don&apos;t. In
          2016, researchers at Columbia and Inria followed a month of tweets
          linking to stories from the BBC, CNN, the New York Times, Fox and the
          Huffington Post. They found that 59 percent of the links shared in
          their sample were never clicked at all. A 2024 Penn State study of
          more than 35 million public Facebook posts found that about 75 percent
          of shares happened without the sharer clicking the link first. For a
          lot of people, the embed is the only part of your website they&apos;ll
          ever see.
        </p>

        <p>
          The fix is cheap. Before you share anything, paste it into a chat with
          yourself and look at the card. If it doesn&apos;t make you want to tap
          it, it won&apos;t make anyone else want to either.
        </p>

        <hr />

        <p className="text-sm">
          Further reading. Gabielkov, Ramachandran, Chaintreau and Legout,{" "}
          <a href="https://doi.org/10.1145/2896377.2901462">
            Social Clicks: What and Who Gets Read on Twitter?
          </a>{" "}
          (SIGMETRICS 2016). Sundar and colleagues,{" "}
          <a href="https://doi.org/10.1038/s41562-024-02067-4">
            Sharing without clicking on news in social media
          </a>{" "}
          (Nature Human Behaviour, 2024). Discord&apos;s page on{" "}
          <a href="https://docs.discord.com/developers/link-previews/overview">
            link previews
          </a>
          , which lists every tag it reads.
        </p>
      </article>
    </PageShell>
  );
}
