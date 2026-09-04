import Link from "next/link";
import PageShell from "@/components/neo/PageShell";

const desc =
  "An adblocker, Privacy Badger, and LocalCDN do three different jobs. Here is what each one actually does, with crewmates, impostors, and one very sus cookie.";

export const metadata = {
  title: "Privacy tools explained with Among Us",
  description: desc,
  openGraph: {
    title: "Privacy tools explained with Among Us",
    description: desc,
    url: "https://stimmie.dev/blog/privacy-tools-explained-with-among-us",
    type: "article",
    images: [{ url: "/blog/privacy-among-us/cover.jpg", width: 1600, height: 900, alt: "Among Us crewmates" }],
  },
  twitter: {
    card: "summary_large_image",
    title: "Privacy tools explained with Among Us",
    description: desc,
    images: ["/blog/privacy-among-us/cover.jpg"],
  },
};

function Figure({ src, alt, caption }) {
  return (
    <figure className="my-6">
      <img src={src} alt={alt} className="w-full border-2 border-current" loading="lazy" />
      {caption && <figcaption className="mt-2 text-sm neo-muted font-mono">{caption}</figcaption>}
    </figure>
  );
}

export default function PrivacyAmongUsPage() {
  return (
    <PageShell title="Privacy tools explained with Among Us" current="/blog" maxWidth="52rem">
      <p className="mb-4 text-base">
        <Link href="/blog">◄ back to blog</Link>
      </p>

      <article className="neo-prose">
        <p className="text-base neo-muted font-mono">September 4, 2026. A longer version of a post I first put on Medium in March.</p>

        <Figure src="/blog/privacy-among-us/cover.jpg" alt="Among Us crewmates in a row" />

        <p>
          When I was deep into the privacy community, doing stuff like fully encrypting my disk with VeraCrypt, keeping
          ProtonVPN on with a kill switch, and using Signal instead of Telegram, I came across a bunch of browser
          extensions that everybody seemed to recommend together. Adblocker, Privacy Badger, LocalCDN. I installed all
          of them and treated them as collectibles. More tools, more privacy.
        </p>

        <p>
          It took me an embarrassingly long time to notice that the three of them do three different jobs, and that
          two of them barely overlap. So here is the version I wish somebody had drawn for me, in the only metaphor
          that ever made it stick.
        </p>

        <h2>The map</h2>

        <p>
          A web page is not one place. When you open a news article, the article itself comes from the news site, but
          the fonts might come from Google, the script library from a CDN, the ads from an ad company, the comments
          from a company the news site contracted, and the Like button from Facebook. The EFF&apos;s own explanation of
          Privacy Badger starts from exactly this picture, and it is the right place to start.
        </p>

        <Figure
          src="/blog/privacy-among-us/diagram-1.png"
          alt="Diagram of a news page loading resources from five third-party domains: Google Fonts, cdnjs, an ad network, an analytics tracker, and a Facebook Like button"
          caption="The site you typed in is the first party. Everyone else is a third party."
        />

        <p>
          In Among Us terms, the site you typed in is the ship you boarded. Every third-party request is a crewmate
          who walked in from somewhere else. Some of them are doing tasks: the font, the script library. Some of them
          are impostors: the analytics pixel whose only job is to remember you were here. The problem is that from the
          outside they look identical. Each one is a request leaving your browser, and each request carries your IP
          address and the page you are on.
        </p>

        <p>
          The three tools are three different ways of dealing with the crew.
        </p>

        <h2>Adblockers: the wanted poster</h2>

        <p>
          uBlock Origin, AdBlock Plus, and the rest work from filter lists. The big one is EasyList, which removes most
          adverts from international webpages and is maintained by four people (Fanboy, MonztA, Khrin, and Yuki2718)
          with help from a forum. Its sibling EasyPrivacy targets tracking rather than ads. A filter list is a long set
          of rules that match URLs and page elements. If a request matches a rule, the blocker stops it before it
          leaves your browser.
        </p>

        <p>
          This is the emergency meeting where everyone already has a list of names, and anyone on the list gets voted
          out on sight. It is fast, it is precise, and it kills ads dead, which is why almost everyone runs one. The
          weakness is the same as the strength. A tracker that is not on the poster walks straight past, and stays
          past until a maintainer notices and adds it.
        </p>

        <h2>Privacy Badger: the sus meter</h2>

        <p>
          Privacy Badger is made by the Electronic Frontier Foundation and it deliberately does not use a list. Their
          FAQ puts it plainly: they define what tracking looks like, and the extension blocks domains it observes
          tracking in the wild. What counts as a tracker depends on how a domain behaves, not on anybody&apos;s
          judgment.
        </p>

        <p>
          Concretely, it watches the third-party domains that embed images, scripts, and ads in the pages you visit,
          and it looks for the techniques trackers use: cookies that uniquely identify you, local storage
          &quot;supercookies&quot;, canvas fingerprinting. If it sees the same third-party host doing that on three
          separate sites, it stops loading anything from that host.
        </p>

        <Figure
          src="/blog/privacy-among-us/diagram-2.png"
          alt="Diagram showing a tracker domain observed on site A, site B, and site C, filling three strikes, then being blocked"
          caption="Three sites, three strikes. That is the whole rule."
        />

        <p>
          That is a sus meter. The crewmate is not on any list. It just keeps turning up in rooms it has no reason to
          be in, and after the third room the Badger calls the meeting and ejects it. Because this is learned rather
          than listed, it catches trackers nobody has written a rule for yet. The extension also ships with a head
          start: the EFF runs a training project called Badger Sett that visits thousands of popular sites and
          pre-learns the trackers on them, so a fresh install is not starting from zero.
        </p>

        <p>
          Two details I found interesting once I read the FAQ properly. First, Privacy Badger does not block ads for
          being ads. It only blocks the ones that track you, which is a deliberate choice to give advertisers a reason
          to behave. Second, it sends the Global Privacy Control and Do Not Track signals to every site, and if a
          tracker ignores them, the Badger learns to block it anyway. Social widgets like the Like button get replaced
          with a click-to-activate placeholder, so the button does not phone home until you press it.
        </p>

        <h2>LocalCDN: never leave the ship</h2>

        <p>
          This one handles a leak the other two mostly ignore. Sites do not host every script themselves. A huge
          number of them load jQuery, Bootstrap, Font Awesome, or a Google font from a shared content delivery
          network: ajax.googleapis.com, cdnjs.cloudflare.com, code.jquery.com, cdn.jsdelivr.net, unpkg.com. Those
          scripts are legitimate. They are crewmates doing tasks. But every time your browser fetches one, the CDN
          operator, which is usually Google or Cloudflare, gets your IP address and the page that asked for the file.
        </p>

        <p>
          An adblocker will not touch these because they are real code the page needs. Privacy Badger usually will not
          either, because serving a file is not tracking by its definition. So the request goes out, on every site,
          all day.
        </p>

        <Figure
          src="/blog/privacy-among-us/diagram-3.png"
          alt="Two flows. Without LocalCDN a request for jquery.min.js goes to cdnjs.cloudflare.com, which logs it. With LocalCDN the request is answered inside the browser and never leaves."
          caption="Same file either way. Only one of them tells Cloudflare about it."
        />

        <p>
          LocalCDN&apos;s own description is that it emulates content delivery networks. It intercepts the request,
          finds the same library at the same version in a bundle it ships with, and injects that instead. Nothing
          leaves. In the game: you never have to go to MIRA HQ to pick up your tools, because somebody already stocked
          the storage room on the ship.
        </p>

        <p>
          The old argument against this was that shared CDNs were good for speed, because a copy of jQuery cached
          from one site could be reused on the next. That argument is dead now. Browsers partition the HTTP cache by
          the site you are on, so a file cached from site A does not help on site B. Safari has done this since 2013,
          Chrome since version 86 in late 2020, and Firefox since version 85 in early 2021. The performance reason for
          public CDNs went away, and the privacy cost stayed.
        </p>

        <h2>Who catches what</h2>

        <p>
          Put the three side by side and the overlap is smaller than the &quot;install all of these&quot; advice
          suggests.
        </p>

        <ul>
          <li>
            The adblocker with EasyList and EasyPrivacy removes ads and every tracker that someone has already written
            a rule for. Most of them.
          </li>
          <li>
            Privacy Badger catches the trackers that are not on any list yet, and handles the social widgets and the
            opt-out signals.
          </li>
          <li>
            LocalCDN closes a specific leak, the shared library request, that neither of the others considers a
            problem.
          </li>
        </ul>

        <h2>What none of them do</h2>

        <p>
          They do nothing about the first party. The site you are on still sees everything you do on it, and the
          Privacy Badger FAQ is honest that this is out of scope. If you are logged in, that site knows who you are.
          They do not hide your IP address from anyone you actually connect to, which is what the VPN was for. And
          they can break things: Privacy Badger&apos;s placeholders exist because blocking a widget outright would
          leave a hole in the page, and LocalCDN can only substitute a library it has a copy of, so a site that uses
          an unusual version will fall back to the network anyway.
        </p>

        <p>
          What I run now is an adblocker with the two Easy lists, Privacy Badger, and LocalCDN on Firefox. Firefox&apos;s
          own tracking protection overlaps with the Badger, and the EFF says the two get along. I no longer think of
          them as a collection. I think of them as three crewmates who happen to be very good at three different
          tasks, and I have stopped expecting any one of them to do the other two&apos;s job.
        </p>

        <hr />

        <p className="text-sm">
          Sources: the{" "}
          <a href="https://privacybadger.org/#faq">Privacy Badger FAQ</a> (how it works, the three-site rule, Badger
          Sett, GPC and DNT, what counts as a third party);{" "}
          <a href="https://easylist.to/">EasyList</a> (maintainers, EasyPrivacy);{" "}
          <a href="https://www.localcdn.org/">LocalCDN</a> (supported CDNs and libraries);{" "}
          <a href="https://developer.chrome.com/blog/http-cache-partitioning">Chrome&apos;s HTTP cache partitioning announcement</a>{" "}
          and{" "}
          <a href="https://blog.mozilla.org/security/2021/01/26/supercookie-protections/">Mozilla&apos;s network partitioning post</a>{" "}
          (why shared CDN caching no longer helps).
        </p>

        <p className="text-sm neo-muted">
          Also on{" "}
          <a href="https://dev.to/stimmie/privacy-tools-explained-with-among-us-2j85">Dev.to</a>,{" "}
          <a href="https://stimmie.hashnode.dev/privacy-tools-explained-with-among-us">Hashnode</a>, and{" "}
          <a href="https://medium.com/@stimmieuwu/privacy-tools-explained-with-among-us-6d8a0bd4179c">Medium</a>{" "}
          (the short original). Source and images on{" "}
          <a href="https://github.com/smmariquit/stimmie.dev">GitHub</a>.
        </p>
      </article>
    </PageShell>
  );
}
