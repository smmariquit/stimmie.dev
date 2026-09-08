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

const TOOLS = [
  ["uBlock Origin", "https://ublockorigin.com/", "ublock"],
  ["AdBlock Plus", "https://adblockplus.org/", "adblockplus"],
  ["Privacy Badger", "https://privacybadger.org/", "privacy-badger"],
  ["LocalCDN", "https://www.localcdn.org/", "localcdn"],
  ["Decentraleyes", "https://decentraleyes.org/", "decentraleyes"],
];

function ToolRow() {
  return (
    <ul className="my-6 flex flex-wrap gap-x-6 gap-y-3 list-none p-0">
      {TOOLS.map(([name, href, icon]) => (
        <li key={icon} className="flex items-center gap-2">
          <img src={`/blog/privacy-among-us/logos/${icon}.png`} alt="" width="32" height="32" loading="lazy" />
          <a href={href}>{name}</a>
        </li>
      ))}
    </ul>
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
          two of them barely overlap. Among Us is how it finally made sense to me, so that is how I will explain it.
        </p>

        <p>
          A web page is not one place. When you open a news article, the article itself comes from the news site, but
          the fonts might come from Google, the script library from a CDN, the ads from an ad company, the comments
          from a company the news site contracted, and the Like button from Facebook. The{" "}
          <a href="https://www.eff.org/">Electronic Frontier Foundation</a>, the nonprofit that makes Privacy Badger,
          opens its own explanation of the tool with this same picture. To check it was not an exaggeration I loaded
          the <a href="https://www.inquirer.net/">Inquirer</a> front page once and logged every request. Four hosts
          were the Inquirer&apos;s own. Twenty-four were other companies, and they got fifty-four requests between
          them.
        </p>

        <table className="my-6 w-full text-sm border-collapse font-mono">
          <thead>
            <tr className="border-b-2 border-current text-left">
              <th className="py-1 pr-4">Company</th><th className="py-1 pr-4">Host</th><th className="py-1 text-right">Requests</th>
            </tr>
          </thead>
          <tbody>
            <tr className="border-t border-current"><td className="py-1 pr-4 align-top">Inquirer</td><td className="py-1 pr-4 break-all">www.inquirer.net</td><td className="py-1 text-right align-top">29</td></tr>
            <tr><td className="py-1 pr-4 align-top"></td><td className="py-1 pr-4 break-all">newsinfo.inquirer.net</td><td className="py-1 text-right align-top">2</td></tr>
            <tr><td className="py-1 pr-4 align-top"></td><td className="py-1 pr-4 break-all">sports.inquirer.net</td><td className="py-1 text-right align-top">1</td></tr>
            <tr><td className="py-1 pr-4 align-top"></td><td className="py-1 pr-4 break-all">analytics.inquirernetwork.net</td><td className="py-1 text-right align-top">2</td></tr>
            <tr className="border-t border-current"><td className="py-1 pr-4 align-top">Google</td><td className="py-1 pr-4 break-all">fundingchoicesmessages.google.com</td><td className="py-1 text-right align-top">14</td></tr>
            <tr><td className="py-1 pr-4 align-top"></td><td className="py-1 pr-4 break-all">securepubads.g.doubleclick.net</td><td className="py-1 text-right align-top">4</td></tr>
            <tr><td className="py-1 pr-4 align-top"></td><td className="py-1 pr-4 break-all">pagead2.googlesyndication.com</td><td className="py-1 text-right align-top">4</td></tr>
            <tr><td className="py-1 pr-4 align-top"></td><td className="py-1 pr-4 break-all">fonts.gstatic.com</td><td className="py-1 text-right align-top">3</td></tr>
            <tr><td className="py-1 pr-4 align-top"></td><td className="py-1 pr-4 break-all">ep2.adtrafficquality.google</td><td className="py-1 text-right align-top">3</td></tr>
            <tr><td className="py-1 pr-4 align-top"></td><td className="py-1 pr-4 break-all">fonts.googleapis.com</td><td className="py-1 text-right align-top">2</td></tr>
            <tr><td className="py-1 pr-4 align-top"></td><td className="py-1 pr-4 break-all">ep1.adtrafficquality.google</td><td className="py-1 text-right align-top">2</td></tr>
            <tr><td className="py-1 pr-4 align-top"></td><td className="py-1 pr-4 break-all">cm.g.doubleclick.net</td><td className="py-1 text-right align-top">1</td></tr>
            <tr><td className="py-1 pr-4 align-top"></td><td className="py-1 pr-4 break-all">www.google.com</td><td className="py-1 text-right align-top">1</td></tr>
            <tr><td className="py-1 pr-4 align-top"></td><td className="py-1 pr-4 break-all">64020afcac2374c8919c09a1eb7e6a94.safeframe.googlesyndication.com</td><td className="py-1 text-right align-top">1</td></tr>
            <tr className="border-t border-current"><td className="py-1 pr-4 align-top">iZooto</td><td className="py-1 pr-4 break-all">cdn.izooto.com</td><td className="py-1 text-right align-top">4</td></tr>
            <tr><td className="py-1 pr-4 align-top"></td><td className="py-1 pr-4 break-all">rec.izooto.com</td><td className="py-1 text-right align-top">1</td></tr>
            <tr><td className="py-1 pr-4 align-top"></td><td className="py-1 pr-4 break-all">sbp.izooto.com</td><td className="py-1 text-right align-top">1</td></tr>
            <tr><td className="py-1 pr-4 align-top"></td><td className="py-1 pr-4 break-all">nhwimp.izooto.com</td><td className="py-1 text-right align-top">1</td></tr>
            <tr className="border-t border-current"><td className="py-1 pr-4 align-top">New Relic</td><td className="py-1 pr-4 break-all">bam.nr-data.net</td><td className="py-1 text-right align-top">2</td></tr>
            <tr><td className="py-1 pr-4 align-top"></td><td className="py-1 pr-4 break-all">js-agent.newrelic.com</td><td className="py-1 text-right align-top">1</td></tr>
            <tr className="border-t border-current"><td className="py-1 pr-4 align-top">Yieldlove</td><td className="py-1 pr-4 break-all">cdn.yld.is</td><td className="py-1 text-right align-top">2</td></tr>
            <tr className="border-t border-current"><td className="py-1 pr-4 align-top">Criteo</td><td className="py-1 pr-4 break-all">static.criteo.net</td><td className="py-1 text-right align-top">1</td></tr>
            <tr><td className="py-1 pr-4 align-top"></td><td className="py-1 pr-4 break-all">gum.criteo.com</td><td className="py-1 text-right align-top">1</td></tr>
            <tr className="border-t border-current"><td className="py-1 pr-4 align-top">Adobe</td><td className="py-1 pr-4 break-all">use.typekit.net</td><td className="py-1 text-right align-top">1</td></tr>
            <tr className="border-t border-current"><td className="py-1 pr-4 align-top">io Technologies</td><td className="py-1 pr-4 break-all">cdn.onthe.io</td><td className="py-1 text-right align-top">1</td></tr>
            <tr className="border-t border-current"><td className="py-1 pr-4 align-top">OpenX</td><td className="py-1 pr-4 break-all">oa.openxcdn.net</td><td className="py-1 text-right align-top">1</td></tr>
            <tr className="border-t border-current"><td className="py-1 pr-4 align-top">RTB House</td><td className="py-1 pr-4 break-all">invstatic101.creativecdn.com</td><td className="py-1 text-right align-top">1</td></tr>
            <tr className="border-t border-current"><td className="py-1 pr-4 align-top">Adgebra</td><td className="py-1 pr-4 break-all">adgebra.co.in</td><td className="py-1 text-right align-top">1</td></tr>
          </tbody>
        </table>
        <p className="text-sm neo-muted font-mono">Every request the Inquirer front page made in one load on 5 September 2026, grouped by the company that received it.</p>

        <p>
          In Among Us terms, the site you typed in is the ship you boarded. Every third-party request is a crewmate
          who walked in from somewhere else. Some of them are doing tasks, like the font and the script library. Some
          of them are impostors, like the analytics pixel whose only job is to remember you were here. The problem is
          that from the outside they look identical. Each one is a request leaving your browser, and each request
          carries your IP address and the page you are on.
        </p>

        <p>The three tools are three different ways of dealing with the crew.</p>

        <h2>Adblockers</h2>

        <p>
          <a href="https://ublockorigin.com/">uBlock Origin</a>, <a href="https://adblockplus.org/">AdBlock Plus</a>,
          and the rest work from filter lists. The big one is <a href="https://easylist.to/">EasyList</a>, which
          removes most adverts from international webpages and is maintained by four people (
          <a href="https://github.com/ryanbr">Fanboy</a>, <a href="https://github.com/monzta">MonztA</a>,{" "}
          <a href="https://github.com/Khrin">Khrin</a>, and <a href="https://github.com/Yuki2718">Yuki2718</a>) with
          help from a forum. Its sibling <a href="https://easylist.to/easylist/easyprivacy.txt">EasyPrivacy</a>{" "}
          targets tracking rather than ads. A filter list is a long set of rules that match URLs and page elements. If
          a request matches a rule, the blocker stops it before it leaves your browser.
        </p>

        <p>
          In the game this is an emergency meeting where everyone already has a list of names, and anyone on the list
          gets voted out on sight. It works well enough that almost everyone runs one, and it takes the ads out of the
          page entirely rather than only stopping the tracking part. The catch is that a tracker not on the list walks
          straight past, and keeps walking past until a maintainer notices and adds it.
        </p>

        <p>
          If you can list everything, why is this not solved? Because the other side reads the list too, and the list
          is only ever a description of what the ads looked like last week. I pulled the{" "}
          <a href="https://github.com/easylist/easylist">EasyList commit history</a> while writing this. In the first
          week of September 2026 the repository got between 138 and 187 commits a day. On the 8th, by mid-afternoon
          UTC, Fanboy alone had pushed over ninety commits titled &quot;M: Update&quot;, roughly one every eight
          minutes since early morning.
        </p>

        <table className="my-6 text-sm border-collapse font-mono">
          <thead>
            <tr className="border-b-2 border-current text-left">
              <th className="py-1 pr-10">Day</th><th className="py-1 text-right">Commits to EasyList</th>
            </tr>
          </thead>
          <tbody>
            {[["1 Sep 2026", 174], ["2 Sep", 149], ["3 Sep", 138], ["4 Sep", 178], ["5 Sep", 173], ["6 Sep", 187], ["7 Sep", 185]].map(([d, n]) => (
              <tr key={d} className="border-t border-current"><td className="py-1 pr-10">{d}</td><td className="py-1 text-right">{n}</td></tr>
            ))}
          </tbody>
        </table>

        <p>
          Most of those commits look like this one. Three new domains, all random letters, added to the ad server
          list.
        </p>

        <pre className="text-sm"><code>{`+||donalpapmeat.com^
+||lzazqrrmqjvov.top^
+||wkzmfbrbxstzq.space^`}</code></pre>

        <p>
          Ad networks register throwaway domains faster than anyone can type them, which is why that one file has over
          forty thousand lines and grows every day. The other kind of commit goes the opposite direction. On the same
          day Khrin changed one existing rule because it was breaking something on CNN, and carved out an exception
          so the rule no longer applies to XMLHttpRequest calls.
        </p>

        <pre className="text-sm"><code>{`-||brightline.tv^$third-party
+||brightline.tv^$third-party,~xmlhttprequest`}</code></pre>

        <p>
          The bigger players do not bother with new domains. Facebook in 2019 started splitting the word Sponsored on
          its ads into scrambled pieces of text so that no rule could match it, the maintainers wrote rules for the
          scrambling, and Facebook changed it again. Some trackers get served from a subdomain of the site you are on,
          through a CNAME record that quietly points at the tracking company, so a blocker that only looks at the
          hostname sees a first-party request and lets it through. uBlock Origin on Firefox has been able to unmask
          those since 2020. When YouTube started blocking adblock users outright in late 2023, the lists were updated
          within hours, YouTube changed its detection, the lists were updated again, and that went on for weeks. A
          filter list is never finished. It works because a handful of people keep it current, every day, by hand.
        </p>

        <h2>Privacy Badger</h2>

        <p>
          <a href="https://privacybadger.org/">Privacy Badger</a> is made by the EFF and it deliberately does not use
          a list. Their <a href="https://privacybadger.org/#faq">FAQ</a> says: &quot;we define what tracking looks
          like, and then Privacy Badger blocks or restricts domains that it observes tracking in the wild.&quot;
          Whether something counts as a tracker depends on how the domain behaves, not on a maintainer&apos;s
          judgment.
        </p>

        <p>
          Concretely, it watches the third-party domains that embed images, scripts, and ads in the pages you visit,
          and it looks for the techniques trackers use: cookies that uniquely identify you, local storage
          &quot;supercookies&quot;, canvas fingerprinting. If it sees the same third-party host doing that on three
          separate sites, it stops loading anything from that host. Three is not many. I opened the{" "}
          <a href="https://www.inquirer.net/">Inquirer</a>, <a href="https://www.rappler.com/">Rappler</a>, and{" "}
          <a href="https://www.philstar.com/">Philstar</a> front pages in a row and these eight domains were on all
          of them.
        </p>

        <table className="my-6 w-full text-sm border-collapse font-mono">
          <thead>
            <tr className="border-b-2 border-current text-left">
              <th className="py-1 pr-4">Domain</th><th className="py-1 pr-4 text-right">Inquirer</th><th className="py-1 pr-4 text-right">Rappler</th><th className="py-1 text-right">Philstar</th>
            </tr>
          </thead>
          <tbody>
            {[
              ["googlesyndication.com", 5, 26, 43],
              ["doubleclick.net", 5, 27, 32],
              ["google.com", 15, 24, 22],
              ["adtrafficquality.google", 5, 5, 5],
              ["criteo.com", 1, 3, 9],
              ["creativecdn.com", 1, 3, 4],
              ["criteo.net", 1, 1, 1],
              ["openxcdn.net", 1, 1, 1],
            ].map(([host, a, b, c]) => (
              <tr key={host} className="border-t border-current">
                <td className="py-1 pr-4 break-all">{host}</td>
                <td className="py-1 pr-4 text-right">{a}</td>
                <td className="py-1 pr-4 text-right">{b}</td>
                <td className="py-1 text-right">{c}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="text-sm neo-muted font-mono">Front pages of the Inquirer, Rappler, and Philstar, loaded one after another on 5 September 2026. Each number is how many requests that domain received during the load.</p>

        <p>
          In the game this is the crewmate nobody reported, who keeps turning up in rooms it has no reason to be in,
          until after the third room the Badger calls the meeting and ejects it. Because the rule is about behaviour
          rather than a name, it catches trackers nobody has written a rule for yet. A fresh install does not start
          from zero either. The EFF runs a training project called Badger Sett that visits thousands of popular sites
          and pre-learns the trackers on them, and ships that with the extension.
        </p>

        <p>
          Privacy Badger does not block ads for being ads, only the ones that track you, which the EFF says is
          deliberate, to give advertisers a reason to behave. It also sends the Global Privacy Control and Do Not
          Track signals to every site, and if a tracker ignores them the Badger learns to block it anyway. Social
          widgets like the Like button get replaced with a click-to-activate placeholder, so the button does not phone
          home until you press it.
        </p>

        <h2>LocalCDN</h2>

        <p>
          The third tool deals with a leak that the other two mostly leave alone. Sites do not host every script
          themselves. A large share of them load jQuery, Bootstrap, Font Awesome, or a Google font from a shared
          content delivery network: ajax.googleapis.com, cdnjs.cloudflare.com, code.jquery.com, cdn.jsdelivr.net,
          unpkg.com. Those scripts are legitimate, crewmates doing tasks, but every time your browser fetches one, the
          CDN operator, usually Google or Cloudflare, gets your IP address and the page that asked for the file.
        </p>

        <p>
          An adblocker will not touch these because they are real code the page needs, and Privacy Badger usually
          will not either, because serving a file is not tracking by its definition. The request goes out on every
          site that uses the library, which is most of them.
        </p>

        <table className="my-6 w-full text-sm border-collapse font-mono">
          <thead>
            <tr className="border-b-2 border-current text-left">
              <th className="py-1 pr-4"></th><th className="py-1 pr-4">Without LocalCDN</th><th className="py-1">With LocalCDN</th>
            </tr>
          </thead>
          <tbody>
            <tr className="border-t border-current"><td className="py-1 pr-4">Page asks for</td><td className="py-1 pr-4 break-all">cdnjs.cloudflare.com/.../jquery.min.js</td><td className="py-1 break-all">cdnjs.cloudflare.com/.../jquery.min.js</td></tr>
            <tr className="border-t border-current"><td className="py-1 pr-4">Answered by</td><td className="py-1 pr-4">Cloudflare</td><td className="py-1">a copy bundled in the extension</td></tr>
            <tr className="border-t border-current"><td className="py-1 pr-4">Leaves your browser</td><td className="py-1 pr-4">yes</td><td className="py-1">no</td></tr>
            <tr className="border-t border-current"><td className="py-1 pr-4">Cloudflare learns</td><td className="py-1 pr-4">your IP, the page you were on</td><td className="py-1">nothing</td></tr>
          </tbody>
        </table>
        <p className="text-sm neo-muted font-mono">The same jQuery request with and without LocalCDN installed.</p>

        <p>
          <a href="https://www.localcdn.org/">LocalCDN</a> describes itself as emulating content delivery networks.
          It intercepts the request, finds the same library at the same version in a bundle it ships with, and injects
          that instead, so nothing leaves the browser. In the game you never have to go to MIRA HQ to pick up your
          tools, because somebody already stocked the storage room on the ship.
        </p>

        <p>
          If the name <a href="https://decentraleyes.org/">Decentraleyes</a> rings a bell, it is the same idea.
          LocalCDN started in 2020 as a fork of Decentraleyes with a longer list of libraries and CDNs, and it is the
          one that still gets regular updates. Run one or the other, never both, or they will fight over the same
          requests.
        </p>

        <p>
          The old argument against this was that shared CDNs were good for speed, because a copy of jQuery cached
          from one site could be reused on the next. That stopped being true a few years ago. Browsers now partition
          the HTTP cache by the site you are on, so a file cached from site A does not help on site B. Safari has done
          this since 2013, Chrome since version 86 in late 2020, and Firefox since version 85 in early 2021. What is
          left of the shared CDN is the privacy cost.
        </p>

        <hr />

        <p>
          Put the three side by side and the overlap is smaller than the &quot;install all of these&quot; advice
          suggests.
        </p>

        <ToolRow />

        <ul>
          <li>
            The adblocker with EasyList and EasyPrivacy removes ads and every tracker that someone has already written
            a rule for, which is most of them.
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

        <hr />

        <p>
          None of them do anything about the first party. The site you are on still sees everything you do on it, and
          the Privacy Badger FAQ says plainly that this is out of scope. If you are logged in, that site knows who you
          are. They do not hide your IP address from anyone you actually connect to, which is what the VPN was for.
          They can also break things. Privacy Badger&apos;s placeholders exist because blocking a widget outright
          would leave a hole in the page, and LocalCDN can only substitute a library it has a copy of, so a site that
          uses an unusual version will fall back to the network anyway.
        </p>

        <p>
          What I run now is an adblocker with the two Easy lists, Privacy Badger, and LocalCDN on Firefox. Firefox&apos;s
          own tracking protection overlaps with the Badger, and the EFF says the two get along. I stopped thinking of
          them as a collection once I understood that they are three crewmates doing three different tasks, and I
          have stopped expecting any one of them to do the other two&apos;s job.
        </p>

        <hr />

        <p className="text-sm">
          Sources: the{" "}
          <a href="https://privacybadger.org/#faq">Privacy Badger FAQ</a> (how it works, the three-site rule, Badger
          Sett, GPC and DNT, what counts as a third party);{" "}
          <a href="https://easylist.to/">EasyList</a> (maintainers, EasyPrivacy) and its <a href="https://github.com/easylist/easylist">GitHub repository</a> (commit counts and diffs, pulled 8 September 2026);{" "}
          <a href="https://www.localcdn.org/">LocalCDN</a> (supported CDNs and libraries);{" "}
          <a href="https://developer.chrome.com/blog/http-cache-partitioning">Chrome&apos;s HTTP cache partitioning announcement</a>{" "}
          and{" "}
          <a href="https://blog.mozilla.org/security/2021/01/26/supercookie-protections/">Mozilla&apos;s network partitioning post</a>{" "}
          (why shared CDN caching no longer helps).
        </p>

        <p className="text-sm neo-muted">
          Also on{" "}
          <a href="https://dev.to/stimmie/privacy-tools-explained-with-among-us-2j85">Dev.to</a>,{" "}
          <a href="https://daily.dev/posts/an-adblocker-privacy-badger-and-localcdn-do-three-different-jobs-real-request-logs-from-three-new-u8a8cu7z1">daily.dev</a>, and{" "}
          <a href="https://medium.com/@stimmieuwu/privacy-tools-explained-with-among-us-6d8a0bd4179c">Medium</a>{" "}
          (the short original). Source and images on{" "}
          <a href="https://github.com/smmariquit/stimmie.dev">GitHub</a>.
        </p>
      </article>
    </PageShell>
  );
}
