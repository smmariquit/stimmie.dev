import Link from "next/link";
import { projects } from "@/data/projects";
import { talks } from "@/data/talks";
import { flattenRedirects, getSitemapHosts } from "@/data/redirects";
import { fetchCloudflareRedirectEntries } from "@/lib/cloudflare";

export const metadata = {
  title: "Links Directory",
  description: "A directory of all pages and live Cloudflare redirects on stimmie.dev",
};

export default async function LinksDirectoryPage() {
  const localRedirects = flattenRedirects();
  
  // Fetch live entries from Cloudflare, just like the XML sitemap does
  const cfEntries = await fetchCloudflareRedirectEntries({
    hosts: getSitemapHosts(),
  });

  const staticRoutes = [
    { title: "Home", path: "/" },
    { title: "Talks", path: "/talks" },
    { title: "Projects", path: "/projects" },
    { title: "Blog", path: "/blog" },
    { title: "Archive", path: "/archive" },
  ];

  return (
    <main className="min-h-screen bg-[#0a0a0a] text-[#ededed] p-6 md:p-12 lg:p-24 font-sans">
      <div className="max-w-5xl mx-auto">
        <header className="mb-12">
          <Link href="/" className="text-blue-400 hover:text-blue-300 text-sm mb-4 inline-block transition-colors">
            ← Back to Home
          </Link>
          <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
            <div>
              <h1 className="text-4xl md:text-5xl font-black mb-2 tracking-tight text-white">Directory</h1>
              <p className="text-white/60">A map of every page and edge-level redirect on this site.</p>
            </div>
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-green-500/10 border border-green-500/20 text-green-400 text-xs font-bold">
              <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
              Live from Cloudflare
            </div>
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-12">
          <div className="lg:col-span-1 space-y-12">
            {/* Main Pages */}
            <section>
              <h2 className="text-xl font-bold mb-6 flex items-center gap-2">
                <span className="w-2 h-2 bg-blue-500 rounded-full"></span>
                Navigation
              </h2>
              <ul className="space-y-3">
                {staticRoutes.map((route) => (
                  <li key={route.path}>
                    <Link href={route.path} className="group flex items-center justify-between p-3 rounded-lg bg-white/5 border border-white/10 hover:bg-white/10 hover:border-white/20 transition-all">
                      <span className="font-medium">{route.title}</span>
                      <span className="text-white/30 text-xs group-hover:text-white/60 transition-colors">{route.path}</span>
                    </Link>
                  </li>
                ))}
              </ul>
            </section>

            {/* Projects */}
            <section>
              <h2 className="text-xl font-bold mb-6 flex items-center gap-2">
                <span className="w-2 h-2 bg-purple-500 rounded-full"></span>
                Projects
              </h2>
              <ul className="space-y-3">
                {projects.map((project) => (
                  <li key={project.slug}>
                    <Link href={`/projects/${project.slug}`} className="group flex items-center justify-between p-3 rounded-lg bg-white/5 border border-white/10 hover:bg-white/10 hover:border-white/20 transition-all">
                      <span className="font-medium truncate mr-4">{project.title}</span>
                      <span className="text-white/30 text-xs shrink-0 group-hover:text-white/60 transition-colors">/projects/{project.slug}</span>
                    </Link>
                  </li>
                ))}
              </ul>
            </section>

            {/* Talks */}
            <section>
              <h2 className="text-xl font-bold mb-6 flex items-center gap-2">
                <span className="w-2 h-2 bg-orange-500 rounded-full"></span>
                Talks
              </h2>
              <ul className="space-y-3">
                {talks.map((talk) => (
                  <li key={talk.slug}>
                    <Link href={`/talks/${talk.slug}`} className="group flex items-center justify-between p-3 rounded-lg bg-white/5 border border-white/10 hover:bg-white/10 hover:border-white/20 transition-all">
                      <span className="font-medium truncate mr-4">{talk.title}</span>
                      <span className="text-white/30 text-xs shrink-0 group-hover:text-white/60 transition-colors">/talks/{talk.slug}</span>
                    </Link>
                  </li>
                ))}
              </ul>
            </section>
          </div>

          <div className="lg:col-span-2">
            {/* Live Cloudflare Redirects */}
            <section>
              <h2 className="text-xl font-bold mb-6 flex items-center gap-2">
                <span className="w-2 h-2 bg-green-500 rounded-full"></span>
                Live Edge Redirects
              </h2>
              
              {cfEntries.length === 0 ? (
                <div className="p-8 rounded-2xl border border-dashed border-white/10 text-center">
                  <p className="text-white/40 text-sm">No live Cloudflare redirects found or API unreachable.</p>
                  <p className="text-white/20 text-xs mt-2 italic">Check your CLOUDFLARE_API_TOKEN in Vercel/Local Env.</p>
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {cfEntries.map((entry, idx) => {
                    const localMatch = localRedirects.find(l => 
                      l.sources.some(s => `https://${s.host}${s.path}` === entry.url)
                    );

                    return (
                      <div key={idx} className="p-4 rounded-xl bg-white/5 border border-white/10 hover:border-white/20 transition-all flex flex-col gap-2">
                        <div className="flex items-center justify-between">
                          <Link href={entry.url} target="_blank" className="text-blue-400 hover:underline font-bold text-sm truncate">
                            {entry.url.replace("https://", "")}
                          </Link>
                          {localMatch?.category && (
                            <span className="text-[10px] uppercase font-black tracking-widest text-white/30">
                              {localMatch.category}
                            </span>
                          )}
                        </div>
                        
                        <div className="flex flex-col gap-1">
                          <div className="flex items-center justify-between text-[10px] text-white/40">
                            <span>Status</span>
                            <span className="text-green-500 font-bold">ACTIVE @ EDGE</span>
                          </div>
                          <div className="flex items-center justify-between text-[10px] text-white/40">
                            <span>Last Synced</span>
                            <span>{new Date(entry.lastModified).toLocaleDateString()}</span>
                          </div>
                        </div>

                        {localMatch && (
                          <div className="mt-2 pt-2 border-t border-white/5">
                            <div className="flex items-center justify-between">
                              <span className="text-[10px] text-white/30 uppercase font-bold">Target</span>
                              <Link 
                                href={localMatch.target} 
                                target="_blank"
                                className="text-[10px] text-white/60 hover:text-white transition-colors truncate max-w-[180px]"
                              >
                                {localMatch.target}
                              </Link>
                            </div>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}

              {/* Pending Local Sync */}
              {localRedirects.length > 0 && (
                <div className="mt-12">
                  <h3 className="text-sm font-black uppercase tracking-widest text-white/40 mb-4">
                    Locally Configured (Pending Sync)
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 opacity-60 grayscale hover:opacity-100 hover:grayscale-0 transition-all">
                    {localRedirects
                      .filter(l => !cfEntries.some(e => l.sources.some(s => `https://${s.host}${s.path}` === e.url)))
                      .map((item, idx) => (
                        <div key={idx} className="p-4 rounded-xl bg-white/5 border border-white/5 border-dashed flex flex-col gap-2">
                          <div className="flex items-center justify-between">
                            <span className="font-bold text-sm">{item.slug}</span>
                            <span className="text-[10px] bg-white/10 text-white/40 px-1.5 py-0.5 rounded uppercase font-bold">
                              {item.placeholder ? "Draft" : "Pending Sync"}
                            </span>
                          </div>
                          <div className="text-[10px] text-white/30 italic">
                            Waiting for next `npm run cf:migrate-redirects`
                          </div>
                        </div>
                      ))
                    }
                    {localRedirects.filter(l => !cfEntries.some(e => l.sources.some(s => `https://${s.host}${s.path}` === e.url))).length === 0 && (
                      <p className="text-white/20 text-xs italic">All local redirects are in sync with Cloudflare.</p>
                    )}
                  </div>
                </div>
              )}
            </section>
          </div>
        </div>

        <footer className="mt-24 pt-12 border-t border-white/10 flex flex-col md:flex-row items-center justify-between gap-4">
          <p className="text-white/30 text-xs">
            &copy; {new Date().getFullYear()} stimmie.dev • Edge data refreshed hourly.
          </p>
          <div className="flex gap-6">
            <Link href="/sitemap.xml" className="text-white/30 hover:text-white text-xs transition-colors">
              XML Sitemap
            </Link>
          </div>
        </footer>
      </div>
    </main>
  );
}
