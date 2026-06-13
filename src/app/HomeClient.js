// src/app/HomeClient.js

"use client";
import Image from "next/image";
import Link from "next/link";
import { GitHubCalendar } from 'react-github-calendar';
import { talks } from "@/data/talks";
import { projects } from "@/data/projects";
import { socialCategories } from "@/data/socials";
import { blogPosts } from "@/data/blogs";

export default function HomeClient({ mediaData }) {
  const { film, book, anime } = mediaData || {};

  return (
    <div className="min-h-screen bg-[#050014] text-gray-200 font-sans selection:bg-pink-500 selection:text-white relative" lang="en">
      
      {/* Background Glows */}
      <div className="absolute top-0 left-0 w-full h-[500px] bg-purple-900/20 blur-[120px] pointer-events-none rounded-full" />
      <div className="absolute top-[20%] right-0 w-[600px] h-[600px] bg-pink-900/10 blur-[150px] pointer-events-none rounded-full" />
      <div className="absolute bottom-0 left-1/4 w-[800px] h-[400px] bg-indigo-900/20 blur-[150px] pointer-events-none rounded-full" />

      {/* The Stimmieverse Main Navigation */}
      <nav className="fixed top-0 w-full px-6 py-4 md:px-8 md:py-6 flex flex-col md:flex-row md:justify-between md:items-center bg-[#050014]/80 backdrop-blur-lg z-50 border-b border-purple-900/30 gap-4">
        <div className="font-bold tracking-widest text-sm uppercase flex-shrink-0 text-transparent bg-gradient-to-r from-pink-500 to-indigo-400 bg-clip-text drop-shadow-[0_0_8px_rgba(236,72,153,0.3)]">
          STIMMIE // CREATOR
        </div>
        <div className="flex gap-6 font-medium text-xs tracking-widest uppercase overflow-x-auto whitespace-nowrap scrollbar-hide w-full md:w-auto pb-2 md:pb-0">
           <span className="text-purple-500/50 hidden lg:inline">THE STIMMIEVERSE //</span>
           {[
                { name: 'Kape', url: 'https://kape.stimmie.dev' },
                { name: 'Room TBA', url: 'https://room-tba.stimmie.dev' },
                { name: 'GradeSim', url: 'https://gradesim.stimmie.dev' },
                { name: 'The Crib', url: 'https://crib.stimmie.dev' },
                { name: 'Atlas', url: 'https://atlas-of-my-skies.stimmie.dev' },
                { name: 'Work', url: '/projects', local: true },
                { name: 'Writing', url: '/blog', local: true },
                { name: 'Archive', url: '/v2', local: true },
              ].map((site) => (
                <Link 
                  key={site.name} 
                  href={site.url} 
                  target={site.local ? "_self" : "_blank"}
                  rel={site.local ? "" : "noopener noreferrer"}
                  className="hover:text-cyan-400 text-gray-400 transition-colors flex-shrink-0"
                >
                  {site.name} {!site.local && '↗'}
                </Link>
           ))}
        </div>
      </nav>

      {/* Hero Section */}
      <header className="pt-40 pb-16 px-6 md:px-12 max-w-7xl mx-auto relative z-10">
        <h1 className="text-5xl md:text-8xl lg:text-9xl font-black uppercase leading-[0.85] tracking-tighter bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400 text-transparent bg-clip-text mb-8 drop-shadow-lg">
          CRAFTING<br/>
          DIGITAL<br/>
          WORLDS.
        </h1>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 border-t border-purple-900/30 pt-8 mt-16">
          <div className="md:col-span-1">
            <h2 className="text-xs font-bold uppercase tracking-widest text-indigo-400 mb-4">[ ABOUT ]</h2>
            <p className="text-sm md:text-base leading-relaxed text-indigo-100/70">
              Creator, tinkerer, and builder of digital experiences. 
              Whether it's writing code, designing interfaces, or crafting data stories, I love bringing wild ideas to life on the internet.
            </p>
          </div>
          <div className="md:col-span-2">
            <h2 className="text-xs font-bold uppercase tracking-widest text-pink-400 mb-4">[ SOCIALS ]</h2>
            <div className="flex flex-wrap gap-x-6 gap-y-4">
              {socialCategories.flatMap(cat => cat.links)
                .filter(link => !['Hackathon Guide', 'Freshie Resources'].includes(link.name))
                .map((link) => {
                  const Wrapper = link.href ? Link : 'div';
                  return (
                    <Wrapper 
                      key={link.name} 
                      href={link.href || undefined} 
                      className={`flex items-center gap-2 text-sm md:text-base font-medium transition-colors ${link.href ? 'hover:text-pink-400' : 'text-gray-400'}`}
                      title={link.alt || link.name}
                    >
                      {link.icon && <img src={link.icon} alt="" className="w-4 h-4 object-contain hover:scale-110 transition-transform" />}
                      {link.name}
                    </Wrapper>
                  );
                })}
            </div>
          </div>
          <div className="md:col-span-1 border-t md:border-t-0 md:border-l border-purple-900/30 pt-8 md:pt-0 md:pl-8">
            <h2 className="text-xs font-bold uppercase tracking-widest text-cyan-400 mb-4">[ RESOURCES ]</h2>
            <div className="flex flex-col gap-5">
              {socialCategories.flatMap(cat => cat.links)
                .filter(link => ['Hackathon Guide', 'Freshie Resources'].includes(link.name))
                .map((link) => (
                <Link 
                  key={link.name} 
                  href={link.href || '#'} 
                  className="group flex items-start gap-3"
                  title={link.alt || link.name}
                >
                  {link.icon && <img src={link.icon} alt="" className="w-5 h-5 object-contain mt-0.5 group-hover:scale-110 transition-transform" />}
                  <div>
                    <span className="text-sm md:text-base font-bold text-gray-200 group-hover:text-cyan-400 transition-colors block leading-tight">{link.name}</span>
                    <span className="text-xs text-cyan-600/70 group-hover:text-cyan-400 font-mono mt-1 block">↗ Read</span>
                  </div>
                </Link>
              ))}
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 md:px-12 pb-32 space-y-32 relative z-10">
        
        {/* Selected Works - Editorial Grid */}
        <section>
          <div className="flex justify-between items-end mb-12">
            <h2 className="text-3xl md:text-5xl font-black uppercase tracking-tighter text-white">Selected Works</h2>
            <Link href="/projects" className="text-sm uppercase tracking-widest font-bold text-pink-500 hover:text-pink-400">View All →</Link>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-12 gap-6 md:gap-8">
            {projects.map((p, idx) => (
              <Link 
                key={p.slug} 
                href={`/projects/${p.slug}`} 
                className={`group block overflow-hidden ${
                  idx % 3 === 0 ? 'md:col-span-12' : 'md:col-span-6'
                }`}
              >
                <div className="relative aspect-[16/9] md:aspect-[3/2] bg-indigo-950/30 overflow-hidden rounded-lg border border-indigo-500/10 group-hover:border-pink-500/50 transition-colors duration-500">
                  <Image 
                    src={p.src} 
                    alt={p.title} 
                    width={1200} 
                    height={800} 
                    className="object-cover w-full h-full group-hover:scale-105 transition-transform duration-700 ease-out" 
                  />
                  <div className="absolute inset-0 bg-indigo-900/10 group-hover:bg-transparent transition-colors duration-500 mix-blend-overlay" />
                </div>
                <div className="mt-4 flex justify-between items-start">
                  <h3 className="text-xl md:text-2xl font-bold uppercase tracking-tight group-hover:text-cyan-400 transition-colors">{p.title}</h3>
                  <span className="text-xs font-bold text-pink-500">0{idx + 1}</span>
                </div>
              </Link>
            ))}
          </div>
        </section>

        {/* Talks & Workshops - Horizontal Scroll/List */}
        <section>
          <div className="flex justify-between items-end mb-12">
            <h2 className="text-3xl md:text-5xl font-black uppercase tracking-tighter text-white">Talks</h2>
            <Link href="/talks" className="text-sm uppercase tracking-widest font-bold text-pink-500 hover:text-pink-400">View All →</Link>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {talks.map((t, idx) => {
              const displayImage = t.actionPhoto || t.slidesThumbnail;
              return (
                <Link key={t.slug} href={`/talks/${t.slug}`} className="group block">
                  <div className="relative aspect-square overflow-hidden bg-indigo-950/30 rounded-lg border border-indigo-500/10 group-hover:border-cyan-500/50 transition-colors duration-500 mb-4">
                    {displayImage && (
                      <Image 
                        src={displayImage} 
                        alt={t.title} 
                        width={600} 
                        height={600} 
                        className="object-cover w-full h-full group-hover:scale-105 transition-transform duration-700 ease-out" 
                      />
                    )}
                  </div>
                  <h3 className="text-base font-bold uppercase tracking-tight leading-tight group-hover:text-pink-400 transition-colors">{t.title}</h3>
                </Link>
              );
            })}
          </div>
        </section>

        {/* Writing & GitHub */}
        <section className="grid grid-cols-1 md:grid-cols-2 gap-16 border-t border-purple-900/30 pt-16">
          <div>
            <h2 className="text-sm font-bold uppercase tracking-widest text-indigo-400 mb-8">[ WRITING ]</h2>
            <div className="space-y-8">
              {blogPosts.slice(0, 4).map((post, idx) => (
                <Link key={idx} href={`/blog/${post.slug || 'slug'}`} className="group block">
                  <span className="text-xs text-pink-500/70 font-mono mb-2 block">{post.date}</span>
                  <h3 className="text-2xl font-bold tracking-tight group-hover:text-cyan-400 transition-colors">{post.title}</h3>
                </Link>
              ))}
            </div>
          </div>
          <div>
            <h2 className="text-sm font-bold uppercase tracking-widest text-pink-400 mb-8">[ ACTIVITY ]</h2>
            <div className="bg-indigo-950/20 p-8 rounded-xl border border-indigo-500/10 overflow-x-auto hover:border-pink-500/30 transition-colors">
              {/* GitHubCalendar natively supports a theme prop to colorize blocks, but we'll let it use dark mode for now */}
              <GitHubCalendar username="smmariquit" colorScheme="dark" blockSize={12} blockMargin={4} fontSize={14} />
            </div>
          </div>
        </section>

        {/* Media & Button */}
        <section className="border-t border-purple-900/30 pt-16">
          <div className="max-w-4xl">
            <h2 className="text-sm font-bold uppercase tracking-widest text-cyan-400 mb-8">[ RECENTLY CONSUMED ]</h2>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-8">
              {/* Film */}
              {film && (
                <div className="space-y-4">
                  <h3 className="text-xs uppercase tracking-widest text-indigo-400/80 border-b border-purple-900/30 pb-2">Film</h3>
                  <div className="aspect-[2/3] relative overflow-hidden bg-indigo-950/30 rounded-lg border border-indigo-500/10">
                    {film.posterUrl && <img src={film.posterUrl} alt={film.title} className="object-cover w-full h-full" />}
                  </div>
                  <div>
                    <h4 className="font-bold text-sm uppercase text-pink-100">{film.title}</h4>
                    {film.rating && <p className="text-xs text-pink-400 mt-1">{'★'.repeat(Math.round(film.rating))}</p>}
                  </div>
                </div>
              )}
              {/* Book */}
              {book && (
                <div className="space-y-4">
                  <h3 className="text-xs uppercase tracking-widest text-indigo-400/80 border-b border-purple-900/30 pb-2">Book</h3>
                  <div className="aspect-[2/3] relative overflow-hidden bg-indigo-950/30 rounded-lg border border-indigo-500/10">
                    {book.coverUrl && <img src={book.coverUrl} alt={book.title} className="object-cover w-full h-full" />}
                  </div>
                  <div>
                    <h4 className="font-bold text-sm uppercase text-pink-100">{book.title}</h4>
                    <p className="text-xs text-cyan-400/70 mt-1">{book.author}</p>
                  </div>
                </div>
              )}
              {/* Anime */}
              {anime && (
                <div className="space-y-4">
                  <h3 className="text-xs uppercase tracking-widest text-indigo-400/80 border-b border-purple-900/30 pb-2">Anime</h3>
                  <div className="aspect-[2/3] relative overflow-hidden bg-indigo-950/30 rounded-lg border border-indigo-500/10">
                    {anime.coverUrl && <img src={anime.coverUrl} alt={anime.title} className="object-cover w-full h-full" />}
                  </div>
                  <div>
                    <h4 className="font-bold text-sm uppercase text-pink-100">{anime.title}</h4>
                    <p className="text-xs text-cyan-400/70 mt-1 uppercase">{anime.status}</p>
                  </div>
                </div>
              )}
            </div>
          </div>
          
          <div className="mt-24 text-center">
             <p className="text-xs text-indigo-400/80 mb-4 uppercase tracking-widest">Grab my button</p>
             <img src="/stimmie_88x31.gif" alt="Stimmie 88x31 Button" className="inline-block rounded shadow-[0_0_15px_rgba(236,72,153,0.3)] hover:scale-110 transition-transform" style={{ imageRendering: 'pixelated' }} />
          </div>
        </section>

      </main>
      
      {/* Global CSS injections for scrollbar hiding */}
      <style dangerouslySetInnerHTML={{__html: `
        .scrollbar-hide::-webkit-scrollbar {
            display: none;
        }
        .scrollbar-hide {
            -ms-overflow-style: none;
            scrollbar-width: none;
        }
      `}} />
    </div>
  );
}
