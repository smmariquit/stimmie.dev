"use client";
import Link from "next/link";
import MosaicBackground from "@/components/MosaicBackground";


export default function CasaPage() {
    return (
        <div className="min-h-screen bg-black w-full overflow-hidden">
            <MosaicBackground />

            {/* Blog post content overlay */}
            <div className="fixed inset-0 flex items-center justify-center pointer-events-none" style={{ zIndex: 50 }}>
                <div className="pointer-events-auto bg-gray-800/70 text-white p-6 rounded backdrop-blur-sm max-h-10/12 max-w-3xl w-full mx-4 overflow-y-auto no-scrollbar">
                    <div className="flex flex-col">
                        {/* Navigation */}
                        <div className="flex items-center gap-4 mb-6">
                            <Link href="/blog" className="text-white/70 hover:text-white transition-colors flex items-center gap-2">
                                <span>←</span>
                                <span>Blog</span>
                            </Link>
                            <span className="text-white/30">|</span>
                            <Link href="/" className="text-white/70 hover:text-white transition-colors">
                                Home
                            </Link>
                        </div>

                        {/* Article header */}
                        <article>

                            <header className="mb-8">
                                <h1 className="font-sans text-4xl sm:text-5xl font-black text-white mb-2">
                                    11 things I learned from my software engineering class
                                </h1>
                                <p className="text-white/50">May 22, 2026</p>
                            </header>

                            {/* Introduction */}
                            <section className="mb-8">
                                <p className="text-white/90 leading-relaxed">
                                    This sem, I had the pleasure to be the project manager, basically the big boss leader, of our software engineering course in UPLB. Over four months, we had to create a full enterprise app for student housing. It turned out to be one of the most chaotic but growth-heavy experience in my engineering career :)
                                </p>
                                <p className="text-white/90 leading-relaxed mt-5">
                                    Visit our app at <Link className="text-blue-400 hover:text-blue-300 underline transition-colors" href="https://uplb.casa">uplb.casa</Link>
                                </p>
                                <p className="text-white/90 leading-relaxed">
                                    See our repository at <Link className="text-blue-400 hover:text-blue-300 underline transition-colors" href="https://uplb.casa">github.com/smmariquit/comsci-128</Link>
                                </p>
                                <p className="text-white/90 leading-relaxed">
                                    Learn how to test it by reading our <Link className="text-blue-400 hover:text-blue-300 underline transition-colors" href="https://github.com/smmariquit/comsci-128/blob/main/betaTesting.md">beta testing guide</Link>
                                </p>
                            </section>

                            <section className="mt-8 pt-8 border-t border-white/10">
                                <h2 className="text-2xl font-bold text-white mb-4">How I was chosen</h2>
                                <p className="text-white/90 leading-relaxed">
                                    While (I think) all other classes had their leaders chosen by the instructor, our instructor opted to let the rest of my class choose. For some reason, my classmates didn't even have to stop and think. They just pointed at me. Perhaps that's because my batchmates in UPLB recognize me as some sort of expert. But for whatever reason, they trusted me with one of the biggest projects of their BSCS degree.
                                </p>
                            </section>
                            <section className="mt-8 pt-8 border-t border-white/10">
                                <h2 className="text-2xl font-bold text-white mb-4">Non-technical lessons</h2>
                                <p className="text-white/90 leading-relaxed mt-3">
                                    1. This is *the* biggest one. You cannot mask ineffective leadership by overcontribution. Yes, I was the one with the most commits, most hours worked. I could give myself a medal. But I realized that's not a good approach because I wasn't going out of my comfort zone. Coding for 14 hours, while difficult, is a difficulty I'm personally accustomed to. But confronting people for their shortcomings, motivating them, and keeping team morale is not something I'm naturally good at.
                                </p>
                                <p className="text-white/90 leading-relaxed mt-3">
                                    2. Make people *own* things instead of just handing out tasks. People will end up more passionate about their work if they fully understand why they are doing it and that it is their brainchild so to speak
                                </p>
                                <p className="text-white/90 leading-relaxed mt-3">
                                    3. The power of passive aggressive corpo-speak messages insanely underrated and works well to get people to do their shit LMAO
                                </p>
                                <p className="text-white/90 leading-relaxed mt-3">
                                    4. Since the last company I worked at was a startup (E-Konsulta), there's sort of a mindset where you should be quick, document as little as possible, and just do the work. When you're leading a bunch of people of which some have underdeveloped communication skills and work ethic, this is less applicable. You should consistently track the tasks of the team.
                                </p>
                                <p className="text-white/90 leading-relaxed mt-3">
                                    5. The beginning of any project is crucial in identifying who among in your team are the GOATs, the responsible people who take ownership, and those who tend to lag behind or procrastinate. Use this information wisely throughout the rest of the project to inform decisions on delegation.
                                </p>
                                <p className="text-white/90 leading-relaxed mt-3">
                                    6. Democracy is unreliable, and you shouldn't form a subteam without a leader.
                                </p>
                                {/*
                                <p className="text-white/90 leading-relaxed mt-3">
                                    7. Make people *own* things instead of just handing out tasks. People will end up more passionate about their work if they fully understand why they are doing it and that it is their brainchild so to speak
                                </p>
                                <p className="text-white/90 leading-relaxed mt-3">
                                    8. Make people *own* things instead of just handing out tasks. People will end up more passionate about their work if they fully understand why they are doing it and that it is their brainchild so to speak
                                </p>
                                <p className="text-white/90 leading-relaxed mt-3">
                                    9. Make people *own* things instead of just handing out tasks. People will end up more passionate about their work if they fully understand why they are doing it and that it is their brainchild so to speak
                                </p>
                                <p className="text-white/90 leading-relaxed mt-3">
                                    10. Make people *own* things instead of just handing out tasks. People will end up more passionate about their work if they fully understand why they are doing it and that it is their brainchild so to speak
                                </p>
                                <p className="text-white/90 leading-relaxed mt-3">

                                </p> */}
                            </section>

                            <section className="mt-8 pt-8 border-t border-white/10">
                                <h2 className="text-2xl font-bold text-white mb-4">Technical lessons</h2>
                                <p className="text-white/90 leading-relaxed">
                                    1. Clearly communicate to people that if you code something, you should be the one to ensure that it gets to production.
                                </p>
                                <p className="text-white/90 leading-relaxed">
                                    2. Clearly communicate the concept of code being stale and how a PR only has a life span before siya maging incompatiible with the current production code</p>

                                <p className="text-white/90 leading-relaxed">
                                    3. GitHub Dependabot and Security code scanning should definitely be used for production projects</p>
                                <p className="text-white/90 leading-relaxed">
                                    4. Keep GitHub PR templates simple. Don't add a huge checklist of tests that don't apply to the project.
                                </p>
                                <p className="text-white/90 leading-relaxed">
                                    5. It is NEVER too late to resolve technical debt. I had the choice of whether or not to resolve something 4 weeks before the final deadline. I chose not too because I thought it's too late, but that specific thing was a big headache for those four weeks.
                                </p>
                                {/* <p className="text-white/90 leading-relaxed">
                                    6. Keep GitHub PR templates simple. Don't add a huge checklist of tests that don't apply to the project.
                                </p>
                                <p className="text-white/90 leading-relaxed">
                                    7. Keep GitHub PR templates simple. Don't add a huge checklist of tests that don't apply to the project.
                                </p>
                                <p className="text-white/90 leading-relaxed">
                                    8. Keep GitHub PR templates simple. Don't add a huge checklist of tests that don't apply to the project.
                                </p>
                                <p className="text-white/90 leading-relaxed">
                                    9. Keep GitHub PR templates simple. Don't add a huge checklist of tests that don't apply to the project.
                                </p>
                                <p className="text-white/90 leading-relaxed">
                                    10. Keep GitHub PR templates simple. Don't add a huge checklist of tests that don't apply to the project.
                                </p> */}

                            </section>
                        </article>
                    </div>
                </div>
            </div>
        </div>
    );
}
