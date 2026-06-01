"use client";
import Link from "next/link";
import MosaicBackground from "@/components/MosaicBackground";

export default function JuneYourPage() {
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
                  Parkour 🤸‍♂️
                </h1>
                <p className="text-white/50">June 1, 2026</p>
              </header>

              {/* Introduction */}
              <section className="mb-8 space-y-4">
                <p className="text-white/90 leading-relaxed">
                  will not bother your feed with a long list of achievements, you can read that here: blog.stimmie.dev/juneyour. #1 on that list is, of course, dyeing my hair brown.
                </p>
                <p className="text-white/90 leading-relaxed">
                  but yes, massive parkour from my previous grades to getting stellar grades in one of the hardest semesters in an Elbi CS degree. i think the shift happened because I finally stopped just mindlessly consuming opportunities and started learning, leading, and building with actual intention.
                </p>
              </section>

              {/* A few things */}
              <section className="mb-8 space-y-4">
                <p className="text-white/90 leading-relaxed">a few things that actually made this possible:</p>
                <ul className="list-disc list-inside text-white/90 leading-relaxed space-y-2 ml-4">
                  <li><span className="font-bold">focus!</span> realizing you can either be really good at a few things or really meh at a lot of things.</li>
                  <li><span className="font-bold">sleep:</span> prioritizing rest, despite what your local hustle-bro/sigma male might tell you.</li>
                  <li><span className="font-bold">resources:</span> honestly, having access to better internet, compute, and physical spaces. sorry not sorry, but I performed my worst when I was staying in a UP dorm. environment matters.</li>
                  <li><span className="font-bold">people:</span> finally felt that im not alone and that my presence is appreciated. i never passed on the opportunity to bond with the people dear to me because im "too busy achieving". you know who you are :))</li>
                </ul>
              </section>

              {/* Thoughts */}
              <section className="mb-8 space-y-4">
                <p className="text-white/90 leading-relaxed">
                  beyond the logistics, there's this weird notion in tech where you have to pick a side: either you build "real-world" software or you get good grades. manosphere college is a scam type sh. but as someone who cares about being an excellent engineer, this sem was about proving that the theory and the execution belong together.
                </p>
                <p className="text-white/90 leading-relaxed">
                  I actually care about Computer Science! not just as a career path, a stack of technologies, or because I'm just after any degree to increase my chances of getting hired, but as a discipline that demands rigor.
                </p>
                <p className="text-white/90 leading-relaxed">
                  I guess I just disagree with the hyper-individualistic idea that we should drop our acads, vibe code, and chase a ton of money with zero regard for morality or civic duty. Digital infrastructure will be core to public service moving forward, and looking at all the contractor and corruption shite we deal with, I don't think we have room left for mediocrity.
                </p>
                <p className="text-white/90 leading-relaxed">
                  so yes i am SO ready for senior year! my passion is at an all time high :))
                </p>
                <p className="text-white/90 leading-relaxed font-semibold">
                  live a balanced life and do what you REALLY want.
                </p>
              </section>

              {/* Achievements list */}
              <section className="mt-12 pt-8 border-t border-white/10">
                <h2 className="text-2xl font-bold text-white mb-6">The Breakdown</h2>
                <p className="text-white/90 leading-relaxed mb-6">
                  If you're curious about what I achieved this semester (roughly from January until now), here is the breakdown! Included are all the slides, links, and resources included so you can hopefully grab the same opportunities for yourself:
                </p>
                
                <ul className="space-y-4">
                  {achievements.map((item, idx) => (
                    <li key={idx} className="bg-white/5 p-4 rounded text-white/80 leading-relaxed">
                      {item}
                    </li>
                  ))}
                </ul>
              </section>
            </article>
          </div>
        </div>
      </div>
    </div>
  );
}

const achievements = [
  <>Colored my hair brown</>,
  <>Got my 5 kilometer run time down to 28m 16s</>,
  <>Served as the HOST for Google Developer Groups: Build with AI Manila in collaboration with Data Engineering Pilipinas</>,
  <>Created <a href="https://room-tba.stimmie.dev" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">https://room-tba.stimmie.dev</a> and <a href="https://gradesim.stimmie.dev" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">https://gradesim.stimmie.dev</a></>,
  <>Delivered a workshop about machine learning to applicants of our organization <a href="https://workshops.stimmie.dev/ml-iris-slides" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">https://workshops.stimmie.dev/ml-iris-slides</a></>,
  <>Somehow hosted a Cubao karaoke session with people in tech... and it worked <a href="https://www.facebook.com/share/p/1EFyPpVhpb/" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">https://www.facebook.com/share/p/1EFyPpVhpb/</a></>,
  <>Led two meetups for Sip & Scale in UPLB!</>,
  <>Early-sem side quest: won in UPLB COSS' Web Design Competition</>,
  <>Got nominated as one of the best CMSC 173 projects this sem! <a href="https://kimiroutes.stimmie.dev" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">https://kimiroutes.stimmie.dev</a></>,
  <>Led one of the biggest projects in a Elbi CS degree, a section of 20 people, skilling the team with advanced Git (Merkle trees, etc), Javascript, React fundamentals, Ci/CD, and Agile</>,
  <>Talked about Python in research at UST <a href="https://workshops.stimmie.dev/" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">https://workshops.stimmie.dev/</a></>,
  <>Contributed to the Online Encyclopedia of Integer Sequences <a href="https://oeis.org/A322054" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">https://oeis.org/A322054</a> - math itself</>,
  <>Got invited as a guest speaker TWICE to talk about Minecraft, my favorite game, and how it relates to engineering lessons! <a href="https://workshops.stimmie.dev/minecraft" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">https://workshops.stimmie.dev/minecraft</a> - <a href="https://www.youtube.com/watch?v=Ko2HDZOv5PY&t=182s" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">https://www.youtube.com/watch?v=Ko2HDZOv5PY&t=182s</a></>,
  <>Established a student-led Discord server for my Institute</>,
  <>Built Room TBA - <a href="https://room-tba.stimmie.dev" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">https://room-tba.stimmie.dev</a></>,
  <>Organized and served as an official mentor for The Innovation Lab 2026, the inaugural hackathon of my CS org</>,
  <>Served as an official mentor and judge for the data hackathon of DataCamp and DOST SEI via START - DOST</>,
  <>Served as a guest speaker for Batangas Eastern College, giving a crash course on presenting insights better (ever since i was a kid i've wanted to transform raw data into actionable business decisions) <a href="https://workshops.stimmie.dev/present-insights-better" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">workshops.stimmie.dev/present-insights-better</a></>,
  <>Launched my minecraft portfolio at <a href="https://minecraft.stimmie.dev" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">https://minecraft.stimmie.dev</a> and started to host a fully functioning Minecraft server for UPLB Batch 2026</>,
  <>Insanely random side quest, but out of 600 participants we were the few teams who was a finalist in the Ateneo Blue Case Competition??? <a href="https://www.facebook.com/share/p/18upyhm524/" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">https://www.facebook.com/share/p/18upyhm524/</a></>,
  <>Actually learned C++, competitive programming, and placed 11th out of ~50+ teams nationwide in Algolympics 2026</>,
  <>Partnered Pizza & Friends <a href="https://joinpizza.fun" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">https://joinpizza.fun</a> with GitHub to host a very goofy, non-serious hackathon and slides party, a break from all the sigma vibes of tech</>,
  <>Successfully replaced the LED screen of my friend's phone... guess this can be my backup job!</>,
  <>Spent more than 40 hours studying, preparing for, and facilitating UPLB Data Science Guild's Python and R workshop for professionals - <a href="https://www.facebook.com/share/p/1EBviYF4vK/" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">https://www.facebook.com/share/p/1EBviYF4vK/</a></>,
  <>Completed the DOST Scholars' Leadership Camp, a 5-day intensive bootcamp on ______</>,
  <>Volunteered some for Pahinungod</>
];
