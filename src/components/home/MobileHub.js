"use client";

import Link from "next/link";
import { useCallback, useState } from "react";
import { STIMMIEVERSE_LINKS } from "./constants";
import {
  ActivityBlock,
  Button88x31,
  MediaSection,
  ProjectsGrid,
  TalksGrid,
  WritingList,
} from "./SectionPreviews";
import SnapshotStrip from "./SnapshotStrip";

const TABS = [
  { id: "overview", label: "Overview" },
  { id: "work", label: "Work" },
  { id: "talks", label: "Talks" },
  { id: "more", label: "More" },
];

export default function MobileHub({
  projects,
  talks,
  mediaData,
  featuredProject,
  featuredTalk,
}) {
  const [activeTab, setActiveTab] = useState("overview");

  const handleKeyDown = useCallback(
    (e) => {
      const currentIndex = TABS.findIndex((t) => t.id === activeTab);
      if (e.key === "ArrowRight") {
        e.preventDefault();
        setActiveTab(TABS[(currentIndex + 1) % TABS.length].id);
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        setActiveTab(TABS[(currentIndex - 1 + TABS.length) % TABS.length].id);
      }
    },
    [activeTab],
  );

  return (
    <div className="md:hidden relative z-10 pb-16">
      <div
        role="tablist"
        aria-label="Homepage sections"
        className="sticky top-[57px] z-40 mx-6 flex gap-1 p-1 bg-[#050014]/90 backdrop-blur-lg border border-purple-900/30 rounded-lg"
        onKeyDown={handleKeyDown}
      >
        {TABS.map((tab) => (
          <button
            key={tab.id}
            type="button"
            role="tab"
            id={`tab-${tab.id}`}
            aria-selected={activeTab === tab.id}
            aria-controls={`panel-${tab.id}`}
            tabIndex={activeTab === tab.id ? 0 : -1}
            onClick={() => setActiveTab(tab.id)}
            className={`flex-1 py-2 px-1 text-[10px] font-bold uppercase tracking-widest rounded-md transition-colors ${
              activeTab === tab.id
                ? "bg-purple-900/40 text-pink-400"
                : "text-gray-500 hover:text-gray-300"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <div className="px-6 pt-4">
        {activeTab === "overview" && (
          <div
            role="tabpanel"
            id="panel-overview"
            aria-labelledby="tab-overview"
          >
            <SnapshotStrip
              variant="grid"
              featuredProject={featuredProject}
              featuredTalk={featuredTalk}
              mediaData={mediaData}
            />
          </div>
        )}

        {activeTab === "work" && (
          <div role="tabpanel" id="panel-work" aria-labelledby="tab-work">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-sm font-bold uppercase tracking-widest text-pink-400">
                [ WORK ]
              </h2>
              <Link
                href="/projects"
                className="text-xs uppercase tracking-widest font-bold text-cyan-400"
              >
                View All →
              </Link>
            </div>
            <ProjectsGrid projects={projects} compact />
          </div>
        )}

        {activeTab === "talks" && (
          <div role="tabpanel" id="panel-talks" aria-labelledby="tab-talks">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-sm font-bold uppercase tracking-widest text-cyan-400">
                [ TALKS ]
              </h2>
              <Link
                href="/talks"
                className="text-xs uppercase tracking-widest font-bold text-cyan-400"
              >
                View All →
              </Link>
            </div>
            <TalksGrid talks={talks} compact />
          </div>
        )}

        {activeTab === "more" && (
          <div role="tabpanel" id="panel-more" aria-labelledby="tab-more">
            <div className="space-y-8">
              <div>
                <h2 className="text-sm font-bold uppercase tracking-widest text-indigo-400 mb-4">
                  [ WRITING ]
                </h2>
                <WritingList />
                <Link
                  href="/blog"
                  className="inline-block mt-4 text-xs uppercase tracking-widest font-bold text-pink-500"
                >
                  View All →
                </Link>
              </div>

              <div>
                <h2 className="text-sm font-bold uppercase tracking-widest text-pink-400 mb-4">
                  [ ACTIVITY ]
                </h2>
                <ActivityBlock compact />
              </div>

              <div>
                <h2 className="text-sm font-bold uppercase tracking-widest text-cyan-400 mb-4">
                  [ RECENTLY CONSUMED ]
                </h2>
                <MediaSection mediaData={mediaData} compact />
              </div>

              <div>
                <h2 className="text-sm font-bold uppercase tracking-widest text-purple-400 mb-4">
                  [ THE STIMMIEVERSE ]
                </h2>
                <div className="flex flex-col gap-3">
                  {STIMMIEVERSE_LINKS.map((site) => (
                    <Link
                      key={site.name}
                      href={site.url}
                      target={site.local ? "_self" : "_blank"}
                      rel={site.local ? "" : "noopener noreferrer"}
                      className="text-sm font-medium text-gray-400 hover:text-cyan-400 transition-colors"
                    >
                      {site.name} {!site.local && "↗"}
                    </Link>
                  ))}
                </div>
              </div>

              <Button88x31 />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
