"use client";

import { useEffect } from 'react';
import { useParams, notFound } from 'next/navigation';

// Define your redirects here - easy to manage!
// Format: { slug: destination_url }
const redirects = {
  // Workshops
  'workshops/dsg-codeit-day1': 'https://docs.google.com/presentation/d/...',
  'workshops/dsg-codeit-day2': 'https://docs.google.com/presentation/d/...',
  'workshops/nextstep-hacks': 'https://docs.google.com/presentation/d/...',
  
  // Talks
  'talks/qcu-ml-python': 'https://docs.google.com/presentation/d/...',
  'talks/dep-ai-study': 'https://docs.google.com/presentation/d/...',
  
  // Resources
  'freshie': 'https://docs.google.com/document/d/1cJya3Zb2ck9vkxIKc1LQjJomQS_LFtBOABlzHrb7Z5s/',
  'hackathons': 'https://docs.google.com/document/d/1nO2-vsOKjl4C_AngSSqc-knkUZvjJhe700unYZzcYsg/edit?usp=sharing',
  
  // Socials
  'linkedin': 'https://www.linkedin.com/in/stimmie',
  'github': 'https://www.github.com/smmariquit',
  'instagram': 'https://www.instagram.com/friedicecrm',
  'spotify': 'https://open.spotify.com/user/opzo90f4votlfqmg9rl94qrra',
  'osu': 'https://osu.ppy.sh/users/14900686',
  
  // Add more redirects as needed!
};

export default function RedirectPage() {
  const params = useParams();
  const slug = Array.isArray(params.slug) ? params.slug.join('/') : params.slug;
  
  useEffect(() => {
    const destination = redirects[slug];
    
    if (destination) {
      window.location.href = destination;
    }
  }, [slug]);

  const destination = redirects[slug];

  if (!destination) {
    notFound();
  }

  return (
    <div className="min-h-screen bg-black flex items-center justify-center">
      <div className="text-center text-white">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-white mx-auto mb-4"></div>
        <p className="text-lg">Redirecting...</p>
        <p className="text-sm text-gray-400 mt-2">
          If you&apos;re not redirected, <a href={destination} className="text-blue-400 hover:underline">click here</a>
        </p>
      </div>
    </div>
  );
}
