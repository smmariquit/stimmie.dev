// src/app/page.js

import { getSiteVersion } from "@/lib/changelog";
import { getAllMediaData } from "@/lib/media";
import HomeClient from "./HomeClient";

// Revalidate every 1 hour (3600 seconds)
export const revalidate = 3600;

export default async function Home() {
  const mediaData = await getAllMediaData();
  const version = getSiteVersion();

  return <HomeClient mediaData={mediaData} version={version} />;
}
