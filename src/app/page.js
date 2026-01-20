import { getAllMediaData } from '@/lib/media';
import HomeClient from './HomeClient';

// Revalidate every 1 hour (3600 seconds)
export const revalidate = 3600;

export default async function Home() {
  // Fetch media data at build time (and revalidate hourly)
  const mediaData = await getAllMediaData();

  return <HomeClient mediaData={mediaData} />;
}
