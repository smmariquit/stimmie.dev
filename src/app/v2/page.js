// src/app/v2/page.js

import { frozenMediaData } from "@/archive/v2/media";
import HomeClient from "./HomeClient";

/** Archived v2 — static snapshot; does not fetch live RSS. */
export const dynamic = "force-static";

export default function ArchivedV2Page() {
  return <HomeClient mediaData={frozenMediaData} />;
}
