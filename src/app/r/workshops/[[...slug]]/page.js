// src/app/r/workshops/[[...slug]]/page.js

import WorkshopRedirect from "./WorkshopRedirect";
import { workshops } from "./workshops";

// Static export needs every slug up front; [] is the /r/workshops index.
export function generateStaticParams() {
  return [{ slug: [] }, ...Object.keys(workshops).map((k) => ({ slug: [k] }))];
}

export default function Page() {
  return <WorkshopRedirect />;
}
