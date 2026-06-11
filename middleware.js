// middleware.js

import { NextResponse } from "next/server";
import {
  APEX_HOST,
  TODO_TARGET,
  lookupSubdomainRedirect,
} from "./src/data/redirects";

// Edge middleware. Handles only the <category>.stimmie.dev/<slug> shape;
// apex /r/<slug> routing is owned by src/app/r/[...slug]/page.js so the App
// Router can render a friendly fallback when a slug isn't configured.
//
// Once the Cloudflare Bulk Redirect migration runs, both surfaces become
// fallbacks: CF will redirect at the edge before the request reaches us.

export function middleware(request) {
  const url = request.nextUrl;
  const hostname = (request.headers.get("host") || "").toLowerCase();

  // Only the apex zone gets subdomain routing. Preview deployments on
  // *.vercel.app don't have category subdomains, so they fall through.
  if (!hostname.endsWith(`.${APEX_HOST}`)) {
    return NextResponse.next();
  }

  const subdomain = hostname.slice(0, -1 * (APEX_HOST.length + 1));
  if (!subdomain || subdomain === "www") {
    return NextResponse.next();
  }

  const slug = url.pathname.replace(/^\/+/, "").replace(/\/+$/, "");
  if (!slug) return NextResponse.next();

  const destination = lookupSubdomainRedirect(subdomain, slug);
  if (!destination || destination === TODO_TARGET) {
    return NextResponse.next();
  }

  return NextResponse.redirect(destination);
}

export const config = {
  matcher: [
    // Skip Next.js internals and any file-served directory. Anything else
    // gets the subdomain-redirect check above.
    "/((?!_next|api|favicon.ico|images|logos|fonts|projects|talks|satoshi).*)",
  ],
};
