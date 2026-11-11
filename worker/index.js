// worker/index.js
//
// Pages are a static export served straight from Workers static assets; this
// script only runs when no asset matches (assets are checked first), so normal
// page views execute no code. It replaces what used to need a server on Vercel:
//   * /api/dt          Daily Ten backend (was src/app/api/dt/route.js)
//   * /api/workshop    resume-workshop form (was a server action)
//   * <category>.stimmie.dev/<slug> and /r/<slug> short links (was middleware.js
//     + src/app/r/[...slug]); Cloudflare Bulk Redirects catch most of these
//     before they get here once scripts/cf-migrate-redirects.mjs has run.
//   * /opportunities/<slug> legacy issue URLs -> /opportunities

import {
  APEX_HOST,
  lookupApexRedirect,
  lookupSubdomainRedirect,
  TODO_TARGET,
} from "../src/data/redirects.js";
import { connect, GET, POST, PUT } from "./dt.js";
import { submitWorkshopRequest } from "./workshop.js";

const dtHandlers = { GET, PUT, POST };
const live = (dest) => (dest && dest !== TODO_TARGET ? dest : null);

export default {
  async fetch(req, env, ctx) {
    const url = new URL(req.url);
    const path = url.pathname.replace(/\/+$/, "");

    if (path === "/api/dt") {
      const handler = dtHandlers[req.method];
      if (!handler) return new Response(null, { status: 405 });
      const db = connect();
      await db.connect();
      try {
        return await handler(req, db);
      } finally {
        ctx.waitUntil(db.end());
      }
    }

    if (path === "/api/workshop" && req.method === "POST") {
      return Response.json(await submitWorkshopRequest(await req.formData()));
    }

    const host = url.hostname.toLowerCase();
    const sub = host.endsWith(`.${APEX_HOST}`)
      ? host.slice(0, -(APEX_HOST.length + 1))
      : "";
    const slug = path.replace(/^\/+/, "");
    const dest =
      sub && sub !== "www"
        ? live(lookupSubdomainRedirect(sub, slug))
        : path.startsWith("/r/")
          ? live(lookupApexRedirect(slug.slice(2)))
          : null;
    if (dest) return Response.redirect(dest, 307);

    if (path.startsWith("/opportunities/")) {
      return Response.redirect(new URL("/opportunities", url), 308);
    }

    const notFound = await env.ASSETS.fetch(new URL("/404.html", url));
    return new Response(notFound.body, {
      status: 404,
      headers: notFound.headers,
    });
  },
};
