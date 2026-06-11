// src/app/robots.js

const SITE_URL = "https://stimmie.dev";

export default function robots() {
  return {
    rules: [
      {
        userAgent: "*",
        allow: "/",
        // /r/ is a client-side redirect handler — not useful in search results.
        disallow: ["/r/"],
      },
    ],
    sitemap: `${SITE_URL}/sitemap.xml`,
  };
}
