import { blogPosts } from "@/data/blogs";

const SITE_URL = "https://www.stimmie.dev";

function escape(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

export const revalidate = 3600;

export function GET() {
  const items = blogPosts
    .map((p) => {
      const url = `${SITE_URL}/blog/${p.slug}`;
      const cover = p.coverImage ? `${SITE_URL}${p.coverImage}` : null;
      return `    <item>
      <title>${escape(p.title)}</title>
      <link>${url}</link>
      <guid isPermaLink="true">${url}</guid>
      <pubDate>${new Date(p.date).toUTCString()}</pubDate>
      <description>${escape(p.excerpt || "")}</description>${cover ? `\n      <enclosure url="${cover}" type="image/${cover.endsWith(".png") ? "png" : "jpeg"}" length="0" />` : ""}
    </item>`;
    })
    .join("\n");

  const xml = `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">
  <channel>
    <title>stimmie.dev</title>
    <link>${SITE_URL}/blog</link>
    <atom:link href="${SITE_URL}/rss.xml" rel="self" type="application/rss+xml" />
    <description>Blog posts by Simonee Ezekiel Mariquit.</description>
    <language>en</language>
    <lastBuildDate>${new Date().toUTCString()}</lastBuildDate>
${items}
  </channel>
</rss>
`;
  return new Response(xml, { headers: { "Content-Type": "application/rss+xml; charset=utf-8" } });
}
