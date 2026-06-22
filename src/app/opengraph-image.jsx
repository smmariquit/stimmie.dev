// src/app/opengraph-image.jsx

import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { ImageResponse } from "next/og";

export const runtime = "nodejs";

export const alt =
  "stimmie's homepage — personal site with projects, talks, writing, and communities";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

const TAGS = ["projects", "talks", "blog", "career", "archive"];

async function loadOgFonts() {
  const fontDir = join(process.cwd(), "public/fonts/og");
  const [fredoka, nunito] = await Promise.all([
    readFile(join(fontDir, "fredoka-latin-700-normal.woff")),
    readFile(join(fontDir, "nunito-latin-600-normal.woff")),
  ]);
  return { fredoka, nunito };
}

export default async function Image() {
  const { fredoka, nunito } = await loadOgFonts();

  return new ImageResponse(
    <div
      style={{
        width: "100%",
        height: "100%",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        backgroundColor: "#0a0a1a",
        backgroundImage:
          "radial-gradient(1px 1px at 20px 30px, #fff, transparent), radial-gradient(1px 1px at 80px 120px, rgba(255,255,255,0.5), transparent), radial-gradient(1px 1px at 160px 60px, rgba(255,255,255,0.4), transparent), radial-gradient(1.5px 1.5px at 240px 180px, #fff, transparent), radial-gradient(1px 1px at 320px 40px, rgba(255,255,255,0.5), transparent), radial-gradient(1px 1px at 400px 200px, rgba(255,255,255,0.4), transparent)",
        padding: 40,
      }}
    >
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          width: "100%",
          height: "100%",
          backgroundColor: "#fffef5",
          border: "4px solid #ff00aa",
          boxShadow: "10px 10px 0 #000",
          padding: "44px 52px",
        }}
      >
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            borderBottom: "3px dashed #d6008f",
            paddingBottom: 28,
            marginBottom: 24,
          }}
        >
          <div
            style={{
              fontFamily: "Fredoka",
              fontSize: 58,
              fontWeight: 700,
              color: "#c00000",
              letterSpacing: "0.04em",
              lineHeight: 1.1,
            }}
          >
            ~* stimmie&apos;s homepage *~
          </div>
          <div
            style={{
              fontFamily: "Nunito",
              fontSize: 24,
              fontWeight: 600,
              color: "#555",
              marginTop: 10,
            }}
          >
            stimmie.dev
          </div>
        </div>

        <div
          style={{
            display: "flex",
            backgroundColor: "#000",
            color: "#0f0",
            fontFamily: "Fredoka",
            fontSize: 17,
            fontWeight: 700,
            padding: "10px 20px",
            marginBottom: 28,
            justifyContent: "center",
            letterSpacing: "0.06em",
          }}
        >
          * welcome to my corner of the web * software engineer * tinkerer *
        </div>

        <div
          style={{
            display: "flex",
            flexDirection: "column",
            flex: 1,
            gap: 22,
          }}
        >
          <div
            style={{
              display: "flex",
              flexWrap: "wrap",
              fontFamily: "Nunito",
              fontSize: 30,
              fontWeight: 600,
              color: "#1a1a1a",
              lineHeight: 1.35,
            }}
          >
            <span>Hi! You found my website on the </span>
            <span style={{ color: "#cc0000", fontWeight: 700 }}>internet</span>
            <span>.</span>
          </div>

          <div
            style={{
              fontFamily: "Nunito",
              fontSize: 24,
              fontWeight: 600,
              color: "#333",
              lineHeight: 1.5,
              maxWidth: 980,
            }}
          >
            I&apos;m Stimmie — software engineer, creator, and tinkerer. My
            personal corner of the retro web: projects, talks, writing,
            communities, and whatever I&apos;m building next.
          </div>

          <div
            style={{
              display: "flex",
              flexWrap: "wrap",
              gap: 12,
              marginTop: "auto",
            }}
          >
            {TAGS.map((tag) => (
              <div
                key={tag}
                style={{
                  display: "flex",
                  fontFamily: "Nunito",
                  fontSize: 18,
                  fontWeight: 700,
                  color: "#1a1a1a",
                  backgroundColor: "#e8f0ff",
                  border: "2px solid #000",
                  borderRadius: 999,
                  padding: "8px 18px",
                }}
              >
                {tag}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>,
    {
      ...size,
      fonts: [
        { name: "Fredoka", data: fredoka, weight: 700, style: "normal" },
        { name: "Nunito", data: nunito, weight: 600, style: "normal" },
      ],
    },
  );
}
