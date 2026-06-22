// src/app/opengraph-image.jsx

import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { ImageResponse } from "next/og";
import { NAV_LINKS } from "@/components/home/constants";

export const runtime = "nodejs";

export const alt =
  "stimmie's homepage: personal site with projects, talks, writing, and communities";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

const TAGS = ["projects", "talks", "blog", "career", "archive"];
const OG_NAV = NAV_LINKS.filter((item) => item.href !== "/v2");

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
          "radial-gradient(1px 1px at 20px 30px, #fff, transparent), radial-gradient(1px 1px at 80px 120px, rgba(255,255,255,0.5), transparent), radial-gradient(1px 1px at 160px 60px, rgba(255,255,255,0.4), transparent), radial-gradient(1.5px 1.5px at 240px 180px, #fff, transparent)",
        padding: 28,
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
          padding: "32px 36px",
        }}
      >
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            borderBottom: "3px dashed #d6008f",
            paddingBottom: 18,
            marginBottom: 16,
          }}
        >
          <div
            style={{
              display: "flex",
              fontFamily: "Fredoka",
              fontSize: 52,
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
              display: "flex",
              fontFamily: "Nunito",
              fontSize: 28,
              fontWeight: 700,
              color: "#1a1a1a",
              marginTop: 8,
            }}
          >
            stimmie.dev
          </div>
        </div>

        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            gap: 10,
            justifyContent: "center",
            padding: "12px 14px",
            marginBottom: 16,
            backgroundColor: "#1a1a1a",
            border: "3px solid #000",
            boxShadow: "4px 4px 0 #d6008f",
          }}
        >
          {OG_NAV.map((item) => (
            <div
              key={item.href}
              style={{
                display: "flex",
                fontFamily: "Fredoka",
                fontSize: 22,
                fontWeight: 700,
                color: "#1a1a1a",
                backgroundColor: "#fffef5",
                border: "2px solid #ff00aa",
                borderRadius: 999,
                padding: "6px 14px",
              }}
            >
              {item.label}
            </div>
          ))}
        </div>

        <div
          style={{
            display: "flex",
            backgroundColor: "#000",
            color: "#0f0",
            fontFamily: "Fredoka",
            fontSize: 22,
            fontWeight: 700,
            padding: "10px 16px",
            marginBottom: 18,
            justifyContent: "center",
            letterSpacing: "0.05em",
          }}
        >
          * welcome to my corner of the web * software engineer * tinkerer *
        </div>

        <div
          style={{
            display: "flex",
            flexDirection: "column",
            flex: 1,
            gap: 14,
          }}
        >
          <div
            style={{
              display: "flex",
              flexWrap: "wrap",
              alignItems: "baseline",
              fontFamily: "Nunito",
              fontSize: 36,
              fontWeight: 700,
              color: "#1a1a1a",
              lineHeight: 1.25,
            }}
          >
            <span>Hi! You found my website on </span>
            <span style={{ color: "#cc0000" }}>THE INTERNET!!</span>
          </div>

          <div
            style={{
              display: "flex",
              fontFamily: "Nunito",
              fontSize: 28,
              fontWeight: 600,
              color: "#1a1a1a",
              lineHeight: 1.35,
            }}
          >
            I&apos;m Stimmie, a software engineer, creator, and tinkerer.
            Projects, talks, writing, communities, and whatever I&apos;m
            building next.
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
                  fontSize: 24,
                  fontWeight: 700,
                  color: "#1a1a1a",
                  backgroundColor: "#e8f0ff",
                  border: "3px solid #000",
                  borderRadius: 999,
                  padding: "8px 20px",
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
