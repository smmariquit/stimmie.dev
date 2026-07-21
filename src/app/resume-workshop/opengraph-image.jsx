// src/app/resume-workshop/opengraph-image.jsx

import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { ImageResponse } from "next/og";

export const runtime = "nodejs";

export const alt =
  "Stimmie's resume and portfolio workshop: send your resume, get a review, pay as you can";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

const STEPS = ["1. draft your resume", "2. send it over", "3. get a review"];
const TAGS = [
  "resume review",
  "portfolio review",
  "interview help",
  "pay-as-you-can",
];

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
            ~* resume workshop *~
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
            stimmie.dev/resume-workshop
          </div>
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
          * 20+ people helped and counting * pay-as-you-can *
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
              fontFamily: "Nunito",
              fontSize: 34,
              fontWeight: 700,
              color: "#1a1a1a",
              lineHeight: 1.3,
            }}
          >
            Get extra eyes on your resume, portfolio, or interview prep. Send it
            over and I&apos;ll review it!
          </div>

          <div
            style={{
              display: "flex",
              flexWrap: "wrap",
              gap: 12,
            }}
          >
            {STEPS.map((step) => (
              <div
                key={step}
                style={{
                  display: "flex",
                  fontFamily: "Fredoka",
                  fontSize: 24,
                  fontWeight: 700,
                  color: "#1a1a1a",
                  backgroundColor: "#fffef5",
                  border: "3px solid #ff00aa",
                  borderRadius: 999,
                  padding: "8px 18px",
                }}
              >
                {step}
              </div>
            ))}
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
