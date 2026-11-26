// src/app/hire/opengraph-image.jsx

import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { ImageResponse } from "next/og";

export const runtime = "nodejs";
export const dynamic = "force-static";

export const alt =
  "Full-stack developer for your React project, beside a screenshot of the Room TBA campus map";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

const OXBLOOD = "#6a0d16";
const ON_OXBLOOD = "#fbf6f5";
const YELLOW = "#ffc928";
const INK = "#1f1315";

const PANEL_WIDTH = 540;
// The screenshot is scaled past the frame and shifted so only the map shows.
const SHOT = { width: 1344, height: 675, left: -308, top: -10 };

async function loadAssets() {
  const pub = join(process.cwd(), "public");
  const [bold, medium, shot] = await Promise.all([
    readFile(join(pub, "fonts/og/schibsted-grotesk-latin-800-normal.woff")),
    readFile(join(pub, "fonts/og/schibsted-grotesk-latin-500-normal.woff")),
    readFile(join(pub, "projects/room-tba.png")),
  ]);
  return {
    bold,
    medium,
    shot: `data:image/png;base64,${shot.toString("base64")}`,
  };
}

export default async function Image() {
  const { bold, medium, shot } = await loadAssets();

  return new ImageResponse(
    <div
      style={{
        width: "100%",
        height: "100%",
        display: "flex",
        fontFamily: "Schibsted Grotesk",
        backgroundColor: OXBLOOD,
      }}
    >
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          width: PANEL_WIDTH,
          height: "100%",
          padding: "52px 48px 44px",
          color: ON_OXBLOOD,
        }}
      >
        <div
          style={{
            display: "flex",
            fontSize: 58,
            fontWeight: 800,
            lineHeight: 1.04,
            letterSpacing: "-0.03em",
          }}
        >
          Full-stack developer for your React project
        </div>
        <div
          style={{
            display: "flex",
            marginTop: 30,
            fontSize: 31,
            fontWeight: 800,
            color: YELLOW,
            letterSpacing: "-0.01em",
          }}
        >
          Start with one paid week.
        </div>
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            marginTop: "auto",
            gap: 4,
          }}
        >
          <div style={{ display: "flex", fontSize: 26, fontWeight: 800 }}>
            Simonee Mariquit
          </div>
          <div style={{ display: "flex", fontSize: 24, fontWeight: 500 }}>
            stimmie.dev/hire
          </div>
        </div>
      </div>

      <div
        style={{
          display: "flex",
          position: "relative",
          width: size.width - PANEL_WIDTH,
          height: "100%",
          overflow: "hidden",
        }}
      >
        {/* biome-ignore lint/performance/noImgElement: next/og renders plain img */}
        <img
          src={shot}
          alt=""
          width={SHOT.width}
          height={SHOT.height}
          style={{
            position: "absolute",
            left: SHOT.left,
            top: SHOT.top,
          }}
        />
        <div
          style={{
            display: "flex",
            position: "absolute",
            left: 28,
            bottom: 28,
            padding: "12px 18px",
            backgroundColor: ON_OXBLOOD,
            color: INK,
            fontSize: 23,
            fontWeight: 500,
            borderRadius: 8,
          }}
        >
          I maintain Room TBA with 20 contributors.
        </div>
      </div>
    </div>,
    {
      ...size,
      fonts: [
        { name: "Schibsted Grotesk", data: bold, weight: 800, style: "normal" },
        {
          name: "Schibsted Grotesk",
          data: medium,
          weight: 500,
          style: "normal",
        },
      ],
    },
  );
}
