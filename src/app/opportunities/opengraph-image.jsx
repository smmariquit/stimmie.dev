import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { ImageResponse } from "next/og";
import {
  formatBoardUpdated,
  getOpportunities,
  getOpportunitiesBoard,
  OPPORTUNITY_TYPE_ORDER,
  OPPORTUNITY_TYPES,
} from "@/data/opportunities";

export const runtime = "nodejs";

export const alt =
  "Stimmie opportunities board with live counts by category";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

const TYPE_STYLES = {
  hackathon: { bg: "#d8ffe0", color: "#075c1f" },
  "game-jam": { bg: "#fff0e8", color: "#9a3d07" },
  internship: { bg: "#e8f0ff", color: "#0a3d7a" },
  program: { bg: "#ffe8f5", color: "#8a0058" },
  event: { bg: "#fffacd", color: "#5a4a00" },
  certificate: { bg: "#efe0ff", color: "#5a0a8a" },
};

async function loadOgFonts() {
  const fontDir = join(process.cwd(), "public/fonts/og");
  const [fredoka, nunito] = await Promise.all([
    readFile(join(fontDir, "fredoka-latin-700-normal.woff")),
    readFile(join(fontDir, "nunito-latin-600-normal.woff")),
  ]);
  return { fredoka, nunito };
}

function getTypeCounts(items) {
  const counts = new Map();
  for (const type of OPPORTUNITY_TYPE_ORDER) {
    counts.set(type, 0);
  }
  for (const item of items) {
    counts.set(item.type, (counts.get(item.type) ?? 0) + 1);
  }
  return OPPORTUNITY_TYPE_ORDER.map((type) => ({
    type,
    label: OPPORTUNITY_TYPES[type].label,
    count: counts.get(type) ?? 0,
    style: TYPE_STYLES[type],
  })).filter((row) => row.count > 0);
}

export default async function Image() {
  const { fredoka, nunito } = await loadOgFonts();
  const board = getOpportunitiesBoard();
  const items = getOpportunities();
  const rows = getTypeCounts(items);
  const updated = formatBoardUpdated(board.lastUpdated);

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
          padding: "28px 32px",
        }}
      >
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "flex-start",
            borderBottom: "3px dashed #d6008f",
            paddingBottom: 16,
            marginBottom: 16,
          }}
        >
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            <div
              style={{
                display: "flex",
                fontFamily: "Fredoka",
                fontSize: 46,
                fontWeight: 700,
                color: "#c00000",
                letterSpacing: "0.04em",
                lineHeight: 1.1,
              }}
            >
              ~ opportunities ~
            </div>
            <div
              style={{
                display: "flex",
                fontFamily: "Nunito",
                fontSize: 22,
                fontWeight: 700,
                color: "#1a1a1a",
              }}
            >
              stimmie.dev/opportunities
            </div>
          </div>
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              backgroundColor: "#000",
              border: "3px solid #000",
              boxShadow: "4px 4px 0 #d6008f",
              padding: "10px 18px",
              minWidth: 150,
            }}
          >
            <div
              style={{
                display: "flex",
                fontFamily: "Fredoka",
                fontSize: 44,
                fontWeight: 700,
                color: "#cc0066",
                lineHeight: 1,
              }}
            >
              {items.length}
            </div>
            <div
              style={{
                display: "flex",
                fontFamily: "Nunito",
                fontSize: 16,
                fontWeight: 700,
                color: "#fff",
                marginTop: 4,
                letterSpacing: "0.06em",
              }}
            >
              LISTINGS
            </div>
          </div>
        </div>

        <div
          style={{
            display: "flex",
            flexDirection: "row",
            flexWrap: "wrap",
            gap: 12,
            flex: 1,
            alignContent: "flex-start",
          }}
        >
          {rows.map((row) => (
            <div
              key={row.type}
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                width: 548,
                backgroundColor: row.style.bg,
                border: "3px solid #000",
                boxShadow: "4px 4px 0 #000",
                padding: "14px 18px",
              }}
            >
              <div
                style={{
                  display: "flex",
                  fontFamily: "Fredoka",
                  fontSize: 24,
                  fontWeight: 700,
                  color: row.style.color,
                  letterSpacing: "0.02em",
                }}
              >
                {row.label}
              </div>
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontFamily: "Fredoka",
                  fontSize: 32,
                  fontWeight: 700,
                  color: row.style.color,
                  backgroundColor: "#fffef5",
                  border: "2px solid #000",
                  minWidth: 56,
                  padding: "4px 12px",
                  boxShadow: "2px 2px 0 #000",
                }}
              >
                {row.count}
              </div>
            </div>
          ))}
        </div>

        <div
          style={{
            display: "flex",
            justifyContent: "center",
            marginTop: 14,
            backgroundColor: "#000",
            color: "#0f0",
            fontFamily: "Fredoka",
            fontSize: 20,
            fontWeight: 700,
            padding: "8px 16px",
            letterSpacing: "0.04em",
          }}
        >
          * PH and online * updated {updated} * verify on official sites *
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
