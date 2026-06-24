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
    label: OPPORTUNITY_TYPES[type].label,
    count: counts.get(type) ?? 0,
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
        backgroundColor: "#111827",
        padding: 48,
      }}
    >
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          width: "100%",
          height: "100%",
          backgroundColor: "#ffffff",
          borderRadius: 20,
          padding: "40px 48px",
        }}
      >
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "flex-start",
            marginBottom: 28,
          }}
        >
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <div
              style={{
                display: "flex",
                fontFamily: "Fredoka",
                fontSize: 44,
                fontWeight: 700,
                color: "#111827",
                lineHeight: 1.1,
              }}
            >
              Opportunities
            </div>
            <div
              style={{
                display: "flex",
                fontFamily: "Nunito",
                fontSize: 24,
                fontWeight: 600,
                color: "#4b5563",
              }}
            >
              Philippines and online listings
            </div>
          </div>
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "flex-end",
              gap: 4,
            }}
          >
            <div
              style={{
                display: "flex",
                fontFamily: "Fredoka",
                fontSize: 56,
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
                fontSize: 22,
                fontWeight: 700,
                color: "#6b7280",
              }}
            >
              listings live
            </div>
          </div>
        </div>

        <div
          style={{
            display: "flex",
            flexDirection: "row",
            flexWrap: "wrap",
            gap: 16,
            flex: 1,
            alignContent: "flex-start",
          }}
        >
          {rows.map((row) => (
            <div
              key={row.label}
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                width: 496,
                backgroundColor: "#f9fafb",
                border: "2px solid #e5e7eb",
                borderRadius: 14,
                padding: "18px 22px",
              }}
            >
              <div
                style={{
                  display: "flex",
                  fontFamily: "Nunito",
                  fontSize: 26,
                  fontWeight: 700,
                  color: "#111827",
                }}
              >
                {row.label}
              </div>
              <div
                style={{
                  display: "flex",
                  fontFamily: "Fredoka",
                  fontSize: 36,
                  fontWeight: 700,
                  color: "#111827",
                  minWidth: 48,
                  justifyContent: "flex-end",
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
            justifyContent: "space-between",
            alignItems: "center",
            marginTop: 24,
            paddingTop: 20,
            borderTop: "2px solid #e5e7eb",
          }}
        >
          <div
            style={{
              display: "flex",
              fontFamily: "Nunito",
              fontSize: 24,
              fontWeight: 700,
              color: "#111827",
            }}
          >
            stimmie.dev/opportunities
          </div>
          <div
            style={{
              display: "flex",
              fontFamily: "Nunito",
              fontSize: 22,
              fontWeight: 600,
              color: "#6b7280",
            }}
          >
            Updated {updated}
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
