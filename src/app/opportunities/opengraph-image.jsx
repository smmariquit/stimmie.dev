import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { ImageResponse } from "next/og";
import {
  getOpportunities,
  getOpportunitiesBoard,
  getOpportunityImagePresentation,
  resolveOpportunityImage,
} from "@/data/opportunities";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export const alt = "Mosaic of opportunity board cover images";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

const GAP = 3;
const ROWS = 3;
const EVEN_COLS = 4;
const ODD_COLS = 5;
const ROW_HEIGHT = Math.floor((size.height - (ROWS - 1) * GAP) / ROWS);

function hashString(value) {
  let hash = 0;
  for (let i = 0; i < value.length; i += 1) {
    hash = (hash * 31 + value.charCodeAt(i)) | 0;
  }
  return hash;
}

function getEvenTileWidth() {
  return Math.floor((size.width - (EVEN_COLS - 1) * GAP) / EVEN_COLS);
}

function buildBrickTiles() {
  const tileWidth = getEvenTileWidth();
  const xOffset = Math.floor((tileWidth + GAP) / 2);
  const tiles = [];

  for (let row = 0; row < ROWS; row += 1) {
    const isOddRow = row % 2 === 1;
    const cols = isOddRow ? ODD_COLS : EVEN_COLS;
    const y = row * (ROW_HEIGHT + GAP);
    const startX = isOddRow ? -xOffset : 0;

    for (let col = 0; col < cols; col += 1) {
      tiles.push({
        x: startX + col * (tileWidth + GAP),
        y,
        width: tileWidth,
        height: ROW_HEIGHT,
      });
    }
  }

  return tiles;
}

function pickMosaicSources(items, count, seed) {
  const seen = new Set();
  const sources = [];

  for (const item of items) {
    const src = resolveOpportunityImage(item);
    const presentation = getOpportunityImagePresentation(item);

    if (seen.has(src)) {
      continue;
    }
    if (presentation.className.includes("favicon")) {
      continue;
    }
    if (src.includes("/defaults/")) {
      continue;
    }
    if (src.endsWith(".webp")) {
      continue;
    }

    seen.add(src);
    sources.push(src);
  }

  const ordered = [...sources].sort(
    (a, b) => hashString(`${seed}:${a}`) - hashString(`${seed}:${b}`),
  );

  return Array.from({ length: count }, (_, index) => {
    return ordered[index % ordered.length];
  });
}

async function loadImageDataUrl(publicPath) {
  const filePath = join(process.cwd(), "public", publicPath.replace(/^\//, ""));
  const extension = filePath.split(".").pop()?.toLowerCase() ?? "png";
  const buffer = await readFile(filePath);

  if (extension === "svg") {
    return `data:image/svg+xml;base64,${buffer.toString("base64")}`;
  }

  const mime =
    extension === "png"
      ? "image/png"
      : extension === "webp"
        ? "image/webp"
        : "image/jpeg";

  return `data:${mime};base64,${buffer.toString("base64")}`;
}

export default async function Image() {
  const board = getOpportunitiesBoard();
  const items = getOpportunities();
  const tiles = buildBrickTiles();
  const sources = pickMosaicSources(items, tiles.length, board.lastUpdated);

  const imageData = await Promise.all(
    sources.map(async (src) => {
      try {
        return await loadImageDataUrl(src);
      } catch {
        return null;
      }
    }),
  );

  return new ImageResponse(
    <div
      style={{
        width: "100%",
        height: "100%",
        display: "flex",
        position: "relative",
        overflow: "hidden",
        backgroundColor: "#0a0a1a",
      }}
    >
      {tiles.map((tile, index) => (
        <div
          key={`${tile.x}-${tile.y}`}
          style={{
            display: "flex",
            position: "absolute",
            left: tile.x,
            top: tile.y,
            width: tile.width,
            height: tile.height,
            overflow: "hidden",
            backgroundColor: "#1a1a2e",
          }}
        >
          {imageData[index] ? (
            <img
              alt=""
              src={imageData[index]}
              style={{
                display: "flex",
                width: "100%",
                height: "100%",
                objectFit: "cover",
              }}
            />
          ) : (
            <div
              style={{
                display: "flex",
                width: "100%",
                height: "100%",
                backgroundColor: "#2a2a4a",
              }}
            />
          )}
        </div>
      ))}
    </div>,
    {
      ...size,
    },
  );
}
