import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { ImageResponse } from "next/og";
import { SSH_KEY } from "@/data/keys";

export const runtime = "nodejs";

export const alt = "Stimmie SSH public key on stimmie.dev/keys";
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

function wrapLine(text, maxLen = 54) {
  const lines = [];
  let rest = text;

  while (rest.length > maxLen) {
    lines.push(rest.slice(0, maxLen));
    rest = rest.slice(maxLen);
  }

  if (rest) {
    lines.push(rest);
  }

  return lines;
}

export default async function Image() {
  const { fredoka, nunito } = await loadOgFonts();
  const ssh = (
    await readFile(join(process.cwd(), "public/keys/ssh.pub"), "utf8")
  ).trim();
  const keyLines = wrapLine(ssh);

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
          padding: "30px 34px",
        }}
      >
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "flex-start",
            borderBottom: "3px dashed #d6008f",
            paddingBottom: 16,
            marginBottom: 18,
          }}
        >
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            <div
              style={{
                display: "flex",
                fontFamily: "Fredoka",
                fontSize: 48,
                fontWeight: 700,
                color: "#c00000",
                letterSpacing: "0.04em",
                lineHeight: 1.1,
              }}
            >
              ~ keys ~
            </div>
            <div
              style={{
                display: "flex",
                fontFamily: "Nunito",
                fontSize: 24,
                fontWeight: 700,
                color: "#1a1a1a",
              }}
            >
              ssh public key · {SSH_KEY.algorithm}
            </div>
          </div>
          <div
            style={{
              display: "flex",
              fontFamily: "Nunito",
              fontSize: 22,
              fontWeight: 700,
              color: "#0a6699",
            }}
          >
            stimmie.dev/keys
          </div>
        </div>

        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 8,
            marginBottom: 16,
            fontFamily: "Nunito",
            fontSize: 20,
            fontWeight: 600,
            color: "#444",
          }}
        >
          <div style={{ display: "flex" }}>
            <span style={{ color: "#888", marginRight: 10 }}>key</span>
            <span style={{ color: "#1a1a1a" }}>{SSH_KEY.label}</span>
          </div>
          <div style={{ display: "flex" }}>
            <span style={{ color: "#888", marginRight: 10 }}>comment</span>
            <span style={{ color: "#1a1a1a" }}>{SSH_KEY.comment}</span>
          </div>
          <div style={{ display: "flex", flexWrap: "wrap" }}>
            <span style={{ color: "#888", marginRight: 10 }}>fingerprint</span>
            <span style={{ color: "#1a1a1a" }}>{SSH_KEY.fingerprint}</span>
          </div>
        </div>

        <div
          style={{
            display: "flex",
            flexDirection: "column",
            flex: 1,
            backgroundColor: "#111",
            border: "3px solid #000",
            boxShadow: "5px 5px 0 #d6008f",
            padding: "18px 20px",
            justifyContent: "center",
            gap: 4,
          }}
        >
          {keyLines.map((line) => (
            <div
              key={line}
              style={{
                display: "flex",
                fontFamily: "Nunito",
                fontSize: 22,
                fontWeight: 600,
                color: "#7CFC7C",
                lineHeight: 1.35,
                letterSpacing: "0.02em",
              }}
            >
              {line}
            </div>
          ))}
        </div>

        <div
          style={{
            display: "flex",
            marginTop: 16,
            fontFamily: "Fredoka",
            fontSize: 20,
            fontWeight: 700,
            color: "#7a00b8",
            justifyContent: "center",
          }}
        >
          verify before you trust · {SSH_KEY.download}
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
