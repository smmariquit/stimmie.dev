import fs from "node:fs";
import path from "node:path";

export function getSiteVersion() {
  const pkg = JSON.parse(
    fs.readFileSync(path.join(process.cwd(), "package.json"), "utf8"),
  );
  return pkg.version;
}

/** Parse CHANGELOG.md into version sections (newest first). */
export function getChangelogEntries() {
  const raw = fs.readFileSync(
    path.join(process.cwd(), "CHANGELOG.md"),
    "utf8",
  );

  return raw
    .split(/\n(?=## )/)
    .slice(1)
    .map((section) => {
      const lines = section.trim().split("\n");
      const heading = lines[0].replace(/^## /, "");
      const body = lines.slice(1).join("\n").trim();
      return { heading, body };
    });
}
