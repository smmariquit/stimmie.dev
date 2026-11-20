// scripts/verify-get14.mjs

// Runs every /get14 practice problem and checks its stored answer.
// Needs python3, node and gcc. Run: node scripts/verify-get14.mjs

import { execFileSync } from "node:child_process";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { LEVELS, normalize, problems } from "../src/data/get14-problems.js";

const dir = mkdtempSync(join(tmpdir(), "get14-"));

function run(problem) {
  const base = join(dir, problem.id);
  if (problem.lang === "python") {
    writeFileSync(`${base}.py`, problem.code);
    return execFileSync("python3", [`${base}.py`], { encoding: "utf8" });
  }
  if (problem.lang === "js") {
    writeFileSync(`${base}.cjs`, problem.code);
    return execFileSync("node", [`${base}.cjs`], { encoding: "utf8" });
  }
  // Snippets without a main get wrapped in one.
  const source = problem.code.includes("int main")
    ? problem.code
    : `#include <stdio.h>\nint main(void) {\n${problem.code}\nreturn 0;\n}\n`;
  writeFileSync(`${base}.c`, source);
  execFileSync("gcc", ["-w", "-o", base, `${base}.c`]);
  return execFileSync(base, { encoding: "utf8" });
}

let failed = 0;
const ids = new Set();
for (const problem of problems) {
  const issues = [];
  if (ids.has(problem.id)) issues.push("duplicate id");
  ids.add(problem.id);
  if (!LEVELS[problem.level - 1]) issues.push(`bad level ${problem.level}`);
  const got = normalize(run(problem));
  if (got !== normalize(problem.answer)) {
    issues.push(
      `expected ${JSON.stringify(problem.answer)}, got ${JSON.stringify(got)}`,
    );
  }
  if (issues.length) {
    failed++;
    console.error(`FAIL ${problem.id}: ${issues.join("; ")}`);
  }
}

console.log(
  `${problems.length - failed} of ${problems.length} problems verified`,
);
process.exit(failed ? 1 : 0);
