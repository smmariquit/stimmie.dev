// scripts/verify-get14.mjs

// Runs every /get14 practice problem and checks its stored answer.
// Needs python3, node and gcc. Run: node scripts/verify-get14.mjs

import { execFileSync } from "node:child_process";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { compare, sheets } from "../src/data/get14-cheatsheet.js";
import { LEVELS, normalize, problems } from "../src/data/get14-problems.js";

const dir = mkdtempSync(join(tmpdir(), "get14-"));

// Turns a blank or write problem into a program, and says what it should print.
function program(problem, answer = problem.answer) {
  if (problem.kind === "blank") {
    return { code: problem.code.replace("____", answer), want: problem.expect };
  }
  const tests = problem.tests.map(([expr]) => expr);
  const want = problem.tests.map(([, out]) => out).join("\n");
  if (problem.lang === "python") {
    return {
      code: `${problem.solution}\n${tests.map((t) => `print(${t})`).join("\n")}`,
      want,
    };
  }
  if (problem.lang === "js") {
    return {
      code: `${problem.solution}\n${tests.map((t) => `console.log(${t});`).join("\n")}`,
      want,
    };
  }
  const calls = tests.map((t) => `printf("%d\\n", ${t});`).join("\n    ");
  return {
    code: `${problem.solution}\n\nint main(void) {\n    ${calls}\n    return 0;\n}`,
    want,
  };
}

function run(problem, code = problem.code) {
  const base = join(dir, problem.id);
  if (problem.lang === "python") {
    writeFileSync(`${base}.py`, code);
    return execFileSync("python3", [`${base}.py`], {
      encoding: "utf8",
      stdio: "pipe",
    });
  }
  if (problem.lang === "js") {
    writeFileSync(`${base}.cjs`, code);
    return execFileSync("node", [`${base}.cjs`], {
      encoding: "utf8",
      stdio: "pipe",
    });
  }
  // Snippets without a main get wrapped in one.
  const head = "#include <math.h>\n#include <stdio.h>\n#include <string.h>\n";
  const source = code.includes("#include")
    ? code
    : code.includes("int main")
      ? head + code
      : `${head}int main(void) {\n${code}\nreturn 0;\n}\n`;
  writeFileSync(`${base}.c`, source);
  execFileSync("gcc", ["-w", "-o", base, `${base}.c`, "-lm"]);
  return execFileSync(base, { encoding: "utf8", stdio: "pipe" });
}

let failed = 0;
const ids = new Set();
for (const problem of problems) {
  const issues = [];
  if (ids.has(problem.id)) issues.push("duplicate id");
  ids.add(problem.id);
  if (!LEVELS[problem.level - 1]) issues.push(`bad level ${problem.level}`);
  const answers =
    problem.kind === "blank"
      ? [problem.answer, ...(problem.accept ?? [])]
      : [null];
  if (problem.kind === "write" && problem.starter === undefined)
    issues.push("no starter");
  for (const answer of answers) {
    const { code, want } = problem.kind
      ? program(problem, answer)
      : { code: problem.code, want: problem.answer };
    const got = normalize(run(problem, code));
    if (got !== normalize(want)) {
      issues.push(
        `${answer ?? ""} expected ${JSON.stringify(want)}, got ${JSON.stringify(got)}`,
      );
    }
  }
  if (issues.length) {
    failed++;
    console.error(`FAIL ${problem.id}: ${issues.join("; ")}`);
  }
}

console.log(
  `${problems.length - failed} of ${problems.length} problems verified`,
);

// Cheat sheet lines with no print call are run as an expression.
function runnable(lang, code, format) {
  if (/print\(|console\.log\(|printf\(/.test(code)) return code;
  if (lang === "python") return `print(${code})`;
  if (lang === "js") return `console.log(${code})`;
  return format
    ? `printf("${format}", ${code});`
    : `printf("%d", (int)(${code}));`;
}

// Spaces, quotes and line breaks don't count on the cheat sheet.
const squash = (text) => text.replace(/[\s'"]/g, "");

const lines = [];
for (const row of compare) {
  for (const lang of ["python", "js", "c"]) {
    if (row[lang]) lines.push([lang, ...row[lang]]);
  }
}
for (const sheet of sheets) {
  for (const group of sheet.groups) {
    for (const row of group.rows) lines.push([sheet.lang, row.code, row.out]);
  }
}

let wrong = 0;
lines.forEach(([lang, code, out, format], index) => {
  let got;
  try {
    got = run({
      id: `sheet-${index}`,
      lang,
      code: runnable(lang, code, format),
    });
  } catch (error) {
    got = /TypeError/.test(String(error.stderr)) ? "TypeError" : "error";
  }
  if (squash(got) !== squash(out)) {
    wrong++;
    console.error(`FAIL ${lang} ${code}: expected ${out}, got ${got.trim()}`);
  }
});

console.log(
  `${lines.length - wrong} of ${lines.length} cheat sheet lines verified`,
);
process.exit(failed || wrong ? 1 : 0);
