# AGENTS.md

## Cursor Cloud specific instructions

This repo is the `stimmie.dev` personal portfolio site — a single Next.js 16 app
(App Router, React 19, Tailwind v4, Turbopack). There is one service. Standard
commands live in `package.json` `scripts`; the notes below are only the
non-obvious bits.

### Running / building / linting

- Dev server: `npm run dev` (Next.js + Turbopack on `http://localhost:3000`).
- Build: `npm run build` — this is the only check CI runs (`.github/workflows/ci.yml`).
- Lint: `npm run lint` runs `biome check`. It currently reports many pre-existing
  formatting/lint findings and exits non-zero on a clean checkout. CI does NOT
  run lint, so a non-zero exit here is expected and not caused by your changes.
  Use `npm run format` (`biome format --write`) only on files you intentionally
  touch — do not mass-reformat the repo.

### Environment variables

- All env vars (see `.env.example`) are OPTIONAL. They only affect helper
  `scripts/*.mjs` (Cloudflare bulk-redirect sync, opportunity screenshots,
  Resend stale-nag emails) — not the app itself. The dev server, build, and all
  pages run fine with no `.env` file.

### No automated tests

- There is no test framework or test script in this repo. Verify changes via
  `npm run build` and by exercising pages in the browser against `npm run dev`.

### Research markdown

- Per `.cursor/rules/research-pipeline.mdc`, all research `.md` files belong
  under `research/` (not the repo root or `src/`).
