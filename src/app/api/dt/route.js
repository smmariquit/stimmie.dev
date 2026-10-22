// Daily Ten backend: a tiny doc store on Postgres (players/*, days/*, meta/*)
// plus Telegram link codes for the nagger. Env: DATABASE_URL, DT_ADMIN_IDS
// (comma-separated player ids), DT_BOT_USERNAME.
import { createHash, randomBytes } from "node:crypto";
import pg from "pg";

export const dynamic = "force-dynamic";

// ponytail: one connection per serverless instance, Heroku essential-0 caps at 20
const pool =
  globalThis.__dtPool ??
  new pg.Pool({
    connectionString: process.env.DATABASE_URL,
    ssl: /@(localhost|127\.0\.0\.1)[:/]/.test(process.env.DATABASE_URL ?? "")
      ? false
      : { rejectUnauthorized: false },
    max: 1,
  });
globalThis.__dtPool = pool;

let ready;
function init() {
  ready ??= pool.query(`
    create table if not exists dt_docs (path text primary key, data jsonb not null, updated_at timestamptz not null default now());
    create table if not exists dt_codes (code text primary key, player_id text not null, created_at timestamptz not null default now());
    create table if not exists dt_tg (player_id text primary key, chat_id text not null);
  `);
  return ready;
}

const PATH = /^(players|days|meta)\/[A-Za-z0-9_-]{1,64}$/;
const COL = /^(players|days|meta)$/;
const MAX_BYTES = 64 * 1024;
const manilaDay = (t = Date.now()) =>
  new Date(t + 8 * 3600e3).toISOString().slice(0, 10);

const json = (body, status = 200) =>
  Response.json(body, { status, headers: { "cache-control": "no-store" } });
const who = (req) => {
  const s = req.headers.get("x-dt-secret");
  return s && s.length >= 32
    ? createHash("sha256").update(s).digest("hex").slice(0, 24)
    : null;
};
const admins = () =>
  (process.env.DT_ADMIN_IDS ?? "").split(",").filter(Boolean);

export async function GET(req) {
  await init();
  const q = new URL(req.url).searchParams;
  if (q.has("act")) {
    const me = who(req);
    if (!me) return json({ linked: false, admin: false });
    const { rowCount } = await pool.query(
      "select 1 from dt_tg where player_id = $1",
      [me],
    );
    return json({ linked: rowCount > 0, admin: admins().includes(me) });
  }
  if (q.has("d")) {
    const p = q.get("d");
    if (!PATH.test(p)) return json({ error: "bad path" }, 400);
    const { rows } = await pool.query(
      "select data from dt_docs where path = $1",
      [p],
    );
    return json({ data: rows[0]?.data ?? null });
  }
  const c = q.get("c");
  if (!COL.test(c ?? "")) return json({ error: "bad collection" }, 400);
  const n = Math.min(Number(q.get("n")) || 1000, 1000);
  const { rows } = await pool.query(
    "select substr(path, length($1) + 2) as id, data from dt_docs where path like $1 || '/%' order by path limit $2",
    [c, n],
  );
  return json(rows);
}

export async function PUT(req) {
  await init();
  const p = new URL(req.url).searchParams.get("d");
  if (!PATH.test(p ?? "")) return json({ error: "bad path" }, 400);
  const raw = await req.text();
  if (raw.length > MAX_BYTES) return json({ error: "too big" }, 413);
  let data;
  try {
    data = JSON.parse(raw);
  } catch {
    return json({ error: "bad json" }, 400);
  }
  if (!data || typeof data !== "object" || Array.isArray(data))
    return json({ error: "bad doc" }, 400);
  const [col, id] = p.split("/");
  const me = who(req);

  if (col === "players") {
    if (me !== id) return json({ error: "not yours" }, 403);
    if (typeof data.nick !== "string" || data.nick.length > 20)
      return json({ error: "bad nick" }, 400);
  } else if (col === "days") {
    // first writer wins so everyone plays the same set
    const near = [-1, 0, 1].map((d) => manilaDay(Date.now() + d * 864e5));
    if (
      !near.includes(id) ||
      !Array.isArray(data.items) ||
      data.items.length !== 10
    )
      return json({ error: "bad day" }, 400);
    await pool.query(
      "insert into dt_docs (path, data) values ($1, $2) on conflict (path) do nothing",
      [p, data],
    );
    return json({ ok: true });
  } else if (!me || !admins().includes(me)) {
    return json({ error: "admin only" }, 403);
  }
  await pool.query(
    "insert into dt_docs (path, data) values ($1, $2) on conflict (path) do update set data = excluded.data, updated_at = now()",
    [p, data],
  );
  return json({ ok: true });
}

export async function POST(req) {
  await init();
  const me = who(req);
  if (!me || new URL(req.url).searchParams.get("act") !== "link")
    return json({ error: "no" }, 400);
  const code = randomBytes(6).toString("hex");
  await pool.query(
    "delete from dt_codes where created_at < now() - interval '1 day'",
  );
  await pool.query("insert into dt_codes (code, player_id) values ($1, $2)", [
    code,
    me,
  ]);
  const bot = process.env.DT_BOT_USERNAME;
  if (!bot) return json({ error: "bot not configured" }, 503);
  return json({ url: `https://t.me/${bot}?start=${code}` });
}
