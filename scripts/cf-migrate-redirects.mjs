#!/usr/bin/env node
/**
 * Sync src/data/redirects.js into a Cloudflare Bulk Redirect List, then
 * ensure the account-level Redirects ruleset has a rule that consults that
 * list. Idempotent — safe to re-run after editing the source map.
 *
 * Usage:
 *   CLOUDFLARE_API_TOKEN=... \
 *   CLOUDFLARE_ACCOUNT_ID=... \
 *   npm run cf:migrate-redirects [-- --dry-run] [-- --prune]
 *
 * Required token scopes (Account-scoped):
 *   - Account Filter Lists: Edit       (create/edit Bulk Redirect Lists)
 *   - Account Rulesets: Edit           (attach rule to redirect ruleset)
 *
 * Flags:
 *   --dry-run   Print the diff but don't mutate anything in CF.
 *   --prune     Delete CF items whose source URL is no longer in the
 *               local map. Default is to leave them alone — safer.
 *   --list-name Override the CF list name. Default: stimmie_redirects.
 *
 * Notes on shape:
 *   - One logical redirect produces TWO CF items (apex /r/<slug> form
 *     AND <category>.stimmie.dev/<slug> form), so visiting either URL
 *     hits the redirect at the edge.
 *   - Items with the TODO sentinel target are skipped (and pruned if
 *     --prune is set, so removing an entry from the map does the right
 *     thing on next sync).
 */

import { pathToFileURL } from "node:url";
import { resolve } from "node:path";

const CF_API = "https://api.cloudflare.com/client/v4";
const STATUS_CODE = 301;

// CF Bulk Redirect list names: lowercase letters, numbers, underscore only;
// must start with a letter. Keep in sync with the rule expression below.
const DEFAULT_LIST_NAME = "stimmie_redirects";
const RULE_DESCRIPTION = "stimmie.dev short-link redirects (auto-synced)";

const args = parseArgs(process.argv.slice(2));
if (args.help) {
  printHelp();
  process.exit(0);
}

const token = requireEnv("CLOUDFLARE_API_TOKEN");
const accountId = requireEnv("CLOUDFLARE_ACCOUNT_ID");
const listName = args.listName || DEFAULT_LIST_NAME;

try {
  await main();
} catch (err) {
  process.stderr.write(`\n[cf-migrate-redirects] ${err.message}\n`);
  if (/403|Authentication|9109/.test(err.message)) {
    process.stderr.write(
      "\nHint: this script needs an account-scoped token with:\n" +
        "  - Account → Account Filter Lists: Edit\n" +
        "  - Account → Account Rulesets: Edit\n\n" +
        "Mint one at https://dash.cloudflare.com/profile/api-tokens.\n"
    );
  }
  process.exit(1);
}

async function main() {
  // Load the redirect map from the same module the runtime uses. Node's
  // ESM loader resolves .js files; on Node < 23 you may need the
  // --experimental-detect-module flag (the npm script sets it).
  const redirectsModulePath = resolve(process.cwd(), "src/data/redirects.js");
  const { flattenRedirects, TODO_TARGET } = await import(
    pathToFileURL(redirectsModulePath).href
  );

  const desired = buildDesiredItems(flattenRedirects(), TODO_TARGET);
  log(`local redirects ready: ${desired.size} item(s)`);

  const list = await ensureList(listName);
  log(`using CF list: ${list.name} (${list.id})`);

  const current = await loadCurrentItems(list.id);
  log(`CF currently has ${current.size} item(s) in this list`);

  const plan = diff(current, desired, { prune: args.prune });
  printPlan(plan);

  if (args.dryRun) {
    log("--dry-run set, exiting without mutations.");
    return;
  }

  if (plan.toAdd.length === 0 && plan.toDelete.length === 0) {
    log("No item changes needed.");
  } else {
    if (plan.toDelete.length) {
      await deleteItems(list.id, plan.toDelete);
    }
    if (plan.toAdd.length) {
      await addItems(list.id, plan.toAdd);
    }
  }

  await ensureRuleAttached(list);
  log("Done.");
}

// --- helpers -------------------------------------------------------------

function parseArgs(argv) {
  const out = { dryRun: false, prune: false };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--dry-run" || a === "-n") out.dryRun = true;
    else if (a === "--prune") out.prune = true;
    else if (a === "--list-name") out.listName = argv[++i];
    else if (a === "-h" || a === "--help") out.help = true;
    else throw new Error(`Unknown arg: ${a}`);
  }
  return out;
}

function printHelp() {
  process.stdout.write(
    `\nSync src/data/redirects.js -> Cloudflare Bulk Redirects.\n\n` +
      `Env:\n` +
      `  CLOUDFLARE_API_TOKEN   (required) token with Account Filter Lists: Edit\n` +
      `                                    + Account Rulesets: Edit\n` +
      `  CLOUDFLARE_ACCOUNT_ID  (required) account that owns the list\n\n` +
      `Flags:\n` +
      `  --dry-run   plan only, no mutations\n` +
      `  --prune     delete CF items missing from the local map\n` +
      `  --list-name override list name (default: ${DEFAULT_LIST_NAME})\n` +
      `  -h, --help  this message\n\n`
  );
}

function requireEnv(key) {
  const v = process.env[key];
  if (!v) {
    process.stderr.write(`Missing required env var: ${key}\n`);
    process.exit(2);
  }
  return v;
}

function log(msg) {
  process.stdout.write(`[cf-migrate-redirects] ${msg}\n`);
}

/**
 * Convert flattened redirects into a Map keyed by canonical source_url.
 * Skips entries whose target is the TODO placeholder.
 */
function buildDesiredItems(flat, todoTarget) {
  const items = new Map();
  for (const entry of flat) {
    if (entry.placeholder || entry.target === todoTarget) continue;
    for (const src of entry.sources) {
      const sourceUrl = `${src.host}${src.path}`;
      if (items.has(sourceUrl)) {
        // Same source URL configured twice with different targets is a
        // bug in the local map — fail loudly rather than silently picking
        // a winner that the runtime won't agree with.
        const existing = items.get(sourceUrl);
        if (existing.target !== entry.target) {
          throw new Error(
            `Duplicate source URL "${sourceUrl}" with conflicting targets: ` +
              `"${existing.target}" vs "${entry.target}"`
          );
        }
        continue;
      }
      items.set(sourceUrl, {
        sourceUrl,
        target: entry.target,
        category: entry.category,
        slug: entry.slug,
      });
    }
  }
  return items;
}

/**
 * Find a redirect-kind list by name, or create it. Returns {id, name, kind}.
 */
async function ensureList(name) {
  const lists = await cf("GET", `/accounts/${accountId}/rules/lists`);
  const found = (lists || []).find((l) => l.name === name);
  if (found) return found;

  log(`creating CF list "${name}"…`);
  const created = await cf("POST", `/accounts/${accountId}/rules/lists`, {
    name,
    kind: "redirect",
    description: "stimmie.dev short links + subdomain redirects (auto-synced)",
  });
  return created;
}

/**
 * Page through every item in the list. Returns Map<source_url, {id, target}>.
 */
async function loadCurrentItems(listId) {
  const out = new Map();
  let cursor;
  for (let page = 0; page < 50; page++) {
    const qs = new URLSearchParams({ per_page: "100" });
    if (cursor) qs.set("cursor", cursor);
    const { result, result_info } = await cfRaw(
      "GET",
      `/accounts/${accountId}/rules/lists/${listId}/items?${qs}`
    );
    for (const item of result || []) {
      const r = item?.redirect;
      if (!r?.source_url) continue;
      out.set(r.source_url, { id: item.id, target: r.target_url });
    }
    cursor = result_info?.cursors?.after;
    if (!cursor) break;
  }
  return out;
}

/**
 * Compute add/delete sets. Updates are modeled as delete-then-add.
 */
function diff(current, desired, { prune }) {
  const toAdd = [];
  const toDelete = [];

  for (const [src, want] of desired) {
    const have = current.get(src);
    if (!have) {
      toAdd.push(want);
    } else if (have.target !== want.target) {
      toDelete.push({ ...have, sourceUrl: src });
      toAdd.push(want);
    }
  }

  if (prune) {
    for (const [src, have] of current) {
      if (!desired.has(src)) toDelete.push({ ...have, sourceUrl: src });
    }
  }

  return { toAdd, toDelete };
}

function printPlan(plan) {
  log(`plan: +${plan.toAdd.length} add, -${plan.toDelete.length} delete`);
  for (const it of plan.toAdd) {
    log(`  + ${it.sourceUrl}  ->  ${it.target}`);
  }
  for (const it of plan.toDelete) {
    log(`  - ${it.sourceUrl}  (id ${it.id})`);
  }
}

async function addItems(listId, items) {
  // CF caps a single POST at 1000 items; chunk to be safe.
  const chunks = chunk(items, 500);
  for (const c of chunks) {
    const body = c.map((it) => ({
      redirect: {
        source_url: it.sourceUrl,
        target_url: it.target,
        status_code: STATUS_CODE,
        preserve_query_string: true,
        include_subdomains: false,
        subpath_matching: false,
        preserve_path_suffix: false,
      },
    }));
    log(`adding ${body.length} item(s)…`);
    const op = await cf(
      "POST",
      `/accounts/${accountId}/rules/lists/${listId}/items`,
      body
    );
    await waitForOperation(op);
  }
}

async function deleteItems(listId, items) {
  const chunks = chunk(items, 500);
  for (const c of chunks) {
    const body = { items: c.map((it) => ({ id: it.id })) };
    log(`deleting ${c.length} item(s)…`);
    const op = await cf(
      "DELETE",
      `/accounts/${accountId}/rules/lists/${listId}/items`,
      body
    );
    await waitForOperation(op);
  }
}

/**
 * Async list mutations return an operation_id; poll until it completes.
 */
async function waitForOperation(opResult) {
  const id = opResult?.operation_id;
  if (!id) return; // some sync paths don't return one
  for (let i = 0; i < 60; i++) {
    const op = await cf(
      "GET",
      `/accounts/${accountId}/rules/lists/bulk_operations/${id}`
    );
    if (op.status === "completed") return;
    if (op.status === "failed") {
      throw new Error(`CF list operation ${id} failed: ${op.error || "unknown"}`);
    }
    await sleep(1000);
  }
  throw new Error(`CF list operation ${id} timed out after 60s`);
}

/**
 * Ensure the account-level http_request_redirect ruleset has a rule that
 * applies our list. Idempotent: matches by description.
 */
async function ensureRuleAttached(list) {
  const phase = "http_request_redirect";
  const entrypoint = await cf(
    "GET",
    `/accounts/${accountId}/rulesets/phases/${phase}/entrypoint`
  ).catch((err) => {
    // The entrypoint ruleset is auto-created on first use; if it doesn't
    // exist yet, fall back to creating a fresh ruleset and walking the
    // standard create flow.
    if (/(404|not.*found|no route)/i.test(String(err.message))) return null;
    throw err;
  });

  const expression = `http.request.full_uri in $${list.name}`;
  const ruleBody = {
    action: "redirect",
    action_parameters: {
      from_list: { name: list.name, key: "http.request.full_uri" },
    },
    expression,
    description: RULE_DESCRIPTION,
    enabled: true,
  };

  if (!entrypoint) {
    log(`creating new ${phase} ruleset and attaching rule…`);
    await cf("POST", `/accounts/${accountId}/rulesets`, {
      name: "default",
      kind: "root",
      phase,
      rules: [ruleBody],
    });
    return;
  }

  const existing = (entrypoint.rules || []).find(
    (r) => r.description === RULE_DESCRIPTION
  );

  if (existing) {
    // Re-PUT in case the expression or list-name changed.
    if (
      existing.expression === expression &&
      existing.action_parameters?.from_list?.name === list.name &&
      existing.enabled
    ) {
      log("rule already attached and current.");
      return;
    }
    log("updating existing rule…");
    await cf(
      "PATCH",
      `/accounts/${accountId}/rulesets/${entrypoint.id}/rules/${existing.id}`,
      ruleBody
    );
    return;
  }

  log("attaching rule to redirect ruleset…");
  await cf(
    "POST",
    `/accounts/${accountId}/rulesets/${entrypoint.id}/rules`,
    ruleBody
  );
}

function chunk(arr, size) {
  const out = [];
  for (let i = 0; i < arr.length; i += size) out.push(arr.slice(i, i + size));
  return out;
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function cf(method, path, body) {
  const json = await cfRaw(method, path, body);
  return json.result;
}

async function cfRaw(method, path, body) {
  const url = path.startsWith("http") ? path : `${CF_API}${path}`;
  const init = {
    method,
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    },
  };
  if (body !== undefined) init.body = JSON.stringify(body);

  const res = await fetch(url, init);
  const text = await res.text();
  let json;
  try {
    json = text ? JSON.parse(text) : {};
  } catch {
    throw new Error(`CF non-JSON ${res.status} ${res.statusText}: ${text}`);
  }
  if (!res.ok || json.success === false) {
    const msg = (json.errors || [])
      .map((e) => `${e.code}: ${e.message}`)
      .join("; ");
    throw new Error(
      `CF ${method} ${path} -> ${res.status} ${res.statusText}: ${msg || text}`
    );
  }
  return json;
}
