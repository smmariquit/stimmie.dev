/* Stand-in for the claude.ai runtime the Daily Ten page was built against.
   Same db/user surface the page calls, backed by /api/dt (Postgres).
   Identity is a random secret kept in localStorage; the player id is its hash. */
(() => {
  const API = '/api/dt';
  const KEY = 'dt-secret';

  const hex = buf => [...new Uint8Array(buf)].map(b => b.toString(16).padStart(2, '0')).join('');
  let secret = null;
  try {
    secret = localStorage.getItem(KEY);
    if (!secret) {
      secret = hex(crypto.getRandomValues(new Uint8Array(32)));
      localStorage.setItem(KEY, secret);
    }
  } catch { secret = null; }

  const idP = secret
    ? crypto.subtle.digest('SHA-256', new TextEncoder().encode(secret)).then(h => hex(h).slice(0, 24))
    : Promise.reject(new Error('no storage'));

  async function call(qs, opt = {}) {
    const r = await fetch(API + '?' + qs, {
      ...opt,
      headers: { 'content-type': 'application/json', 'x-dt-secret': secret || '' },
    });
    if (!r.ok) {
      const e = new Error('dt ' + r.status);
      e.code = r.status === 403 ? 'invalid_argument' : r.status === 413 ? 'quota_exceeded' : 'unavailable';
      throw e;
    }
    return r.json();
  }

  const snapDoc = (id, data) => ({ id, exists: data != null, data: () => data });
  const poll = (fn, ok, bad, ms = 8000) => {
    let last = null, dead = false;
    const run = async () => {
      if (dead || document.hidden) return;
      try {
        const v = await fn(), s = JSON.stringify(v);
        if (s !== last) { last = s; ok(v); }
      } catch (e) { bad && bad(e); }
    };
    run();
    const t = setInterval(run, ms);
    return () => { dead = true; clearInterval(t); };
  };

  const colSnap = rows => ({ docs: rows.map(r => snapDoc(r.id, r.data)) });

  const db = {
    collection(name) {
      const q = { name, n: 1000 };
      const get = async () => colSnap(await call('c=' + name + '&n=' + q.n));
      return {
        limit(n) { q.n = n; return this; },
        get,
        onSnapshot: (ok, bad) => poll(() => call('c=' + name + '&n=' + q.n), v => ok(colSnap(v)), bad),
      };
    },
    doc(path) {
      const id = path.split('/')[1];
      const get = async () => snapDoc(id, (await call('d=' + path)).data);
      return {
        get,
        set: data => call('d=' + path, { method: 'PUT', body: JSON.stringify(data) }),
        // the server keeps the first write for days/*, so no lock is needed
        acquire: async () => ({ acquired: true }),
        onSnapshot: (ok, bad) => poll(async () => (await call('d=' + path)).data, v => ok(snapDoc(id, v)), bad),
      };
    },
  };

  const user = {
    id: () => idP,
    me: async () => ({ name: '' }),
    canEdit: async () => !!(await call('act=me')).admin,
    tgStatus: async () => !!(await call('act=me')).linked,
    tgLink: async () => (await call('act=link', { method: 'POST', body: '{}' })).url,
  };

  window.claude = { use: async name => (name === 'db' ? db : name === 'user' ? user : Promise.reject(new Error('unknown ' + name))) };
})();
