const ROUNDS = 7, SECONDS = 25, EPOCH = Date.UTC(2026, 10, 26), KEY = 'alamilyon.v1', DAY = 864e5;
const app = document.getElementById('app'), deep = document.getElementById('deep');
const esc = s => s.replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmt = n => n.toLocaleString('en-US');
const manila = () => Date.now() + 8 * 36e5;
const dayNo = () => Math.max(1, Math.floor((manila() - EPOCH) / DAY) + 1);

const norm = s => s.toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, '').replace(/&/g, ' and ')
  .replace(/[^a-z0-9]+/g, ' ').trim().replace(/^(ang|ng|mga|sa) /, '');
const key = s => norm(s).replace(/ /g, '');

function rng(a) {
  return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
}
function todays(day) {
  const r = rng(day * 7919), ix = CORPUS.map((_, i) => i);
  for (let i = ix.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [ix[i], ix[j]] = [ix[j], ix[i]]; }
  return ix.slice(0, ROUNDS).map(i => CORPUS[i]);
}

function index(p) {
  if (p.map) return p.map;
  const m = new Map(); p.max = 0;
  for (const [name, c, al] of p.a) {
    p.max = Math.max(p.max, c);
    for (const t of [name, ...(al ? al.split('/') : [])]) m.set(key(t), { name, c });
  }
  return p.map = m;
}
function lev(a, b) {
  if (Math.abs(a.length - b.length) > 2) return 9;
  let p = [...Array(b.length + 1).keys()];
  for (let i = 1; i <= a.length; i++) {
    const c = [i];
    for (let j = 1; j <= b.length; j++) c[j] = Math.min(p[j] + 1, c[j - 1] + 1, p[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
    p = c;
  }
  return p[b.length];
}
function match(p, text) {
  const m = index(p), k = key(text);
  if (!k) return null;
  if (m.has(k)) return m.get(k);
  const tol = k.length < 5 ? 0 : k.length < 9 ? 1 : 2;
  let best = null, bd = tol + 1;
  for (const [kk, v] of m) { const d = lev(k, kk); if (d < bd) { bd = d; best = v; } }
  return best;
}
const points = (c, max) => c <= 1 ? 100 : Math.max(5, Math.round(100 * (1 - Math.sqrt(c / max))));

let S = load();
const Q = todays(S.day);
function load() {
  try { const s = JSON.parse(localStorage.getItem(KEY)); if (s && s.day === dayNo()) return s; } catch {}
  return { day: dayNo(), cur: 0, results: [], pending: false, started: false };
}
const save = () => localStorage.setItem(KEY, JSON.stringify(S));
const meters = () => S.results.reduce((a, r) => a + r.pts * 10, 0);
const setDepth = () => { deep.style.opacity = Math.min(1, meters() / (ROUNDS * 1000)); };
const header = n => `<header class="bar"><span>${n} / ${ROUNDS}</span><span class="depth">${fmt(meters())} m</span></header>`;

function start() {
  app.innerHTML = `<section class="start">
    <h1 class="mark">Alami<span>lyon</span></h1>
    <p class="lede">Pitong tanong tungkol sa Pilipinas. Isang sagot lang bawat tanong, 25 seconds. Mas bihira ang sagot mo, mas malalim ang sisid mo.</p>
    <button id="go">Sumisid</button>
    <p class="fine">Laro #${S.day}. Hindi pa galing sa totoong players ang mga bilang, kaya puwedeng magbago ang points.</p>
  </section>`;
  document.getElementById('go').onclick = round;
}

function round() {
  const p = Q[S.cur];
  S.pending = S.started = true; save(); setDepth();
  app.innerHTML = `${header(S.cur + 1)}
    <div class="timer" aria-hidden="true"><i></i></div>
    <form class="round" autocomplete="off">
      <label class="prompt" for="ans">${esc(p.q)}</label>
      <p class="hint">Isang sagot lang. Mas bihira, mas malalim.</p>
      <div class="field"><input id="ans" maxlength="60" autocapitalize="off" spellcheck="false" placeholder="i-type ang sagot mo"><button>Sagot</button></div>
    </form>`;
  const input = document.getElementById('ans'), timer = app.querySelector('.timer'), bar = timer.firstChild;
  input.focus();
  const t0 = performance.now();
  const id = setInterval(() => {
    const left = SECONDS - (performance.now() - t0) / 1000;
    bar.style.transform = `scaleX(${Math.max(0, left / SECONDS)})`;
    timer.classList.toggle('low', left <= 5);
    if (left <= 0) done('');
  }, 100);
  const done = text => { clearInterval(id); answer(p, text.trim()); };
  app.querySelector('form').onsubmit = e => { e.preventDefault(); done(input.value); };
}

function answer(p, text) {
  const hit = match(p, text);
  const r = { q: p.q, text, name: hit ? hit.name : null, pts: hit ? points(hit.c, p.max) : 0 };
  if (text && !hit) {
    const u = JSON.parse(localStorage.getItem('alamilyon.unlisted') || '[]');
    u.push({ id: p.id, text, day: S.day });
    localStorage.setItem('alamilyon.unlisted', JSON.stringify(u));
  }
  S.results.push(r); S.cur++; S.pending = false; save(); setDepth();
  feedback(p, r);
}

const rarity = pts => pts >= 100 ? 'Pambihira. Halos walang nakaisip nito.' : pts >= 70 ? 'Bihira ang sagot na yan.' : pts >= 40 ? 'Medyo bihira ang sagot mo.' : 'Maraming sumagot nito.';
const list = a => a.length < 2 ? a.join('') : a.slice(0, -1).join(', ') + ' at ' + a[a.length - 1];

function feedback(p, r) {
  const top = p.a.slice().sort((a, b) => b[1] - a[1]).slice(0, 3).map(x => x[0]);
  const last = S.cur >= ROUNDS;
  const body = r.name
    ? `<p class="pts">+${r.pts}</p><p class="meters">${fmt(r.pts * 10)} m pababa</p>
       <p class="note">${rarity(r.pts)}${norm(r.name) !== norm(r.text) ? ` Binilang namin na ${esc(r.name)}.` : ''}</p>`
    : `<p class="pts miss">0</p><p class="note">${r.text ? `Wala pa sa listahan namin ang "${esc(r.text)}".` : 'Naubusan ka ng oras.'}</p>`;
  app.innerHTML = `${header(S.cur)}
    <section class="card" aria-live="polite">
      <p class="said">${esc(p.q)}</p>${body}
      <p class="common">Madalas sagot dito ang ${esc(list(top))}.</p>
    </section>
    <button class="next" id="next">${last ? 'Tingnan ang depth' : 'Susunod'}</button>`;
  const next = document.getElementById('next');
  next.focus();
  next.onclick = () => last ? results() : round();
}

const tier = m => m < 1500 ? 'Tampisaw' : m < 3000 ? 'Laot' : m < 4500 ? 'Malalim na' : m < 6000 ? 'Sobrang lalim' : 'Wala nang liwanag';

function results() {
  const m = meters(), bars = '▁▂▃▄▅▆▇█';
  setDepth();
  app.innerHTML = `<section>
      <h1 class="score">${fmt(m)}<small>m</small></h1><p class="tier">${tier(m)}</p>
    </section>
    <ol class="rows">${S.results.map(r => `<li>
      <span class="q">${esc(r.q)}</span>
      <span class="a">${r.text ? esc(r.text) : 'Walang sagot'}</span><span class="p">${r.pts}</span>
      <span class="meter"><i style="transform:scaleX(${r.pts / 100})"></i></span></li>`).join('')}</ol>
    <div class="actions"><button id="share">Kopyahin ang score</button></div>
    <p class="fine">Bagong laro sa loob ng <span id="cd"></span>.</p>`;
  const share = document.getElementById('share');
  share.onclick = async () => {
    const line = S.results.map(r => bars[Math.min(7, Math.floor(r.pts / 12.5))]).join('');
    await navigator.clipboard.writeText(`Alamilyon #${S.day}\nSumisid ako ng ${fmt(m)} m\n${line}\n${location.href.split(/[?#]/)[0]}`);
    share.textContent = 'Nakopya na';
    setTimeout(() => share.textContent = 'Kopyahin ang score', 2000);
  };
  const cd = document.getElementById('cd');
  const tick = () => {
    const s = Math.floor((DAY - manila() % DAY) / 1000), p = n => String(n).padStart(2, '0');
    cd.textContent = `${p(Math.floor(s / 3600))}:${p(Math.floor(s % 3600 / 60))}:${p(s % 60)}`;
  };
  tick(); setInterval(tick, 1000);
}

if (S.pending) { S.results.push({ q: Q[S.cur].q, text: '', name: null, pts: 0 }); S.cur++; S.pending = false; save(); }
setDepth();
if (S.results.length >= ROUNDS) results(); else if (S.started) round(); else start();
