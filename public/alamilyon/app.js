const ROUNDS = 7, SECONDS = 25, EPOCH = Date.UTC(2026, 10, 26), KEY = 'alamilyon.v1', DAY = 864e5;
const $ = id => document.getElementById(id), app = $('app');
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

const TIERS = ['Hindi lumubog', 'May ipon pa', 'Petsa de peligro', 'Nganga', 'Baon sa utang', 'Walang-wala'];
const tierIx = pts => pts >= 100 ? 5 : pts >= 75 ? 4 : pts >= 50 ? 3 : pts >= 25 ? 2 : pts > 0 ? 1 : 0;

World.onDepth = d => { $('hd').textContent = `${fmt(Math.round(d))} m`; };
function hud() {
  $('hud').hidden = false; app.className = '';
  $('hs').textContent = `${meters() / 10} pts`;
  $('hp').setAttribute('aria-label', `Tanong ${Math.min(S.cur + 1, ROUNDS)} sa ${ROUNDS}`);
  $('hp').innerHTML = Array.from({ length: ROUNDS }, (_, i) =>
    `<li class="${i < S.results.length ? 't' + tierIx(S.results[i].pts) : i === S.cur ? 'cur' : ''}"></li>`).join('');
}

function start() {
  World.scene('start'); app.className = 'is-start';
  app.innerHTML = `<section class="start">
    <div class="sky"><h1 class="mark">Alamilyon</h1><p class="sub">sisid sa kahirapan</p></div>
    <div class="under">
      <p class="lede">7 tanong tungkol sa Pilipinas, 25 seconds bawat isa. Mas bihira ang sagot mo, mas malalim ang lubog mo.</p>
      <button id="go">▼ Sumisid ▼</button>
      <p class="fine">Laro #${S.day}. Hindi pa galing sa totoong players ang mga bilang, kaya puwedeng magbago ang points.</p>
    </div>
  </section>`;
  $('go').onclick = round;
}

function round() {
  const p = Q[S.cur];
  S.pending = S.started = true; save(); hud(); World.scene('play');
  app.innerHTML = `<form class="panel round" autocomplete="off">
      <label class="prompt" for="ans">${esc(p.q)}</label>
      <p class="hint">Isang sagot lang. Mas bihira, mas malalim ang lubog.</p>
      <div class="field"><input id="ans" maxlength="60" autocapitalize="off" spellcheck="false" placeholder="sagot mo"><button>Sagot</button></div>
      <div class="clock" aria-hidden="true"><i><b></b></i><span>${SECONDS}</span></div>
    </form>`;
  const input = $('ans'), clock = app.querySelector('.clock'), bar = clock.querySelector('b'), num = clock.querySelector('span');
  input.focus();
  const t0 = performance.now();
  const id = setInterval(() => {
    const left = SECONDS - (performance.now() - t0) / 1000;
    bar.style.transform = `scaleX(${Math.max(0, left / SECONDS)})`;
    num.textContent = Math.max(0, Math.ceil(left));
    clock.classList.toggle('low', left <= 5);
    if (left <= 0) done('');
  }, 100);
  const done = text => { clearInterval(id); answer(p, text.trim()); };
  app.querySelector('form').onsubmit = e => { e.preventDefault(); done(input.value); };
}

function answer(p, text) {
  const hit = match(p, text), from = meters();
  const r = { q: p.q, text, name: hit ? hit.name : null, pts: hit ? points(hit.c, p.max) : 0 };
  if (text && !hit) {
    const u = JSON.parse(localStorage.getItem('alamilyon.unlisted') || '[]');
    u.push({ id: p.id, text, day: S.day });
    localStorage.setItem('alamilyon.unlisted', JSON.stringify(u));
  }
  S.results.push(r); S.cur++; S.pending = false; save(); hud();
  if (!r.pts) return feedback(p, r);
  app.innerHTML = '<p class="sr" role="status">Lumulubog ka na.</p>';
  World.sink(from, from + r.pts * 10, r.name, () => feedback(p, r));
}

const rarity = pts => pts >= 100 ? 'Pambihira. Halos walang nakaisip nito.' : pts >= 70 ? 'Bihira ang sagot na yan.' : pts >= 40 ? 'Medyo bihira ang sagot mo.' : 'Maraming sumagot nito.';
const list = a => a.length < 2 ? a.join('') : a.slice(0, -1).join(', ') + ' at ' + a[a.length - 1];

function feedback(p, r) {
  const top = p.a.slice().sort((a, b) => b[1] - a[1]).slice(0, 3).map(x => x[0]);
  const last = S.cur >= ROUNDS, ix = tierIx(r.pts);
  World.scene('fb');
  const body = r.name
    ? `<h2 class="tier t${ix}">${TIERS[ix]}</h2><p class="said">“${esc(r.text)}”</p>
       <p class="gain">+${r.pts} points, ${fmt(r.pts * 10)} m pababa</p>
       <p class="note">${rarity(r.pts)}${norm(r.name) !== norm(r.text) ? ` Binilang namin na ${esc(r.name)}.` : ''}</p>`
    : `<h2 class="tier t0">${r.text ? 'Wala sa lista' : 'Naubusan ng oras'}</h2>
       <p class="note">${r.text ? `Wala pa sa listahan namin ang “${esc(r.text)}”, kaya hindi ka lumubog.` : 'Wala kang naisagot, kaya hindi ka lumubog.'}</p>`;
  app.innerHTML = `<section class="fb" aria-live="polite">${body}
      <p class="common">Madalas sagot dito ang ${esc(list(top))}.</p>
    </section>
    <button class="next" id="next">${last ? 'Tingnan ang lalim' : 'Susunod ▼'}</button>`;
  $('next').focus();
  $('next').onclick = () => last ? results() : round();
}

function results() {
  const m = meters(), ix = m ? tierIx(m / 70) : 0, bars = '▁▂▃▄▅▆▇█';
  hud(); World.scene('end');
  app.innerHTML = `<section class="end">
      <h1 class="score">${fmt(m)}<small>m</small></h1><p class="tier t${ix}">${TIERS[ix]}</p>
      <p class="note">Ganyan kalalim ang lubog mo sa kahirapan ngayon.</p>
    </section>
    <ol class="rows panel">${S.results.map(r => `<li>
      <span class="q">${esc(r.q)}</span>
      <span class="a">${r.text ? esc(r.text) : 'Walang sagot'}</span><span class="p t${tierIx(r.pts)}">${r.pts}</span>
      <span class="meter t${tierIx(r.pts)}"><i style="transform:scaleX(${r.pts / 100})"></i></span></li>`).join('')}</ol>
    <div class="actions"><button id="share">Kopyahin ang score</button></div>
    <p class="fine">Bagong laro sa loob ng <span id="cd"></span>.</p>`;
  const share = $('share');
  share.onclick = async () => {
    const line = S.results.map(r => bars[Math.min(7, Math.floor(r.pts / 12.5))]).join('');
    await navigator.clipboard.writeText(`Alamilyon #${S.day}\nLumubog ako ng ${fmt(m)} m sa kahirapan (${TIERS[ix]})\n${line}\n${location.href.split(/[?#]/)[0]}`);
    share.textContent = 'Nakopya na';
    setTimeout(() => share.textContent = 'Kopyahin ang score', 2000);
  };
  const cd = $('cd');
  const tick = () => {
    const s = Math.floor((DAY - manila() % DAY) / 1000), p = n => String(n).padStart(2, '0');
    cd.textContent = `${p(Math.floor(s / 3600))}:${p(Math.floor(s % 3600 / 60))}:${p(s % 60)}`;
  };
  tick(); setInterval(tick, 1000);
}

if (S.pending) { S.results.push({ q: Q[S.cur].q, text: '', name: null, pts: 0 }); S.cur++; S.pending = false; save(); }
World.jump(meters());
if (S.results.length >= ROUNDS) results(); else if (S.started) round(); else start();
