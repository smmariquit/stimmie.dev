const World = (() => {
  const cv = document.getElementById('world'), cx = cv.getContext('2d');
  const U = 3, PPM = 2, FLOOR = 7000, BROWNOUT = 3800;
  const still = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const ANCHOR = { start: 0.52, play: 0.62, fb: 0.26 };
  const rest = () => scene === 'end' ? 100 / H : ANCHOR[scene];
  const INK = '255,240,214', GOLD = '255,215,94', PINK = '255,122,154';
  const SKY = [[0, [24, 20, 70]], [.35, [70, 40, 120]], [.6, [160, 68, 140]], [.8, [238, 108, 124]], [.94, [255, 178, 106]], [1, [255, 214, 140]]];
  const DEEP = [[0, [112, 58, 84]], [350, [78, 44, 84]], [1200, [52, 36, 82]], [2500, [30, 24, 58]], [3700, [16, 13, 34]], [3900, [9, 7, 20]], [FLOOR, [4, 3, 9]]];
  // [depth, text, zone?, sign?]
  const MARKS = [
    [120, 'dito natatapos ang sweldo'],
    [300, 'huling milk tea mo na to'],
    [480, 'PETSA DE PELIGRO ZONE', 1],
    [680, 'jeep na lang, wala nang Grab'],
    [900, '"libre mo naman ako" territory'],
    [1150, 'GCash mo, 12 pesos na lang'],
    [1400, 'pancit canton ulit mamayang gabi'],
    [1700, 'UTANG ZONE', 1],
    [1950, 'may lista ka na sa tindahan', 0, 'SARI-SARI'],
    [2250, '"bayaran kita sa sweldo"'],
    [2600, 'zero load. piso wifi na lang', 0, 'PISO WIFI'],
    [3000, 'SANGLA ZONE', 1, 'SANGLAAN'],
    [3300, 'nakasangla na ang phone mo'],
    [BROWNOUT, 'BROWNOUT ZONE', 1],
    [3950, 'naputulan ka na ng kuryente'],
    [4300, 'kandila at pamaypay na lang'],
    [4700, '5-6 territory', 0, '5-6'],
    [5100, 'tuyo at kanin, three times a day'],
    [5500, 'naniningil na ang landlady'],
    [6000, 'WALANG-WALA ZONE', 1],
    [6400, 'basag na pati ang alkansya'],
    [FLOOR, 'ang pinakailalim. perfect na lubog.'],
  ];
  const STEPS = [[25, 'PETSA DE PELIGRO'], [50, 'NGANGA'], [75, 'BAON SA UTANG'], [100, 'WALANG-WALA']];

  let W = 0, H = 0, T = 0, last = 0, depth = 0, anchor = ANCHOR.start, scene = 'start';
  let sink = null, lines = null, tag = null, skyline = null, onDepth = () => {}, coinAt = 0;
  const coins = [], drift = Array.from({ length: 9 }, (_, i) => ({ i, x: -999, d: 0 }));

  const hash = n => { let t = Math.imul(n ^ 0x9E3779B9, 0x85EBCA6B); t ^= t >>> 13; t = Math.imul(t, 0xC2B2AE35); return ((t ^ t >>> 16) >>> 0) / 4294967296; };
  const ramp = (stops, x) => {
    let i = 1; while (i < stops.length - 1 && x > stops[i][0]) i++;
    const [a, ca] = stops[i - 1], [b, cb] = stops[i], t = Math.min(1, Math.max(0, (x - a) / (b - a)));
    return ca.map((v, k) => Math.round(v + (cb[k] - v) * t));
  };
  const rgb = (c, a = 1) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
  // dark silhouettes up top, faint lighter ones once the lights are out
  const sil = (d, k, lift) => { const t = Math.min(1, Math.max(0, (d - 3500) / 400)); return rgb(ramp(DEEP, d).map(v => Math.round(v * (1 - k) * (1 - t) + (v + lift) * t))); };
  const Y = d => anchor * H + (d - depth) * PPM;
  const dAt = y => depth + (y - anchor * H) / PPM;

  function canvas(w, h) { const c = document.createElement('canvas'); c.width = w; c.height = h; return [c, c.getContext('2d')]; }
  function sprite(rows, pal) {
    const [c, g] = canvas(Math.max(...rows.map(r => r.length)), rows.length);
    rows.forEach((r, y) => [...r].forEach((ch, x) => { if (pal[ch]) { g.fillStyle = pal[ch]; g.fillRect(x, y, 1, 1); } }));
    return c;
  }
  function put(s, x, y, flip) {
    x = Math.round(x); y = Math.round(y);
    if (!flip) return cx.drawImage(s, x, y, s.width * U, s.height * U);
    cx.save(); cx.translate(x + s.width * U, y); cx.scale(-1, 1); cx.drawImage(s, 0, 0, s.width * U, s.height * U); cx.restore();
  }

  const PIG = sprite([
    '.......kkkk.d.d...',
    '....ppppppppdpdd..',
    '..ppplllpppppppp..',
    '.pppllppppppppppp.',
    'dppplppppppppwkpp.',
    '.dpppppppppppppplll',
    'd.ppppppppppppppklk',
    '..pppppppppppppplll',
    '..ppppppppppppppp.',
    '...ppppppppppppp..',
    '...dd.dd...dd.dd..',
    '...dd.dd...dd.dd..',
  ], { p: '#ff7a9a', d: '#c9486e', l: '#ffb3c4', k: '#3a1430', w: '#fff' });
  const COIN = [
    ['.ggg.', 'gyyyg', 'gyoyg', 'gyyyg', '.ggg.'],
    ['..g..', '.gyg.', '.gog.', '.gyg.', '..g..'],
    ['..g..', '..y..', '..y..', '..y..', '..g..'],
  ].map(r => sprite(r, { g: '#d99a1c', y: '#ffd75e', o: '#fff2b0' }));
  const BIRD = [['x.....x', '.xx.xx.', '...x...'], ['.......', 'xxx.xxx', '...x...']].map(r => sprite(r, { x: '#2a1638' }));
  const BAT = [['x..x.x..x', 'xxxxxxxxx', '.xx.x.xx.', '....x....'], ['....x....', '.xxxxxxx.', 'xx.xxx.xx', 'x...x...x']].map(r => sprite(r, { x: '#3a2c56' }));
  const BAG = sprite(['.x..x.', '.x..x.', 'xxxxxx', 'xoooox', 'xoooox', 'xoooox', '.xoox.', '..xx..'], { x: 'rgba(255,240,230,.5)', o: 'rgba(255,240,230,.13)' });
  const BILL = sprite(['wwwww', 'wkkkw', 'wwwww', 'wkkww', 'wrrrw', 'wwwww', 'w.w.w'], { w: 'rgba(255,244,225,.8)', k: '#5a3c5a', r: '#e0486e' });
  const SHIRT = ['#c96a7a', '#d9b25e', '#7fa0b8', '#e8dccb'].map(c => sprite(['cc.cc', 'ccccc', '.ccc.', '.ccc.', '.ccc.'], { c }));
  const CLOUD = sprite([
    '........wwwww.........',
    '.....wwwwwwwwww.......',
    '...wwwwwwwwwwwwww.www.',
    '.wwwwwwwwwwwwwwwwwwwww',
    'wwwwwwwwwwwwwwwwwwwww.',
    '.sssssssssssssssssss..',
  ], { w: '#ffe3d0', s: '#e39ab0' });
  const JEEP = (() => {
    const [c, g] = canvas(38, 15), k = '#231033', body = '#f3e6d0';
    g.fillStyle = k; g.fillRect(3, 1, 24, 1); for (const x of [4, 14, 25]) g.fillRect(x, 0, 1, 2);
    g.fillStyle = '#ffd75e'; g.fillRect(2, 2, 26, 2);
    g.fillStyle = body; g.fillRect(1, 4, 28, 7); g.fillRect(29, 7, 7, 4);
    g.fillStyle = k; g.fillRect(36, 9, 1, 2); g.fillRect(0, 9, 1, 2); g.fillRect(29, 10, 7, 1);
    for (const x of [4, 9, 14, 19]) { g.fillRect(x, 5, 4, 3); g.fillStyle = '#ffb3c4'; g.fillRect(x + 1, 6, 2, 2); g.fillStyle = k; }
    g.fillStyle = '#8fd3e8'; g.fillRect(25, 5, 3, 3);
    g.fillStyle = '#ff7a9a'; g.fillRect(1, 9, 28, 1); g.fillRect(29, 8, 7, 1);
    for (const x of [5, 27]) { g.fillStyle = k; g.fillRect(x, 10, 5, 4); g.fillRect(x + 1, 14, 3, 1); g.fillStyle = body; g.fillRect(x + 2, 12, 1, 1); }
    g.fillStyle = '#fff2b0'; g.fillRect(35, 8, 1, 1);
    return c;
  })();

  function buildSkyline() {
    const w = Math.ceil(W / U), h = 46, [c, g] = canvas(w, h);
    let x = 0, n = 0;
    g.fillStyle = '#b85a86';
    while (x < w) {
      const bw = 5 + (hash(n * 3 + 1) * 8 | 0), bh = 12 + (hash(n * 3 + 2) * 30 | 0);
      g.fillRect(x, h - bh, bw, bh);
      if (hash(n * 3 + 3) > .7) g.fillRect(x + (bw >> 1), h - bh - 5, 1, 5);
      x += bw + (hash(n * 7) * 3 | 0); n++;
    }
    x = -2; n = 100;
    while (x < w) {
      const bw = 6 + (hash(n * 3 + 1) * 10 | 0), bh = 5 + (hash(n * 3 + 2) * 20 | 0);
      g.fillStyle = '#3b1c47'; g.fillRect(x, h - bh, bw, bh);
      g.fillStyle = '#ffd98a';
      for (let wy = h - bh + 2; wy < h - 2; wy += 3) for (let wx = x + 1; wx < x + bw - 1; wx += 2) if (hash(wx * 131 + wy * 17) > .62) g.fillRect(wx, wy, 1, 1);
      x += bw + (hash(n * 7) * 2 | 0); n++;
    }
    return c;
  }

  function resize() {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    W = cv.clientWidth; H = cv.clientHeight;
    cv.width = W * dpr; cv.height = H * dpr;
    cx.setTransform(dpr, 0, 0, dpr, 0, 0); cx.imageSmoothingEnabled = false;
    skyline = buildSkyline();
    for (const o of drift) respawn(o, true);
    if (still) draw();
  }

  function sky(sy) {
    for (let i = 0; i < 26; i++) {
      const y = hash(i * 5 + 1) * sy * .4;
      cx.fillStyle = `rgba(${INK},${.35 + .35 * Math.sin(T * 2 + i)})`;
      cx.fillRect(Math.round(hash(i * 5 + 2) * W), Math.round(y), 2, 2);
    }
    const sx = Math.round(W * .68), sun = Math.round(sy - 30 * U);
    for (const [r, c] of [[21, 'rgba(255,220,150,.22)'], [17, 'rgba(255,226,160,.4)'], [13, '#fff0b8']]) {
      cx.fillStyle = c;
      for (let j = -r; j <= r && sun + j * U < sy; j++) { const w = Math.round(Math.sqrt(r * r - j * j)); cx.fillRect(sx - w * U, sun + j * U, w * 2 * U, U); }
    }
    [[.1, .2, 5], [.62, .34, 3], [.3, .52, 7], [.85, .6, 4]].forEach(([fx, fy, v], i) => {
      const span = W + 80, x = ((fx * span + T * v) % span) - 70;
      cx.globalAlpha = .55 + fy * .6; put(CLOUD, x, sy * fy, i % 2); cx.globalAlpha = 1;
    });
    cx.drawImage(skyline, 0, Math.round(sy) - skyline.height * U, skyline.width * U, skyline.height * U);
    cx.fillStyle = '#241230';
    for (let x = 0; x < W; x += 7 * U) cx.fillRect(x, sy - 2 * U, U, 2 * U);
    cx.fillRect(0, sy - 3 * U, W, U);
    const jx = W * .12, jy = sy - JEEP.height * U + (Math.floor(T * 8) % 2);
    if (scene === 'start') put(PIG, jx + 6 * U, jy - PIG.height * U + U);
    put(JEEP, jx, jy);
    cx.fillStyle = '#241230'; cx.fillRect(0, sy, W, 2 * U);
    cx.fillStyle = '#3a1f48'; cx.fillRect(0, sy + 2 * U, W, U);
  }

  // drawing inside art-pixel space, mirrored for the right side
  function edge(s, y, fn) {
    cx.save();
    if (s) { cx.translate(W, 0); cx.scale(-1, 1); }
    cx.translate(0, Math.round(y)); cx.scale(U, U); fn(); cx.restore();
  }
  function wire(x0, y0, x1, y1, sag) {
    const n = Math.abs(x1 - x0);
    for (let k = 0; k <= n; k++) { const t = k / n; cx.fillRect(Math.round(x0 + (x1 - x0) * t), Math.round(y0 + (y1 - y0) * t + sag * 4 * t * (1 - t)), 1, 1); }
  }
  function win(x, y, w, h, d, seed, col) {
    const r = hash(seed);
    if (d < BROWNOUT) cx.fillStyle = r < .6 ? 'rgba(255,205,120,.85)' : 'rgba(0,0,0,.3)';
    else cx.fillStyle = r < .4 ? `rgba(255,150,60,${.3 + .12 * Math.sin(T * 7 + seed)})` : 'rgba(0,0,0,.3)';
    cx.fillRect(x, y, w, h);
    if (d >= BROWNOUT && r < .4) { cx.fillStyle = '#ffd75e'; cx.fillRect(x + (w >> 1), y + h - 2, 1, 1); cx.fillStyle = '#fff2b0'; cx.fillRect(x + (w >> 1), y + h - 3, 1, 1); }
    cx.fillStyle = col;
  }
  function near(i, s) {
    const d = i * 150 + s * 75 + 60, y = Y(d);
    if (y < -260 || y > H + 260) return;
    const seed = i * 16 + s, kind = hash(seed), a = hash(seed + 2), b = hash(seed + 4);
    const col = sil(d, .55, 15);
    edge(s, y, () => {
      cx.fillStyle = col;
      if (kind < .3) {
        const x = 7 + (a * 8 | 0);
        cx.fillRect(x, -30, 2, 62); cx.fillRect(x - 5, -26, 12, 1); cx.fillRect(x - 4, -21, 10, 1); cx.fillRect(x + 2, -17, 4, 6);
        cx.fillRect(x, -31, 9, 1); cx.fillRect(x + 8, -30, 2, 1);
        if (d < BROWNOUT) { cx.fillStyle = 'rgba(255,220,140,.9)'; cx.fillRect(x + 8, -29, 2, 1); cx.fillStyle = 'rgba(255,220,140,.08)'; cx.fillRect(x + 5, -28, 8, 14); cx.fillStyle = col; }
        wire(0, -22 + (b * 6 | 0), x - 5, -26, 3); wire(0, -14, x - 4, -21, 4); wire(x + 2, -26, x + 14, -12, 5); wire(x - 3, -21, x + 9, -6, 6);
        const far = Math.ceil(W / U);
        cx.globalAlpha = .45;
        wire(x + 6, -26, far, -26 + (a * 44 - 22 | 0), 10 + (b * 10 | 0));
        wire(x + 5, -21, far, -10 + (b * 50 - 20 | 0), 14 + (a * 8 | 0));
        cx.globalAlpha = 1;
      } else if (kind < .82) {
        const w = 13 + (a * 10 | 0), h = 12 + (b * 8 | 0);
        cx.fillRect(0, 0, w, h); cx.fillRect(0, -2, w + 3, 2); cx.fillRect(0, -3, w + 1, 1);
        cx.fillRect(2, h, 1, 7); cx.fillRect(w - 3, h, 1, 7); cx.fillRect(0, h + 3, w * .6 | 0, 1);
        if (a > .45) { cx.fillRect(w - 4, -13, 1, 10); cx.fillRect(w - 7, -12, 7, 1); cx.fillRect(w - 6, -10, 5, 1); }
        else { cx.fillRect(3, -9, 6, 5); cx.fillRect(4, -4, 1, 2); cx.fillRect(7, -4, 1, 2); }
        win(w - 9, 3, 5, 4, d, seed + 6, col); cx.fillRect(w - 7, 3, 1, 4);
        if (w > 18) win(2, 4, 4, 3, d, seed + 8, col);
      } else {
        const w = 24 + (a * 8 | 0);
        cx.fillRect(0, -4, 2, 14); wire(0, 0, w, 2, 1); cx.fillRect(w, -6, 1, 16);
        cx.globalAlpha = d < BROWNOUT ? .6 : .28;
        for (let k = 0; k < 4; k++) { const sx = 3 + k * 6; if (sx + 5 < w) cx.drawImage(SHIRT[(seed + k) % 4], sx, 2 + (k % 2)); }
        cx.globalAlpha = 1;
      }
    });
  }
  function far(i, s) {
    const y = anchor * H + (i * 240 + s * 120 - depth) * PPM * .6;
    if (y < -220 || y > H + 60) return;
    const seed = i * 16 + s + 9000, w = 14 + (hash(seed) * 12 | 0), h = 28 + (hash(seed + 1) * 30 | 0), d = dAt(y);
    const col = sil(d, .3, 7);
    edge(s, y, () => {
      cx.fillStyle = col;
      cx.fillRect(0, 0, w, h); cx.fillRect(2, -6, 5, 6); cx.fillRect(w - 3, -9, 1, 9);
      for (let r = 0; r * 6 + 4 < h - 3; r++) for (let c = 0; c * 5 + 3 < w - 2; c++) {
        const lit = hash(seed + r * 13 + c * 7) > .55;
        cx.fillStyle = d < BROWNOUT && lit ? 'rgba(255,205,120,.4)' : 'rgba(0,0,0,.22)';
        cx.fillRect(c * 5 + 2, r * 6 + 4, 3, 3);
      }
    });
  }

  function text(s, x, y, size, color, align = 'left') {
    cx.font = `${size}px VT323, monospace`; cx.textAlign = align; cx.textBaseline = 'middle';
    cx.fillStyle = color; cx.fillText(s, Math.round(x), Math.round(y));
  }
  function fit(s, size, max) { cx.font = `${size}px VT323, monospace`; return Math.max(13, Math.floor(size * Math.min(1, max / cx.measureText(s).width))); }
  function notched(x, y, w, h, n = U * 2) {
    x = Math.round(x); y = Math.round(y);
    cx.fillRect(x + n, y, w - 2 * n, h); cx.fillRect(x, y + n, w, h - 2 * n);
  }

  function marks() {
    if (scene === 'start') return;
    const quiet = sink ? .7 : scene === 'play' ? .42 : .13, room = W - 104;
    MARKS.forEach(([d, s, zone, sign], i) => {
      const y = Y(d); if (y < -40 || y > H + 40) return;
      if (lines && STEPS.some(([p]) => Math.abs(Y(lines.from + p * 10) - y) < 22)) return;
      const right = i % 2, size = fit(s, zone ? 22 : 18, room), col = `rgba(${zone ? GOLD : INK},${zone ? quiet + .3 : quiet})`;
      cx.fillStyle = col;
      if (right) { cx.fillRect(W - 76, Math.round(y), 18, 1); text(s, W - 82, y, size, col, 'right'); }
      else { cx.fillRect(10, Math.round(y), 22, 1); text(s, 38, y, size, col); }
      if (!sign) return;
      cx.font = '20px VT323, monospace';
      const sw = Math.ceil(cx.measureText(sign).width) + 20, sx = right ? 0 : W - 22 - sw, sy = Math.round(y) - 62, lit = d < BROWNOUT;
      cx.fillStyle = lit ? 'rgba(255,200,90,.12)' : 'transparent'; cx.fillRect(sx - 6, sy - 6, sw + 12, 42);
      cx.fillStyle = '#1b0f2c'; cx.fillRect(sx, sy, sw, 30);
      cx.fillStyle = lit ? `rgb(${GOLD})` : 'rgba(255,215,94,.3)';
      cx.fillRect(sx, sy, sw, 2); cx.fillRect(sx, sy + 28, sw, 2); cx.fillRect(sx, sy, 2, 30); cx.fillRect(sx + sw - 2, sy, 2, 30);
      text(sign, sx + sw / 2, sy + 15, 20, lit ? `rgb(${PINK})` : 'rgba(255,122,154,.35)', 'center');
    });
  }

  function ruler(sy) {
    const x = W - 14, top = Math.max(0, sy), d0 = Math.max(0, Math.ceil(dAt(top) / 20) * 20);
    cx.fillStyle = `rgba(${INK},.3)`; cx.fillRect(x, top, 1, H - top);
    for (let d = d0; Y(d) < H && d <= FLOOR; d += 20) {
      const y = Math.round(Y(d)), big = d % 100 === 0;
      cx.fillStyle = `rgba(${INK},${big ? .5 : .25})`; cx.fillRect(x - (big ? 8 : 4), y, big ? 8 : 4, 1);
      if (big && (scene === 'start' || Math.abs(y - anchor * H) > 12)) text(`${d}m`, x - 12, y, 16, `rgba(${INK},.55)`, 'right');
    }
    if (scene === 'start') return;
    const y = Math.round(anchor * H);
    cx.fillStyle = `rgb(${PINK})`;
    for (let k = 0; k < 4; k++) cx.fillRect(x - 3 - k * 2, y - k, 2, k * 2 + 1);
    text('IKAW', x - 14, y, 18, `rgb(${PINK})`, 'right');
  }

  function respawn(o, anywhere) {
    const dir = Math.random() < .5 ? -1 : 1;
    o.d = dAt(Math.random() * H); o.dir = dir; o.ph = Math.random() * 6;
    o.x = anywhere ? Math.random() * W : dir > 0 ? -40 : W + 40;
    const r = Math.random();
    o.kind = o.d < 700 ? 'bird' : o.d < BROWNOUT ? (r < .5 ? 'bag' : 'bill') : (r < .6 ? 'bat' : 'bag');
    o.v = { bird: 30, bat: 40, bag: 10, bill: 12 }[o.kind] * (.8 + Math.random() * .5);
    o.vd = { bird: 0, bat: 0, bag: -4, bill: 5 }[o.kind];
  }
  function drifters(dt) {
    for (const o of drift) {
      o.x += o.dir * o.v * dt; o.d += o.vd * dt;
      let y = Y(o.d);
      if (o.x < -60 || o.x > W + 60 || y < -90 || y > H + 90) { respawn(o); y = Y(o.d); }
      if (o.d > FLOOR - 20) continue;
      const f = Math.floor(T * 6 + o.i) % 2;
      if (o.kind === 'bird') put(BIRD[f], o.x, y + Math.sin(T * 2 + o.ph) * 4, o.dir < 0);
      else if (o.kind === 'bat') put(BAT[f], o.x, y + Math.sin(T * 5 + o.ph) * 6, o.dir < 0);
      else if (o.kind === 'bag') put(BAG, o.x + Math.sin(T + o.ph) * 8, y);
      else put(BILL, o.x + Math.sin(T * 3 + o.ph) * 10, y, f);
    }
  }

  function draw() {
    const sy = Y(0);
    for (let y = 0; y < H; y += 4) {
      cx.fillStyle = rgb(y < sy ? ramp(SKY, y / sy) : ramp(DEEP, dAt(y)));
      cx.fillRect(0, y, W, 4);
    }
    if (sy > -10) sky(sy);
    const top = dAt(0), bot = dAt(H);
    for (let s = 0; s < 2; s++) {
      for (let i = Math.max(1, Math.floor((depth - H) / 240)); i * 240 < depth + H; i++) far(i, s);
      for (let i = Math.max(1, Math.floor((top - 200) / 150)); i * 150 < Math.min(bot + 200, FLOOR - 120); i++) near(i, s);
    }
    for (let i = 0; i < 36; i++) {
      const y = (((hash(i * 3 + 7) * H * 2 - depth * PPM * .35) % H) + H) % H;
      if (y > sy + 8) { cx.fillStyle = `rgba(${INK},${.08 + hash(i) * .14})`; cx.fillRect(Math.round(hash(i * 3 + 8) * W), Math.round(y), 2, 2); }
    }
    marks();
    if (lines) for (const [p, name] of STEPS) {
      const d = lines.from + p * 10, y = Math.round(Y(d)); if (y < 0 || y > H) continue;
      const col = depth >= d - 1 ? `rgba(${GOLD},.95)` : `rgba(${INK},.6)`;
      cx.fillStyle = col;
      for (let x = 12; x < W - 64; x += 12) cx.fillRect(x, y, 6, 2);
      text(`${name} ${p}`, W - 66, y - 11, 18, col, 'right');
    }
    drifters(still ? 0 : dtNow);
    const fy = Y(FLOOR) + 62;
    if (fy < H) {
      cx.fillStyle = '#1c1632';
      for (let x = 0; x < W; x += 4 * U) cx.fillRect(x, fy - (hash(x) * 4 | 0) * U, 4 * U, 6 * U);
      cx.fillStyle = '#120e22'; cx.fillRect(0, fy, W, H - fy);
    }
    for (const c of coins) put(COIN[[0, 1, 2, 1][Math.floor(c.t * 8 + c.ph) % 4]], c.x, Y(c.d));
    if (scene !== 'start') {
      const px = W * .2 + Math.sin(T * .7) * 8 + (sink ? Math.sin(T * 40) * 1.5 : 0), py = anchor * H + 12 + Math.sin(T * 2) * 3;
      if (tag) {
        const size = fit(tag, 24, W * .7), tw = Math.min(W * .7, cx.measureText(tag).width * size / 24) + 28, tx = (W - tw) / 2;
        cx.fillStyle = 'rgba(22,13,38,.9)'; notched(tx, py + 46, tw, 34);
        text(tag, W / 2, py + 63, size, `rgb(${GOLD})`, 'center');
      }
      put(PIG, px, py);
    }
    ruler(sy);
  }

  let dtNow = 0;
  function frame(now) {
    dtNow = Math.min(.05, (now - last) / 1000 || 0); last = now; T = now / 1000;
    anchor += (rest() - anchor) * Math.min(1, dtNow * 5);
    if (sink) {
      const k = Math.min(1, (now - sink.t0) / sink.dur), e = k < .5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2;
      depth = sink.from + (sink.to - sink.from) * e; onDepth(depth);
      if (k === 1) { const done = sink.done; sink = null; tag = null; done(); }
    }
    if (scene !== 'start' && now > coinAt && coins.length < 80) {
      coinAt = now + (sink ? 30 : 900);
      coins.push({ x: W * .2 + 18 + Math.random() * 14, d: dAt(anchor * H + 10), v: 22 + Math.random() * 30, ph: Math.random() * 6, t: 0 });
    }
    for (let i = coins.length; i--;) {
      const c = coins[i]; c.t += dtNow; c.d -= c.v * dtNow; c.x += Math.sin(c.t * 3 + c.ph) * 12 * dtNow;
      if (c.t > 5 || c.d < 4 || Y(c.d) < -20) coins.splice(i, 1);
    }
    draw();
    requestAnimationFrame(frame);
  }

  addEventListener('resize', resize);
  resize();
  document.fonts.load('20px VT323').then(() => still && draw());
  if (!still) requestAnimationFrame(frame);

  return {
    set onDepth(f) { onDepth = f; },
    scene(s) { scene = s; if (s === 'play' || s === 'end') lines = null; if (still) { anchor = rest(); draw(); } },
    jump(d) { depth = d; onDepth(d); if (still) draw(); },
    sink(from, to, label, done) {
      lines = { from }; scene = 'fb';
      if (still) { depth = to; anchor = ANCHOR.fb; onDepth(to); draw(); return done(); }
      tag = label.toUpperCase();
      sink = { from, to, done, t0: performance.now(), dur: 900 + (to - from) * 2.2 };
    },
  };
})();
