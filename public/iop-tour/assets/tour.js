// Wing C virtual tour. Reads the rooms from the list in index.html and builds
// a cutaway 3D model around them. The page works without this script; it only
// adds the model, the guided tour and the photo lightbox.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DObject, CSS2DRenderer } from "three/addons/renderers/CSS2DRenderer.js";
import { FLOOR_HEIGHT, FLOORS, HALLS, OBLATION, OUTLINE, STAIRS, WING_ANGLE_DEG } from "./building.js";

const params = new URLSearchParams(location.search);
const RECORD = params.has("record");
const DWELL_MS = (Number(params.get("dwell")) || 6) * 1000;
const FLY_MS = 1800;
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");

const SLAB = 0.3;
// Cutaway height: low enough for the camera to see into every room.
const WALL_H = 1.4;
const WALL_T = 0.14;
const DOOR_W = 1.2;

// The tour is its own page here, and a block inside a page on the IoP site;
// `root` is the element that carries its tokens and the `js` class either way.
const root = document.querySelector(".iop-tour") ?? document.documentElement;
const css = getComputedStyle(root);
const color = (name) => new THREE.Color(css.getPropertyValue(name).trim());
const COLORS = {
  lab: color("--type-lab"),
  office: color("--type-office"),
  facility: color("--type-facility"),
  venue: color("--type-venue"),
  hall: color("--type-hall"),
  wall: color("--model-wall"),
  slab: color("--model-slab"),
  block: color("--model-block"),
  glass: color("--model-glass"),
  ground: color("--model-ground"),
  sky: color("--model-sky"),
};

// --- Coordinates -----------------------------------------------------------
// Local metres from the outline centroid (x east, y north), the same
// projection the Room TBA viewer uses. World space: x east, y up, z south.
const M_PER_DEG = 111_320;
const lon0 = OUTLINE.reduce((s, p) => s + p[0], 0) / OUTLINE.length;
const lat0 = OUTLINE.reduce((s, p) => s + p[1], 0) / OUTLINE.length;
const lonScale = Math.cos((lat0 * Math.PI) / 180) * M_PER_DEG;
const toLocal = ([lon, lat]) => ({ x: (lon - lon0) * lonScale, y: (lat - lat0) * M_PER_DEG });

const theta = (WING_ANGLE_DEG * Math.PI) / 180;
const ALONG = { x: Math.cos(theta), y: Math.sin(theta) };
const ACROSS = { x: -Math.sin(theta), y: Math.cos(theta) };
const wing = (u, v) => ({ x: u * ALONG.x + v * ACROSS.x, y: u * ALONG.y + v * ACROSS.y });
const toWorld = (p, h = 0) => new THREE.Vector3(p.x, h, -p.y);
const floorBase = (f) => (f - 1) * FLOOR_HEIGHT;

// --- Rooms from the page ---------------------------------------------------
const pair = (s) => s.trim().split(/\s+/).map(Number).sort((a, b) => a - b);
const rooms = [...document.querySelectorAll(".room")].map((li) => {
  const u = pair(li.dataset.along);
  const v = pair(li.dataset.across);
  const floor = Number(li.dataset.floor);
  return {
    li,
    details: li.querySelector("details"),
    code: li.dataset.code,
    title: li.querySelector(".room-code").textContent,
    name: li.querySelector(".room-name").textContent,
    desc: li.querySelector(".room-desc")?.textContent ?? "",
    photos: [...li.querySelectorAll(".room-photo img")].map((img) => ({ src: img.getAttribute("src"), alt: img.alt })),
    type: li.dataset.type,
    floor,
    u,
    v,
    door: li.dataset.door ?? null,
    inTour: li.dataset.tour !== "no",
    center: toWorld(wing((u[0] + u[1]) / 2, (v[0] + v[1]) / 2), floorBase(floor) + SLAB),
    size: Math.max(u[1] - u[0], v[1] - v[0]),
  };
});

// --- Scene -----------------------------------------------------------------
const stage = document.querySelector(".stage");
const canvasHost = stage.querySelector(".stage-canvas");
const labelHost = stage.querySelector(".stage-labels");
const panel = document.querySelector(".panel");

const scene = new THREE.Scene();
scene.background = COLORS.sky;
// Fog fades the ground disc into the background instead of ending at a hard rim.
scene.fog = new THREE.Fog(COLORS.sky, 140, 460);
const camera = new THREE.PerspectiveCamera(40, 1, 0.5, 2000);
let renderer;
try {
  renderer = new THREE.WebGLRenderer({ antialias: true });
} catch (err) {
  // No WebGL: fall back to the plain room list (see html:not(.js) in tour.css).
  root.classList.remove("js");
  throw err;
}
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFShadowMap;
canvasHost.append(renderer.domElement);
const labelRenderer = new CSS2DRenderer({ element: labelHost });

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 6;
controls.maxDistance = 260;
controls.maxPolarAngle = (78 * Math.PI) / 180;

scene.add(new THREE.HemisphereLight(0xfff4ec, 0x2a1f26, 1.1));
const sun = new THREE.DirectionalLight(0xffffff, 2.2);
sun.position.set(-60, 120, -40);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -90, right: 90, top: 90, bottom: -90, near: 10, far: 400 });
sun.shadow.bias = -0.0005;
sun.shadow.radius = 3;
scene.add(sun);

const mat = (c, extra = {}) => new THREE.MeshStandardMaterial({ color: c, roughness: 0.85, ...extra });
const wallMat = mat(COLORS.wall);

// Finishes from the door photos: cream painted block, light floor tile, a tile
// ceiling with light panels. Drawn on canvases, so no image files to load. On the
// map the floor tile is tinted by room type; inside a room it is plain.
const BLOCK_M = [0.8, 0.4]; // canvas covers two blocks by two courses
const TILE_M = 1.2; // two 0.6 m tiles each way
function canvasTexture(w, h, draw) {
  const c = Object.assign(document.createElement("canvas"), { width: w, height: h });
  draw(c.getContext("2d"));
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 4;
  return t;
}
const blockTex = canvasTexture(256, 128, (g) => {
  g.fillStyle = "#f4eddd";
  g.fillRect(0, 0, 256, 128);
  g.fillStyle = "#cbc0a8";
  g.fillRect(0, 0, 256, 4);
  g.fillRect(0, 64, 256, 4);
  for (const [x, y] of [[0, 0], [128, 0], [64, 64], [192, 64]]) g.fillRect(x, y, 4, 64);
});
const tileTex = canvasTexture(128, 128, (g) => {
  g.fillStyle = "#d6d0c4";
  g.fillRect(0, 0, 128, 128);
  g.fillStyle = "#b5ada0";
  for (const p of [0, 64]) {
    g.fillRect(p, 0, 2, 128);
    g.fillRect(0, p, 128, 2);
  }
});
const ceilingTex = canvasTexture(256, 128, (g) => {
  g.fillStyle = "#e9e5dc";
  g.fillRect(0, 0, 256, 128);
  g.fillStyle = "#fffdf6";
  g.fillRect(70, 6, 116, 52); // one light panel per 2.4 x 1.2 m
  g.fillStyle = "#c9c3b7";
  for (let p = 0; p < 256; p += 64) g.fillRect(p, 0, 2, 128);
  g.fillRect(0, 0, 256, 2);
  g.fillRect(0, 64, 256, 2);
});
const repeated = (tex, x, y) => {
  const t = tex.clone();
  t.repeat.set(x, y);
  t.needsUpdate = true;
  return t;
};


const ground = new THREE.Mesh(new THREE.CircleGeometry(600, 64), mat(COLORS.ground, { roughness: 1 }));
ground.rotation.x = -Math.PI / 2;
ground.position.y = -0.02;
ground.receiveShadow = true;
ground.name = "ground";
scene.add(ground);

/** A box lying in the wing frame: centre (u, v), size du x dv, height h, bottom at y. */
function wingBox(u, v, du, dv, h, y, material, name = "") {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(du, h, dv), material);
  mesh.name = name;
  mesh.position.copy(toWorld(wing(u, v), y + h / 2));
  mesh.rotation.y = theta;
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

function makeLabel(text, className, parent, position) {
  const el = document.createElement("div");
  el.className = `label ${className}`;
  el.textContent = text;
  const obj = new CSS2DObject(el);
  obj.position.copy(position);
  parent.add(obj);
  return { el, obj, w: 0, h: 0 };
}
const labels = [];

// Building shell: one group per storey, shown as a solid block with a window
// band (below the focused floor), an open slab with a low parapet (the focused
// floor) or a ghost outline (above).
const outline = OUTLINE.slice(0, -1).map(toLocal);
const shape = new THREE.Shape(outline.map((l) => new THREE.Vector2(l.x, l.y)));
const extrude = (depth) => new THREE.ExtrudeGeometry(shape, { depth, bevelEnabled: false }).rotateX(-Math.PI / 2);
const blockGeom = extrude(FLOOR_HEIGHT - 0.06);
const slabGeom = extrude(SLAB);
const edgeGeom = new THREE.EdgesGeometry(blockGeom);

// The outline runs clockwise, so the outward normal of edge a->b is (-dy, dx).
const facades = outline.map((a, i) => {
  const b = outline[(i + 1) % outline.length];
  const len = Math.hypot(b.x - a.x, b.y - a.y);
  return { mid: { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }, len, angle: Math.atan2(b.y - a.y, b.x - a.x), out: { x: -(b.y - a.y) / len, y: (b.x - a.x) / len } };
});
/** A box along one facade edge, pushed `offset` metres outward. */
function facadeBox(f, length, thickness, h, y, offset, material, name) {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(length, h, thickness), material);
  mesh.name = name;
  mesh.position.copy(toWorld({ x: f.mid.x + f.out.x * offset, y: f.mid.y + f.out.y * offset }, y + h / 2));
  mesh.rotation.y = f.angle;
  mesh.castShadow = mesh.receiveShadow = true;
  return mesh;
}
const PARAPET_H = 1.0;
const glassMat = mat(COLORS.glass, { roughness: 0.3 });

const storeys = [];
for (let f = 1; f <= FLOORS; f++) {
  const y = floorBase(f);
  const block = new THREE.Group();
  const shell = new THREE.Mesh(blockGeom, mat(COLORS.block));
  shell.name = "block";
  shell.position.y = y;
  shell.castShadow = shell.receiveShadow = true;
  block.add(shell);
  // ponytail: one window band per facade edge, not real window bays.
  for (const e of facades) if (e.len > 2) block.add(facadeBox(e, e.len - 1, 0.06, 1.3, y + 1.1, 0.02, glassMat, "glass"));
  // The rest of the building's floor plate, tiled like the rooms but in plain concrete.
  const slab = new THREE.Mesh(slabGeom, mat(COLORS.slab, { map: repeated(tileTex, 1 / TILE_M, 1 / TILE_M) }));
  slab.position.y = y;
  slab.name = "slab";
  slab.receiveShadow = true;
  const ghost = new THREE.LineSegments(edgeGeom, new THREE.LineBasicMaterial({ color: COLORS.wall, transparent: true, opacity: 0.14, fog: false }));
  ghost.position.y = y;
  const contents = new THREE.Group();
  for (const e of facades) contents.add(facadeBox(e, e.len + WALL_T, WALL_T, PARAPET_H, y + SLAB, -WALL_T / 2, wallMat, "parapet"));
  scene.add(block, slab, ghost, contents);
  storeys.push({ floor: f, block, slab, ghost, contents });
}
const storey = (f) => storeys[f - 1];

for (const hall of HALLS) {
  const [u0, u1] = hall.along;
  const [v0, v1] = hall.across;
  const y = floorBase(hall.floor) + SLAB;
  storey(hall.floor).contents.add(wingBox((u0 + u1) / 2, (v0 + v1) / 2, u1 - u0, v1 - v0, 0.04, y, mat(COLORS.hall), "floor-hall"));
  if (hall.label) {
    labels.push({ ...makeLabel(hall.label, "is-hall", storey(hall.floor).contents, toWorld(wing((u0 + u1) / 2, (v0 + v1) / 2 + 5), y + 0.5)), rank: 3 });
  }
}

for (const s of STAIRS) {
  const [u0, u1] = s.along;
  const [v0, v1] = s.across;
  const y = floorBase(s.floor) + SLAB;
  // Steps climb toward the front of the wing (the Oblation side).
  const steps = 8;
  const run = (v1 - v0) / steps;
  for (let i = 0; i < steps; i++) {
    const h = ((i + 1) / steps) * WALL_H;
    storey(s.floor).contents.add(wingBox((u0 + u1) / 2, v0 + run * (i + 0.5), u1 - u0, run, h, y, mat(COLORS.wall), "stairs"));
  }
  labels.push({ ...makeLabel("Stairs", "is-hall", storey(s.floor).contents, toWorld(wing((u0 + u1) / 2, (v0 + v1) / 2), y + WALL_H + 0.6)), rank: 3 });
}

/** Wall segments around a room, inset half a wall so neighbours never overlap. */
function roomWalls(room, y) {
  const t = WALL_T;
  const [a0, a1] = [room.u[0] + t / 2, room.u[1] - t / 2];
  const [b0, b1] = [room.v[0] + t / 2, room.v[1] - t / 2];
  const len = a1 - a0;
  const segments = [
    [a0, (b0 + b1) / 2, t, b1 - b0 + t],
    [a1, (b0 + b1) / 2, t, b1 - b0 + t],
  ];
  for (const [side, v] of [["-", b0], ["+", b1]]) {
    if (room.door === side) {
      // A gap in the middle of the corridor side reads as the door.
      const piece = (len - DOOR_W) / 2;
      segments.push([a0 + piece / 2, v, piece, t], [a1 - piece / 2, v, piece, t]);
    } else {
      segments.push([(a0 + a1) / 2, v, len + t, t]);
    }
  }
  const block = (len) => mat(0xffffff, { map: repeated(blockTex, len / BLOCK_M[0], WALL_H / BLOCK_M[1]) });
  return segments.map(([u, v, du, dv]) => wingBox(u, v, du, dv, WALL_H, y, block(Math.max(du, dv)), "wall"));
}

// --- Furniture -------------------------------------------------------------
// Simple boxes so every room reads as what it is, from above and from inside.
// ponytail: generic layouts per kind, not measured. Swap in real counts when
// the Institute sends them.
const propMats = new Map();
const propMat = (hex) => {
  if (!propMats.has(hex)) propMats.set(hex, mat(new THREE.Color(hex)));
  return propMats.get(hex);
};
const WOOD = "#8b6a4f", DARK = "#2b2b30", BOARD = "#f2f2f0", METAL = "#9aa0a6", SEAT = "#6b5246";
// Boards hang on walls the sun doesn't reach; a little glow keeps them white.
propMats.set(BOARD, mat(new THREE.Color(BOARD), { emissive: new THREE.Color(BOARD), emissiveIntensity: 0.45, roughness: 0.3 }));
// Faint grain on wood and a little sheen on metal, so tables and shelves read up close.
const grainTex = canvasTexture(256, 64, (g) => {
  g.fillStyle = "#ffffff";
  g.fillRect(0, 0, 256, 64);
  g.strokeStyle = "rgba(60, 35, 15, 0.2)";
  for (let y = 3; y < 64; y += 5) {
    g.lineWidth = 1 + (y % 3);
    g.beginPath();
    g.moveTo(0, y);
    g.bezierCurveTo(80, y - 2, 170, y + 3, 256, y);
    g.stroke();
  }
});
propMats.set(WOOD, mat(new THREE.Color(WOOD), { map: grainTex, roughness: 0.6 }));
propMats.set(METAL, mat(new THREE.Color(METAL), { roughness: 0.4 }));
const PROP_NAMES = { [WOOD]: "wood", [DARK]: "dark", [BOARD]: "board", [METAL]: "metal", [SEAT]: "seat" };

const DEFAULT_KIND = { lab: "benches", office: "desks", facility: "shelves", venue: "conference" };

function furnish(room, y) {
  const [u0, u1] = room.u;
  const [v0, v1] = room.v;
  const du = u1 - u0, dv = v1 - v0;
  const um = (u0 + u1) / 2, vm = (v0 + v1) / 2;
  // The wall facing the corridor has the door; furniture leans on the other one.
  const inner = room.door === "+" ? v0 + 0.05 : v1 - 0.05;
  const inward = room.door === "+" ? 1 : -1;
  const out = [];
  const put = (u, v, a, b, h, hex, lift = 0) => out.push(wingBox(u, v, a, b, h, y + lift, propMat(hex), `prop-${PROP_NAMES[hex]}`));
  // A top on two end panels, so tables read as tables from inside the room.
  const table = (u, v, a, b, h) => {
    put(u, v, a, b, 0.05, WOOD, h - 0.05);
    for (const s of [-1, 1]) {
      if (a >= b) put(u + s * (a / 2 - 0.08), v, 0.05, b - 0.12, h - 0.05, METAL);
      else put(u, v + s * (b / 2 - 0.08), a - 0.12, 0.05, h - 0.05, METAL);
    }
  };
  const stools = (u, v, len, side) => {
    for (let k = u - len / 2 + 0.5; k < u + len / 2; k += 1.1) put(k, v + side, 0.34, 0.34, 0.6, SEAT);
  };
  const board = () => put(u0 + 0.12, vm, 0.06, Math.min(3, dv - 1.5), 1.2, BOARD, 0.9);
  const rows = (first, last, step, depth, h, hex, margin = 0.8) => {
    for (let u = first; u <= last; u += step) put(u, vm, depth, dv - 2 * margin, h, hex);
  };

  switch (room.li.dataset.furnish ?? DEFAULT_KIND[room.type]) {
    case "benches":
      board();
      for (const f of [0.33, 0.67]) {
        table(um + 0.4, v0 + dv * f, du - 2.6, 0.9, 0.9);
        for (const side of [-0.85, 0.85]) stools(um + 0.4, v0 + dv * f, du - 2.6, side);
      }
      break;
    case "computers":
      board();
      for (const f of [0.33, 0.67]) {
        table(um + 0.4, v0 + dv * f, du - 2.6, 0.8, 0.75);
        for (let u = u0 + 1.8; u < u1 - 0.8; u += 0.9) put(u, v0 + dv * f, 0.5, 0.06, 0.35, DARK, 0.8);
      }
      break;
    case "optics": {
      put(um, vm, Math.min(3, du - 1.4), 1.5, 0.85, DARK);
      table(um, inner + inward * 0.4, du - 1, 0.7, 0.9);
      const beam = mat(COLORS.lab, { emissive: COLORS.lab, emissiveIntensity: 0.6 });
      out.push(wingBox(um, vm, Math.min(3, du - 1.4) - 0.3, 0.02, 0.02, y + 1.0, beam, "beam"));
      room.fx = (t) => (beam.emissiveIntensity = 0.6 + 0.6 * Math.sin(t / 500));
      break;
    }
    case "chemlab":
      put(um, inner + inward * 0.4, du - 1, 0.75, 0.9, WOOD);
      put(u1 - 0.8, vm, 1.2, 0.9, 2.2, METAL);
      break;
    case "desks":
      for (let u = u0 + 1; u + 0.6 < u1 - 0.3; u += 1.6) table(u, inner + inward * 0.4, 1.2, 0.65, 0.75);
      put(u1 - 0.35, vm, 0.4, Math.min(1.2, dv - 2), 1.9, WOOD);
      break;
    case "director":
      table(um, inner + inward * 1.2, 1.6, 0.8, 0.75);
      table(um, vm + inward * 1.2, 1.2, 1.2, 0.72);
      put(u1 - 0.35, vm, 0.4, 1.2, 1.9, WOOD);
      break;
    case "ta":
      for (let u = u0 + 1; u + 0.6 < u1 - 1.5; u += 1.6) table(u, inner + inward * 0.4, 1.2, 0.65, 0.75);
      put(u1 - 0.4, vm, 0.6, dv - 2, 0.9, DARK);
      break;
    case "conference": {
      const len = Math.max(1.6, du - 2.4);
      table(um, vm, len, 1.2, 0.75);
      for (let u = um - len / 2 + 0.4; u < um + len / 2; u += 0.8) {
        for (const s of [-1, 1]) put(u, vm + s * 1.0, 0.45, 0.45, 0.45, SEAT);
      }
      break;
    }
    case "lecture":
      board();
      table(u0 + 1.4, vm, 0.7, 1.6, 0.75);
      rows(u0 + 2.8, u1 - 0.8, 1.0, 0.45, 0.45, SEAT);
      break;
    case "hall":
      board();
      table(u0 + 1.8, vm, 0.8, 2.4, 0.75);
      // Tiered: each row a little higher than the one in front.
      for (let u = u0 + 3.5, i = 0; u <= u1 - 0.8; u += 1.0, i++) put(u, vm, 0.5, dv - 2.4, 0.45, SEAT, Math.min(1.2, i * 0.08));
      break;
    case "shelves":
      // Shelving down both side walls with an aisle from the door.
      for (const u of [u0 + 0.35, u1 - 0.35]) put(u, vm, 0.45, dv - 1.6, 2, WOOD);
      put(um, vm - inward * 0.6, 0.45, dv - 3.4, 1.8, METAL);
      break;
    case "servers": {
      const leds = ["#4dff88", "#4dff88", "#ffb347"].map((c) => mat(new THREE.Color(c), { emissive: new THREE.Color(c), emissiveIntensity: 1 }));
      for (let i = 0; i < 3; i++) {
        const u = um - 0.9 + i * 0.9;
        put(u, vm - inward * 0.6, 0.6, 1.0, 2.0, DARK);
        for (let k = 0; k < 6; k++) {
          out.push(wingBox(u - 0.15 + (k % 2) * 0.3, vm - inward * 0.09, 0.04, 0.02, 0.03, y + 0.6 + k * 0.22, leds[(i + k) % 3], "led"));
        }
      }
      room.fx = (t) => leds.forEach((m, i) => (m.emissiveIntensity = Math.sin(t / 380 + i * 2.1) > 0.2 ? 1.4 : 0.15));
      break;
    }
    case "toilets":
      put(um, vm, 0.14, dv - 0.2, WALL_H, BOARD);
      for (const u of [u0 + 1, um + 1]) put(u, inner + inward * 0.8, 0.05, 1.5, 1.6, METAL);
      break;
  }
  return out;
}

const pickables = [];
for (const room of rooms) {
  const y = floorBase(room.floor) + SLAB;
  const [u0, u1] = room.u;
  const [v0, v1] = room.v;
  room.baseY = y;
  room.tileMat = mat(COLORS[room.type] ?? COLORS.hall, { emissive: 0x000000, map: repeated(tileTex, (u1 - u0) / TILE_M, (v1 - v0) / TILE_M) });
  const tile = wingBox((u0 + u1) / 2, (v0 + v1) / 2, u1 - u0, v1 - v0, 0.06, y, room.tileMat, `floor-${room.type}`);
  tile.userData.room = room;
  room.tile = tile;
  pickables.push(tile);
  const group = storey(room.floor).contents;
  room.walls = roomWalls(room, y);
  group.add(tile, ...room.walls, ...furnish(room, y));
  room.label = { ...makeLabel(room.code, "", group, room.center.clone().setY(y + WALL_H + 0.8)), room, rank: 2 };
  labels.push(room.label);
}

// Oblation Park: a label and a dashed line from the Wing C entrance, so the
// model is oriented the way people know the building. The statue itself is
// left out; a stand-in shape would not do it justice.
{
  const p = toLocal(OBLATION);
  labels.push({ ...makeLabel("Oblation Park", "is-landmark", scene, toWorld(p, 1)), rank: 1 });
  const entrance = toWorld(wing(3, 36), 0.05);
  const path = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([entrance, toWorld(p, 0.05)]),
    new THREE.LineDashedMaterial({ color: COLORS.wall, dashSize: 1.5, gapSize: 1.2, transparent: true, opacity: 0.5 }),
  );
  path.computeLineDistances();
  scene.add(path);
}

// --- Floors ----------------------------------------------------------------
let focusFloor = 2;
const floorButtons = [...document.querySelectorAll("[data-floor-button]")];

function setFloor(f) {
  focusFloor = f;
  for (const s of storeys) {
    s.block.visible = s.floor < f;
    s.slab.visible = s.contents.visible = s.floor === f;
    s.ghost.visible = s.floor > f;
  }
  for (const b of floorButtons) b.setAttribute("aria-pressed", String(Number(b.dataset.floorButton) === f));
}

// --- Camera ----------------------------------------------------------------
// Every shot looks from behind the wing toward Oblation Park, slightly angled
// along the corridor, so a recorded tour reads as one continuous walk.
const viewDir = toWorld({ x: -ACROSS.x - 0.25 * ALONG.x, y: -ACROSS.y - 0.25 * ALONG.y }).normalize();
const PITCH = (50 * Math.PI) / 180;

function shot(target, distance) {
  const pos = target.clone()
    .addScaledVector(viewDir, Math.sin(PITCH) * distance)
    .add(new THREE.Vector3(0, Math.cos(PITCH) * distance, 0));
  return { pos, target };
}

/** Distance at which a span of `widthM` x `depthM` metres fills the stage. */
function fitDistance(widthM, depthM) {
  const halfV = (camera.fov * Math.PI) / 360;
  const halfH = Math.atan(Math.tan(halfV) * camera.aspect);
  return Math.max(widthM / 2 / Math.tan(halfH), depthM / 2 / Math.tan(halfV)) * 1.1;
}
// Frames everything mapped on a floor: its rooms and halls.
function overviewShot(floor = focusFloor) {
  const boxes = [...rooms.filter((r) => r.floor === floor), ...HALLS.filter((h) => h.floor === floor).map((h) => ({ u: h.along, v: h.across }))];
  const [u0, u1] = [Math.min(...boxes.map((b) => b.u[0])), Math.max(...boxes.map((b) => b.u[1]))];
  const [v0, v1] = [Math.min(...boxes.map((b) => b.v[0])), Math.max(...boxes.map((b) => b.v[1]))];
  // Seen at 50 degrees the depth shrinks to about cos(50) = 0.64; the extra
  // metres leave room for the toolbar above and the caption below.
  return shot(toWorld(wing((u0 + u1) / 2, (v0 + v1) / 2), floorBase(floor) + SLAB), fitDistance(u1 - u0 + 4, (v1 - v0) * 0.64 + 22));
}
const roomShot = (room) => shot(room.center.clone().setY(room.center.y + 1), THREE.MathUtils.clamp(room.size * 2.2 + 8, 16, 42));

const BASE_FOV = camera.fov;
const LIMITS = { minDistance: 6, maxDistance: 260 };
let inside = null;
let tween = null;

/**
 * Move the camera. Distance limits are lifted for the flight (stepping into a
 * room passes well under the usual minimum) and come back when it lands.
 */
function flyTo({ pos, target, fov = BASE_FOV }, ms = FLY_MS, onDone) {
  Object.assign(controls, { minDistance: 0.05, maxDistance: 1000 });
  const finish = () => {
    if (!inside) Object.assign(controls, LIMITS);
    onDone?.();
  };
  if (ms === 0 || reducedMotion.matches) {
    tween = null;
    camera.position.copy(pos);
    controls.target.copy(target);
    camera.fov = fov;
    camera.updateProjectionMatrix();
    finish();
    return;
  }
  tween = {
    fromPos: camera.position.clone(), fromTarget: controls.target.clone(), fromFov: camera.fov,
    pos, target, fov, finish, start: performance.now(), ms,
  };
}
function stepTween(now) {
  if (!tween) return;
  const t = Math.min(1, (now - tween.start) / tween.ms);
  const e = t < 0.5 ? 4 * t ** 3 : 1 - (-2 * t + 2) ** 3 / 2;
  camera.position.lerpVectors(tween.fromPos, tween.pos, e);
  controls.target.lerpVectors(tween.fromTarget, tween.target, e);
  camera.fov = tween.fromFov + (tween.fov - tween.fromFov) * e;
  camera.updateProjectionMatrix();
  if (t === 1) endTween();
}
function endTween() {
  const finish = tween?.finish;
  tween = null;
  finish?.();
}

// --- Selection -------------------------------------------------------------
let active = null;

function select(room, { fly = true, scroll = true } = {}) {
  if (inside && inside !== room) leaveRoom({ fly: false });
  if (active && active !== room) {
    active.tileMat.emissive.setHex(0x000000);
    active.label.el.classList.remove("is-active");
    active.label.el.textContent = active.code;
    active.label.w = 0;
    if (active.details.open) active.details.open = false;
  }
  active = room;
  syncButtons();
  if (!room) return;
  if (room.floor !== focusFloor) setFloor(room.floor);
  room.tileMat.emissive.copy(room.tileMat.color).multiplyScalar(0.45);
  room.label.el.classList.add("is-active");
  room.label.el.textContent = `${room.code} ${room.name}`;
  room.label.w = 0;
  if (!room.details.open) room.details.open = true;
  // Only when the list scrolls on its own (side panel). On phones the list sits
  // under the model, and scrolling the page would push the model off screen.
  // Scroll the list itself, never the page around it (the tour can sit inside a page).
  if (scroll && !RECORD && panel.scrollHeight > panel.clientHeight) {
    const item = room.li.getBoundingClientRect();
    const box = panel.getBoundingClientRect();
    const delta = item.top < box.top ? item.top - box.top : Math.max(0, item.bottom - box.bottom);
    if (delta) panel.scrollBy({ top: delta, behavior: reducedMotion.matches ? "auto" : "smooth" });
  }
  if (fly) flyTo(roomShot(room));
}

// --- Inside a room ---------------------------------------------------------
// The walls rise to full height, the camera stands at eye level by the door,
// and dragging turns your head: the orbit target sits 0.3 m in front of the eye.
const FULL_H = 2.9;
const EYE_H = 1.6;
const insideButton = document.querySelector('[data-action="inside"]');
const outsideButton = document.querySelector('[data-action="outside"]');
// One ceiling, moved into whichever room you step into. Unlit, so it reads as
// a lit ceiling without a lamp; it casts no shadow, so daylight still gets in.
const ceiling = new THREE.Mesh(new THREE.BoxGeometry(1, 0.06, 1), new THREE.MeshBasicMaterial({ map: ceilingTex.clone() }));
ceiling.name = "ceiling";
ceiling.visible = false;
scene.add(ceiling);

/** Inside a room the floor drops its map tint. Built on first visit. */
function dress(room, real) {
  room.realFloor ??= mat(0xffffff, { map: room.tileMat.map, roughness: 0.55 });
  room.tile.material = real ? room.realFloor : room.tileMat;
}

function syncButtons() {
  const focused = document.activeElement;
  insideButton.hidden = !active || Boolean(inside);
  outsideButton.hidden = !inside;
  // The two swap places, so keyboard focus moves across instead of dropping to the page.
  if (focused === insideButton && insideButton.hidden && !outsideButton.hidden) outsideButton.focus();
  if (focused === outsideButton && outsideButton.hidden && !insideButton.hidden) insideButton.focus();
}

function setWalls(room, h) {
  for (const w of room.walls) {
    w.scale.y = h / WALL_H;
    w.position.y = room.baseY + h / 2;
    w.material.map.repeat.y = h / BLOCK_M[1]; // keep the courses 0.2 m tall
  }
}

function enterRoom(room) {
  pauseTour();
  if (active !== room) select(room, { fly: false });
  inside = room;
  setWalls(room, FULL_H);
  dress(room, true);
  room.tileMat.emissive.setHex(0x000000);
  for (const s of storeys) s.ghost.visible = false;
  stage.classList.add("is-inside");
  syncButtons();
  const um = (room.u[0] + room.u[1]) / 2;
  const vm = (room.v[0] + room.v[1]) / 2;
  ceiling.position.copy(toWorld(wing(um, vm), room.baseY + FULL_H + 0.03));
  ceiling.rotation.y = theta;
  ceiling.scale.set(room.u[1] - room.u[0], 1, room.v[1] - room.v[0]);
  ceiling.material.map.repeat.set((room.u[1] - room.u[0]) / 2.4, (room.v[1] - room.v[0]) / 1.2);
  ceiling.visible = true;
  // Stand just inside the door and look across toward the west wall, where the
  // board hangs, a little downward. Rooms without a door: from the west end, facing in.
  const eye = room.door
    ? wing(um, room.door === "+" ? room.v[1] - 0.6 : room.v[0] + 0.6)
    : wing(room.u[0] + 0.8, vm);
  const look = room.door ? wing(room.u[0] + 0.3 * (room.u[1] - room.u[0]), vm) : wing(um, vm);
  const pos = toWorld(eye, room.baseY + EYE_H);
  const dir = toWorld(look, room.baseY + EYE_H - 0.35).sub(pos).normalize();
  flyTo({ pos, target: pos.clone().addScaledVector(dir, 0.3), fov: 70 }, FLY_MS, () => {
    if (inside !== room) return;
    Object.assign(controls, {
      minDistance: 0.3, maxDistance: 0.3, enablePan: false, enableZoom: false,
      rotateSpeed: -0.4, minPolarAngle: 0.9, maxPolarAngle: 2.2,
    });
  });
}

function leaveRoom({ fly = true } = {}) {
  if (!inside) return;
  const room = inside;
  inside = null;
  setWalls(room, WALL_H);
  dress(room, false);
  ceiling.visible = false;
  if (active === room) room.tileMat.emissive.copy(room.tileMat.color).multiplyScalar(0.45);
  Object.assign(controls, {
    enablePan: true, enableZoom: true, rotateSpeed: 1, minPolarAngle: 0, maxPolarAngle: (78 * Math.PI) / 180,
  });
  setFloor(focusFloor);
  stage.classList.remove("is-inside");
  syncButtons();
  if (fly) flyTo(roomShot(room));
}

for (const room of rooms) {
  room.details.addEventListener("toggle", () => {
    if (room.details.open && active !== room) {
      visit(room);
    } else if (!room.details.open && active === room) {
      select(null);
    }
  });
}

// --- Tour ------------------------------------------------------------------
const intro = {
  title: "Institute of Physics, Wing C",
  desc: "The Institute of Physics rooms on the 2nd floor of Wing C, plus two lecture rooms on the ground floor. The main entrance faces Oblation Park.",
  photos: [{ src: "assets/img/fo-santos-hall.webp", alt: "Entrance of Francisco O. Santos Hall with the UPLB building sign" }],
};
const outro = {
  title: "Institute of Physics",
  desc: "Physical Sciences Building, Wing C, University of the Philippines Los Baños.",
  photos: [{ src: "assets/img/fo-santos-hall.webp", alt: "Entrance of Francisco O. Santos Hall with the UPLB building sign" }],
};
const tourRooms = rooms.filter((r) => r.inTour);
const stops = [intro, ...tourRooms, outro];
let stopIndex = -1;
let playing = false;
let timer = 0;

const caption = stage.querySelector(".caption");
const captionMeta = caption.querySelector(".caption-meta");
const captionTitle = caption.querySelector(".caption-title");
const captionDesc = caption.querySelector(".caption-desc");
const captionPhoto = caption.querySelector(".caption-photo");
const playButton = document.querySelector('[data-action="play"]');
const floorLabel = (f) => (f === 1 ? "Ground floor" : "2nd floor");

function setCaption(stop) {
  caption.hidden = false;
  const photo = stop.photos?.[0];
  captionPhoto.hidden = !photo;
  if (photo) {
    captionPhoto.src = photo.src;
    captionPhoto.alt = photo.alt;
  }
  if (!stop.li) {
    captionTitle.textContent = stop.title;
    captionMeta.hidden = true;
    captionDesc.textContent = stop.desc;
    return;
  }
  const n = tourRooms.indexOf(stop) + 1;
  const where = `${stop.title}, ${floorLabel(stop.floor)}`;
  captionTitle.textContent = stop.name;
  captionMeta.hidden = false;
  captionMeta.textContent = n > 0 ? `${where}. Stop ${n} of ${tourRooms.length}.` : `${where}.`;
  captionDesc.textContent = stop.desc;
}

/** A room picked by hand: show it, and let Next carry on from there. */
function visit(room) {
  pauseTour();
  select(room);
  setCaption(room);
  if (stops.includes(room)) stopIndex = stops.indexOf(room);
}

function showStop(i) {
  stopIndex = (i + stops.length) % stops.length;
  const stop = stops[stopIndex];
  setCaption(stop);
  if (stop.li) {
    select(stop, { scroll: !playing });
  } else {
    select(null);
    setFloor(2);
    flyTo(overviewShot(2));
  }
  schedule();
}

function schedule() {
  clearTimeout(timer);
  if (!playing) return;
  if (stopIndex === stops.length - 1) {
    pauseTour();
    return;
  }
  timer = setTimeout(() => showStop(stopIndex + 1), (reducedMotion.matches ? 0 : FLY_MS) + DWELL_MS);
}

function playTour() {
  playing = true;
  playButton.textContent = "Pause tour";
  showStop(stopIndex >= 0 && stopIndex < stops.length - 1 ? stopIndex : 0);
}
function pauseTour() {
  playing = false;
  clearTimeout(timer);
  playButton.textContent = stopIndex >= 0 ? "Resume tour" : "Play tour";
}

document.querySelector(".stage-controls").addEventListener("click", (e) => {
  const button = e.target.closest("button");
  if (!button) return;
  const action = button.dataset.action;
  if (action === "play") (playing ? pauseTour : playTour)();
  if (action === "next") showStop(stopIndex + 1);
  if (action === "prev") showStop(Math.max(0, stopIndex - 1));
  if (action === "inside" && active) enterRoom(active);
  if (action === "outside") leaveRoom();
  if (action === "overview") {
    pauseTour();
    caption.hidden = true;
    select(null);
    flyTo(overviewShot());
  }
  if (button.dataset.floorButton) {
    pauseTour();
    caption.hidden = true;
    select(null);
    setFloor(Number(button.dataset.floorButton));
    flyTo(overviewShot());
  }
});

addEventListener("keydown", (e) => {
  // Escape means nothing to a button, so it leaves the room even from "Back to map".
  // Inside a host page, keys outside the tour keep scrolling the page.
  if (!root.contains(e.target)) return;
  if (e.key === "Escape" && inside && !e.target.closest("dialog")) return leaveRoom();
  if (e.target.closest("button, summary, a, input, dialog")) return;
  if (e.key === " ") {
    e.preventDefault();
    (playing ? pauseTour : playTour)();
  } else if (e.key === "ArrowRight") showStop(stopIndex + 1);
  else if (e.key === "ArrowLeft") showStop(Math.max(0, stopIndex - 1));
});

// Taking the camera pauses the tour, so nobody fights the autopilot.
controls.addEventListener("start", () => {
  endTween();
  if (playing) pauseTour();
});

// --- Picking ---------------------------------------------------------------
const raycaster = new THREE.Raycaster();
const ndc = new THREE.Vector2();
let downAt = null;
function pick(e) {
  const r = renderer.domElement.getBoundingClientRect();
  ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  raycaster.setFromCamera(ndc, camera);
  const hit = raycaster.intersectObjects(pickables.filter((m) => m.userData.room.floor === focusFloor), false)[0];
  return hit?.object.userData.room ?? null;
}
renderer.domElement.addEventListener("pointerdown", (e) => (downAt = [e.clientX, e.clientY]));
renderer.domElement.addEventListener("pointerup", (e) => {
  // A drag that orbits the camera is not a click.
  if (inside || !downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 6) return;
  const room = pick(e);
  if (room) visit(room);
});
renderer.domElement.addEventListener("dblclick", (e) => {
  const room = inside ? null : pick(e);
  if (room) enterRoom(room);
});
renderer.domElement.addEventListener("pointermove", (e) => {
  if (e.buttons || inside) return;
  renderer.domElement.style.cursor = pick(e) ? "pointer" : "";
});

// --- Lightbox --------------------------------------------------------------
const lightbox = document.querySelector(".lightbox");
function openPhoto(src, alt) {
  const img = lightbox.querySelector(".lightbox-img");
  img.src = src;
  img.alt = alt;
  lightbox.querySelector(".lightbox-caption").textContent = alt;
  lightbox.showModal();
}
panel.addEventListener("click", (e) => {
  const link = e.target.closest(".room-photo");
  if (!link) return;
  e.preventDefault();
  const img = link.querySelector("img");
  openPhoto(img.getAttribute("src"), img.alt);
});
captionPhoto.addEventListener("click", () => openPhoto(captionPhoto.getAttribute("src"), captionPhoto.alt));

// --- Labels ----------------------------------------------------------------
// Greedy declutter: nearest and most important labels keep their spot.
const projected = new THREE.Vector3();
function declutter(width, height) {
  const items = [];
  for (const l of labels) {
    if (l.el.style.display === "none") continue;
    if (!l.w) {
      l.w = l.el.offsetWidth;
      l.h = l.el.offsetHeight;
    }
    projected.setFromMatrixPosition(l.obj.matrixWorld);
    const depth = projected.distanceTo(camera.position);
    projected.project(camera);
    const rank = l.room && l.room === active ? 0 : l.rank;
    items.push({ l, rank, depth, x: (projected.x * 0.5 + 0.5) * width, y: (-projected.y * 0.5 + 0.5) * height });
  }
  items.sort((a, b) => a.rank - b.rank || a.depth - b.depth);
  // Buttons, the compass and the caption count as taken, so no label hides under them.
  const origin = stage.getBoundingClientRect();
  const kept = [...stage.querySelectorAll(".stage-controls .btn, .compass, .caption")]
    .filter((el) => el.offsetParent)
    .map((el) => {
      const r = el.getBoundingClientRect();
      return { l: r.left - origin.left, r: r.right - origin.left, t: r.top - origin.top, b: r.bottom - origin.top };
    });
  for (const { l, x, y } of items) {
    const box = { l: x - l.w / 2 - 3, r: x + l.w / 2 + 3, t: y - l.h / 2 - 2, b: y + l.h / 2 + 2 };
    if (kept.some((k) => box.l < k.r && k.l < box.r && box.t < k.b && k.t < box.b)) {
      l.el.style.display = "none";
    } else {
      kept.push(box);
    }
  }
}

// --- Loop ------------------------------------------------------------------
const compass = document.querySelector(".compass");
function resize() {
  const w = stage.clientWidth;
  const h = stage.clientHeight;
  renderer.setSize(w, h);
  labelRenderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(stage);

renderer.setAnimationLoop((now) => {
  stepTween(now);
  if (active?.fx && !reducedMotion.matches) active.fx(now);
  controls.update();
  compass.style.setProperty("--bearing", `${(controls.getAzimuthalAngle() * 180) / Math.PI}deg`);
  renderer.render(scene, camera);
  labelRenderer.render(scene, camera);
  declutter(stage.clientWidth, stage.clientHeight);
});

// --- Start -----------------------------------------------------------------
// tools/export-glb.mjs reads the scene through this to build the Blender render.
if (params.has("export")) window.tourExport = { scene, rooms, setWalls, FULL_H, storeys };

document.querySelector(".stage-controls").hidden = false;
resize();
setFloor(2);
flyTo(overviewShot(2), 0);

if (RECORD) {
  document.body.classList.add("record");
  let idle = 0;
  addEventListener("pointermove", () => {
    document.body.classList.remove("idle");
    clearTimeout(idle);
    idle = setTimeout(() => document.body.classList.add("idle"), 1500);
  });
  // Recording mode is a production tool: it starts on its own even under
  // reduced motion (the camera then cuts instead of flying).
  setTimeout(playTour, 1500);
}
