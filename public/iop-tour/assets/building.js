// Geometry for the Physical Sciences Building (F.O. Santos Hall), UPLB.
// Room content (names, descriptions, photos, room boxes) lives in index.html;
// this file only holds the shell the rooms sit in.

// Building outline from OpenStreetMap (© OpenStreetMap contributors, ODbL),
// [lon, lat], first point repeated at the end like the source way.
export const OUTLINE = [
  [121.2416125, 14.1642209], [121.2419549, 14.1643768], [121.241917, 14.1644552],
  [121.2417559, 14.1643881], [121.2417017, 14.164481], [121.2418619, 14.1645542],
  [121.241859, 14.1645605], [121.2418555, 14.164568], [121.2418382, 14.164605],
  [121.2418772, 14.1646197], [121.2419107, 14.1646336], [121.2419334, 14.164644],
  [121.2422479, 14.1647774], [121.2422917, 14.1647631], [121.2423453, 14.1646601],
  [121.2421532, 14.1645763], [121.2419666, 14.164502], [121.2419925, 14.1644529],
  [121.2422046, 14.1645381], [121.2423174, 14.1643073], [121.2420906, 14.1642075],
  [121.2421096, 14.1641583], [121.2422587, 14.1642229], [121.2424156, 14.1639086],
  [121.2421674, 14.1637937], [121.2420969, 14.1639216], [121.2420752, 14.1639612],
  [121.2417743, 14.1638317], [121.241708, 14.1639899], [121.2417354, 14.1640025],
  [121.2420628, 14.16415], [121.2420321, 14.1642142], [121.2417085, 14.1640682],
  [121.2416803, 14.1640574], [121.2416125, 14.1642209],
];

/** UPLB Oblation statue (OpenStreetMap). The Wing C entrance faces it. */
export const OBLATION = [121.2415518, 14.1650027];

export const FLOORS = 3;
export const FLOOR_HEIGHT = 3.6;

/**
 * Wing C runs along its front facade (the side facing Oblation Park).
 * Room boxes in index.html use this frame, in metres from the outline's
 * centroid: `along` follows the facade toward the east end, `across` grows
 * toward the front.
 *
 * ponytail: the angle is read off one OSM facade edge. If the outline is ever
 * remapped, re-read it from the new front edge.
 */
export const WING_ANGLE_DEG = 23.52;

/** Circulation spaces. Not rooms, so they live here instead of the room list. */
export const HALLS = [
  { label: "Hallway", floor: 2, along: [-1, 7], across: [15.8, 34.4] },
  { label: null, floor: 2, along: [7, 51], across: [25.7, 28.8] },
  { label: null, floor: 2, along: [-20, -1], across: [24.5, 28] },
  { label: "Hallway", floor: 1, along: [-1, 7], across: [6, 34.4] },
];

/**
 * The Wing C entrance bay: the stretch of the front facade (along, in metres) where
 * the ground-floor hallway opens toward Oblation Park, and that facade's `across`.
 * Read off the outline (the short front edges between the two long wings).
 */
export const ENTRANCE = { along: [-2.3, 8.8], across: 34.4 };

export const STAIRS = [
  { floor: 1, along: [1.25, 5.75], across: [16.75, 22.75] },
  { floor: 2, along: [1.25, 5.75], across: [16.75, 22.75] },
];
