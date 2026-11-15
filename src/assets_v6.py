"""CC0 furniture library of v6 (PLAN_V6 milestone 1).

Sources: Kenney Furniture Kit 2.0 and Poly Haven models (both CC0, see assets/LICENSES.md), downloaded by
scripts/download_assets_v6.py into data/assets_raw/.

Normalisation (load_asset): glTF is right-handed (x right, y up, +z towards the viewer); the v6 world frame
of src/layout.py is x right, y up, z forward (left-handed), so every vertex is imported as (x, y, -z), which
is a rotation-free change of frame, not a mirror image of the model. Then the model is rotated about y by
its catalogue `front` angle so that its FRONT FACES +z (towards
the room, when its back is against a wall at z = 0), its footprint is centred at x = z = 0 and its lowest
point is at y = 0. Kenney models (arbitrary scale) and Poly Haven models (metres) both end up in metres
after `fit_to` scales them to the layout footprint.

fit_to(mesh, w, d, h): scales the model into a footprint w (x, across the front) × d (z, depth) and height
h. The scale is uniform when possible; per-axis scale factors are kept within MAX_DISTORT of each other
(about 20 %) and the model is then centred inside the footprint (it may be a bit smaller than the box).

Colours: each model's texture / material colour is sampled per vertex and averaged per face
(face_colors), used for previews and as the fallback colour of hidden faces in the dollhouse.
"""

from __future__ import annotations

import functools
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "assets_raw"
KENNEY = RAW / "kenney" / "kit" / "Models" / "GLTF format"
POLYHAVEN = RAW / "polyhaven"
MAX_DISTORT = 1.2          # max ratio between per-axis scale factors

# class -> list of (source, model id, front angle in degrees, style tags)
# front angle: rotation about +y (towards +x) that turns the imported model so its front faces +z;
# all 180: every model faces glTF +z, i.e. -z after the frame change (checked on results/figures/v6_assets_sheet.png).
K, P = "kenney", "polyhaven"
CATALOG: dict[str, list] = {
    "bed": [(K, "bedDouble", 180, {"modern", "minimal", "scandinavian", "coastal", "contemporary"}),
            (P, "GothicBed_01", 180, {"traditional", "rustic", "art deco", "mediterranean", "farmhouse"})],
    "nightstand": [(P, "ClassicNightstand_01", 180, {"traditional", "art deco", "mediterranean"}),
                   (P, "painted_wooden_nightstand", 180, {"farmhouse", "rustic", "coastal", "bohemian"}),
                   (P, "side_table_01", 180, {"modern", "scandinavian", "mid-century", "contemporary", "industrial"})],
    "wardrobe": [(P, "painted_wooden_cabinet_02", 180, {"farmhouse", "coastal", "scandinavian"}),
                 (P, "vintage_cabinet_01", 180, {"traditional", "rustic", "art deco", "mediterranean"}),
                 (P, "chinese_cabinet", 180, {"bohemian", "industrial", "contemporary", "mid-century"})],
    "desk": [(P, "WoodenTable_03", 180, {"rustic", "farmhouse", "scandinavian", "coastal"}),
             (P, "metal_office_desk", 180, {"industrial", "modern", "contemporary"})],
    "chair": [(P, "dining_chair_02", 180, {"modern", "contemporary", "art deco", "mid-century"}),
              (P, "painted_wooden_chair_01", 180, {"farmhouse", "coastal", "scandinavian", "rustic"}),
              (P, "gallinera_chair", 180, {"mediterranean", "bohemian", "traditional", "mid-century"}),
              (P, "GreenChair_01", 180, {"industrial"})],
    "armchair": [(P, "ArmChair_01", 180, {"traditional", "art deco", "mediterranean", "farmhouse"}),
                 (P, "mid_century_lounge_chair", 180, {"mid-century", "modern", "contemporary", "industrial",
                                                       "scandinavian", "coastal", "minimal", "bohemian"})],
    "sofa": [(P, "Sofa_01", 180, {"traditional", "art deco", "mediterranean"}),
             (P, "sofa_02", 180, {"modern", "contemporary", "scandinavian", "coastal", "mid-century"}),
             (P, "sofa_03", 180, {"bohemian", "industrial", "farmhouse", "rustic"})],
    "dining table": [(P, "painted_wooden_table", 180, {"farmhouse", "rustic", "coastal", "mediterranean", "industrial"}),
                     (P, "wooden_table_02", 180, {"scandinavian", "bohemian", "traditional", "art deco", "modern", "contemporary", "mid-century"})],
    "coffee table": [(P, "CoffeeTable_01", 180, {"traditional", "farmhouse", "rustic"}),
                     (P, "modern_coffee_table_01", 180, {"modern", "contemporary", "minimal", "scandinavian"}),
                     (P, "coffee_table_round_01", 180, {"mid-century", "art deco", "mediterranean", "bohemian", "coastal"})],
    "TV unit": [(P, "modern_wooden_cabinet", 180, {"modern", "contemporary", "scandinavian", "mid-century"}),
                (P, "ClassicConsole_01", 180, {"traditional", "art deco", "mediterranean", "coastal"}),
                (K, "cabinetTelevision", 180, {"industrial", "bohemian", "farmhouse", "rustic"})],
    "television": [(K, "televisionModern", 180, set())],
    "sideboard": [(P, "GothicCommode_01", 180, {"traditional", "art deco", "rustic"}),
                  (P, "painted_wooden_cabinet", 180, {"farmhouse", "coastal", "mediterranean", "bohemian"}),
                  (P, "modern_wooden_cabinet", 180, {"modern", "contemporary", "scandinavian", "mid-century", "industrial"})],
    "shelf": [(P, "wooden_bookshelf_worn", 180, {"rustic", "farmhouse", "traditional", "industrial"}),
              (P, "Shelf_01", 180, {"modern", "scandinavian", "minimal", "contemporary", "mid-century", "coastal", "bohemian", "art deco", "mediterranean"})],
    "plant": [(K, "pottedPlant", 180, set())],
    "floor lamp": [(K, "lampRoundFloor", 180, set())],
    "counter": [(K, "kitchenCabinet", 180, set())],
    "stove": [(K, "kitchenStove", 180, set())],
    "sink": [(K, "kitchenSink", 180, set())],
    "fridge": [(K, "kitchenFridgeLarge", 180, set())],
    "island": [(K, "kitchenBar", 180, set())],
    "bathtub": [(K, "bathtub", 180, set())],
    "toilet": [(K, "toilet", 180, set())],
    "vanity": [(K, "bathroomCabinetDrawer", 180, set())],
    "rug": [(K, "rugRectangle", 180, set())],
}


def _path(src: str, mid: str) -> Path:
    return KENNEY / f"{mid}.glb" if src == K else POLYHAVEN / mid / f"{mid}.gltf"


def _rot_y(a_deg: float) -> np.ndarray:
    a = np.deg2rad(a_deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


@functools.lru_cache(maxsize=64)
def _load_raw(src: str, mid: str):
    """trimesh.Trimesh of a model in the v6 frame (z flipped), with per-face colours."""
    import trimesh
    warnings.filterwarnings("ignore")
    scene = trimesh.load(str(_path(src, mid)), force="scene")
    meshes = []
    for node in scene.graph.nodes_geometry:
        T, gname = scene.graph[node]
        g = scene.geometry[gname]
        if not hasattr(g, "faces") or len(g.faces) == 0:
            continue
        g = g.copy()
        try:
            vc = g.visual.to_color().vertex_colors[:, :3].astype(np.float32)
        except Exception:  # noqa: BLE001 (material without texture or colour)
            vc = np.full((len(g.vertices), 3), 180, np.float32)
        g.apply_transform(T)
        fc = vc[g.faces].mean(1).astype(np.uint8)
        m = trimesh.Trimesh(g.vertices, g.faces, process=False)
        m.visual.face_colors = np.concatenate([fc, np.full((len(fc), 1), 255, np.uint8)], 1)
        meshes.append(m)
    mesh = trimesh.util.concatenate(meshes)
    v = np.asarray(mesh.vertices).copy()
    v[:, 2] *= -1                                                     # glTF (right-handed) -> v6 world frame
    fc = mesh.visual.face_colors.copy()
    mesh = trimesh.Trimesh(v, np.asarray(mesh.faces)[:, ::-1], process=False)   # keep outward winding
    mesh.visual.face_colors = fc
    return mesh


def load_asset(src: str, mid: str, front: float = 0.0):
    """Normalised copy: front faces +z, footprint centred at the origin, lowest point at y = 0."""
    m = _load_raw(src, mid).copy()
    v = np.asarray(m.vertices) @ _rot_y(front).T
    lo, hi = v.min(0), v.max(0)
    v -= np.array([(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2])
    m.vertices = v
    return m


def fit_to(mesh, w: float, d: float, h: float):
    """Scale a normalised model into footprint w × d and height h (see module doc); returns a copy and
    the per-axis scale factors."""
    ext = mesh.extents
    s = np.array([w, h, d]) / np.maximum(ext, 1e-6)
    lo = s.min()
    s = np.minimum(s, lo * MAX_DISTORT)                               # proportions kept within 20 %
    m = mesh.copy()
    m.vertices = np.asarray(m.vertices) * s
    return m, s


def choose(cls: str, style: str, rng: np.random.Generator):
    """(source, id, front) of a model of class `cls`, preferring the style's tags; seeded."""
    opts = CATALOG[cls]
    words = set(style.lower().replace("-", " - ").split()) | {style.lower()}
    tagged = [o for o in opts if o[3] & words or style.lower() in o[3]]
    pool = tagged or opts
    o = pool[int(rng.integers(len(pool)))]
    return o[0], o[1], o[2]
