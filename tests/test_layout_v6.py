"""Unit tests of the v6 layout rules (PLAN_V6 milestone 1).  Run:  python -m pytest tests/test_layout_v6.py -q"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import layout_v6 as L  # noqa: E402

PROMPTS = [f"a {s} style {r}" for r in L.ROOMS for s in ("scandinavian", "industrial", "coastal")]
SEEDS = list(range(1, 31))


@pytest.fixture(scope="module")
def layouts():
    return {(p, s): L.generate(p, s) for p in PROMPTS for s in SEEDS}


def test_no_violations(layouts):
    bad = {k: v for k, lay in layouts.items() if (v := L.violations(lay))["total"]}
    assert not bad, f"{len(bad)} layouts with violations, e.g. {list(bad.items())[:3]}"


def test_main_piece_present(layouts):
    for (p, s), lay in layouts.items():
        assert any(i["cls"] == L.MAIN[lay["room"]] for i in lay["items"]), (p, s)


def test_wall_aligned(layouts):
    for lay in layouts.values():
        for i in lay["items"]:
            assert i["yaw"] in (0, 90, 180, 270)


def test_dining_chairs_face_table(layouts):
    for (p, s), lay in layouts.items():
        if lay["room"] != "dining room":
            continue
        chairs = [i for i in lay["items"] if i["cls"] == "chair"]
        assert len(chairs) >= 4, (p, s)
        t = next(i for i in lay["items"] if i["cls"] == "dining table")
        for c in chairs:
            assert np.dot(L.front_dir(c["yaw"]), [t["cx"] - c["cx"], t["cz"] - c["cz"]]) > 0


def test_bed_has_nightstands(layouts):
    n = [sum(i["cls"] == "nightstand" for i in lay["items"]) for lay in layouts.values() if lay["room"] == "bedroom"]
    assert np.mean([k == 2 for k in n]) >= 0.8, n


def test_room_size_and_ceiling(layouts):
    for lay in layouts.values():
        (w0, w1), (d0, d1) = L.ROOMS[lay["room"]]
        assert w0 <= lay["W"] <= w1 and d0 <= lay["D"] <= d1 and 2.6 <= lay["H"] <= 3.0


def test_windows(layouts):
    for lay in layouts.values():
        win = [o for o in lay["openings"] if o["kind"] == "window"]
        assert 1 <= len(win) <= 2
        assert all(o["w"] <= 1.6 and o["y1"] - o["y0"] <= 1.45 for o in win)
        assert sum(o["kind"] == "door" for o in lay["openings"]) == 1


def test_camera_clearance(layouts):
    for lay in layouts.values():
        x, y, z = lay["camera"]
        assert y == 1.5
        for i in lay["items"]:
            x0, z0, x1, z1 = L.extent(i)
            dx, dz = max(x0 - x, 0, x - x1), max(z0 - z, 0, z - z1)
            assert np.hypot(dx, dz) >= 0.5, (lay["prompt"], lay["seed"], i["cls"])


def test_deterministic_and_varied():
    a, b = L.generate("a coastal style bedroom", 22), L.generate("a coastal style bedroom", 22)
    assert a == b
    sigs = {tuple((i["cls"], round(i["cx"], 1), round(i["cz"], 1)) for i in L.generate("a coastal style bedroom", s)["items"])
            for s in range(10)}
    assert len(sigs) >= 8
