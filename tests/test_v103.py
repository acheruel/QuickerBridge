"""v1.03 (Notion "QuickerBridge update 1.03"): independent parts of a deck do
not load each other, and the FT summary shows the shear fraction without Fs."""

import json
from pathlib import Path

import numpy as np
import pytest

from quickerbridge import distribution as D
from quickerbridge import engine
from quickerbridge.models import Model

ROOT = Path(__file__).resolve().parents[1]


def bridge(lengths, supports, simple=()):
    data = json.loads(
        (ROOT / "examples" / "example-two-span-steel.quickerbridge.json").read_text(
            encoding="utf-8"
        )
    )
    m = data["model"]
    span = m["spans"][0]
    m["spans"] = [
        {**span, "length": length, "simple": i in simple}
        for i, length in enumerate(lengths)
    ]
    m["supports"] = supports
    m["live"]["vehicle"] = "CL750QC"
    return Model.model_validate(m)


def two_structures():
    # The bridge of the report: 4 + 3 continuous spans on a split pier.
    return bridge(
        [76, 95, 95, 76, 82.5, 102.5, 82.5],
        ["roller", "pin", "pin", "roller", "split", "roller", "pin", "roller"],
    )


def test_lane_load_stays_on_its_own_structure():
    basis = engine.Basis(two_structures()).build()
    units = basis.span_units()
    x, nx = basis.x, basis.nx
    right = np.flatnonzero(x > 342.001)
    left = np.flatnonzero(x < 341.999)
    for block in range(3):  # V, M, δ
        assert not units[:4][:, block * nx + right].any()
        assert not units[4:][:, block * nx + left].any()
    index = nx + int(np.argmin(np.abs(x - 355.41)))
    for sense in ("max", "min"):
        pieces = engine.lane_pieces(basis, "spans", index, sense)
        assert pieces and all(a >= 342.0 - 1e-9 for a, _, _ in pieces)
    assert engine.lane_pieces(basis, "spans", index, "max") == [
        (342.0, 424.5, 1.0),
        (527.0, 609.5, 1.0),
    ]


def test_axles_have_no_effect_across_a_split_pier():
    basis = engine.Basis(two_structures()).build()
    x, nx, ns = basis.x, basis.nx, basis.ns
    u = basis.unit(np.array([100.0, 300.0, 400.0, 600.0]))
    right = np.flatnonzero(x > 342.001)
    left = np.flatnonzero(x < 341.999)
    for block in range(3):
        assert not u[:2][:, block * nx + right].any()
        assert not u[2:][:, block * nx + left].any()
    # Reactions: supports 1-5 for the left part, 5-8 for the right part.
    r = u[:, 3 * nx : 3 * nx + ns]
    assert not r[:2, 5:].any() and not r[2:, :4].any()
    np.testing.assert_allclose(r.sum(axis=1), 1.0, atol=1e-9)
    # A load on the split pier itself goes to its bearing on either side.
    assert basis.coupled(np.array([342.0])).all()


def test_simple_span_is_an_independent_part():
    model = bridge(
        [30, 40, 25, 40, 30], ["roller", "pin", "pin", "pin", "pin", "roller"], [2]
    )
    basis = engine.Basis(model).build()
    assert list(basis.part_bounds) == [70.0, 95.0]
    units = basis.span_units()
    nx = basis.nx
    inside = np.flatnonzero((basis.x > 70.001) & (basis.x < 94.999))
    outside = np.flatnonzero((basis.x < 69.999) | (basis.x > 95.001))
    assert not units[2][nx + outside].any()  # M outside the simple span
    assert not units[[0, 1, 3, 4]][:, nx + inside].any()
    assert not units[:2][:, nx + outside[basis.x[outside] > 95]].any()


def test_continuous_bridge_has_one_part():
    basis = engine.Basis(bridge([30, 40, 30], ["roller", "pin", "pin", "roller"]))
    assert not len(basis.part_bounds) and basis.coupled(np.array([10.0])) is None


def test_envelope_unchanged_by_the_other_structure():
    # The right structure analysed alone gives the same extremes.
    whole = engine.analyse(two_structures())
    alone = engine.analyse(
        bridge([82.5, 102.5, 82.5], ["roller", "roller", "pin", "roller"])
    )
    x = np.array(whole["x"])
    keep = np.flatnonzero(x > 342.0 + 1e-6)
    for key in ("max", "min"):
        got = np.array(whole[key]["M"])[keep]
        np.testing.assert_allclose(
            got, np.array(alone[key]["M"])[-len(keep) :], rtol=1e-6, atol=1e-3
        )


def test_ft_summary_shows_shear_fraction_without_fs():
    model = bridge([30, 40, 30], ["roller", "pin", "pin", "roller"])
    model.distribution.enabled = True
    model.distribution.skew = 30
    model.distribution.girder = "exterior"
    zones = D.truck_fraction(model)["zones"]
    assert any(z["Fs"] > 1.001 for z in zones)
    for z in zones:
        assert z["FT_V"] >= z["FT_V_raw"]
    span = next(z for z in zones if z["where"] == "span:1")
    assert span["FT_V"] == pytest.approx(span["FT_V_raw"] * span["Fs"])
    model.distribution.girder = "interior"
    for z in D.truck_fraction(model)["zones"]:
        assert z["FT_V"] == pytest.approx(z["FT_V_raw"])
    js = (ROOT / "dist" / "axle.js").read_text(encoding="utf-8")
    assert "fmt(z.FT_V_raw??z.FT_V,3)" in js
    assert js.count("axleFsNote:") == 2  # FR and EN
