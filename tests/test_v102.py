"""v1.02: a project sent in the address (#project=...), and the project files
written by BridgeSketch 3D 0.7.3 (only the fields it knows, schema 17)."""

from pathlib import Path

import pytest

from quickerbridge.engine import analyse
from quickerbridge.models import Model
from quickerbridge.projects import SCHEMA_VERSION, validate_project

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted((ROOT / "tests" / "bridgesketch").glob("*.quickerbridge.json"))


def test_link_reader_is_wired_in_both_languages():
    js = ROOT / "dist" / "js"
    project = (js / "project.js").read_text(encoding="utf-8")
    assert "function qbLinkedText(hash)" in project
    assert "async function qbOpenLinked()" in project
    assert "history.replaceState" in project  # the fragment leaves the address
    assert "qbOpenLinked();" in (js / "extensions.js").read_text(encoding="utf-8")
    words = (js / "i18n.js").read_text(encoding="utf-8")
    assert words.count("linkInvalid:") == 2  # FR and EN


def test_bridgesketch_files_are_listed():
    assert len(FILES) == 5


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name.split(".")[0])
def test_bridgesketch_project_opens_and_solves(path):
    project = validate_project(path.read_text(encoding="utf-8"))
    assert project["schema_version"] == SCHEMA_VERSION
    model = Model.model_validate(project["model"])
    assert len(model.supports) == len(model.spans) + 1
    result = analyse(model)
    assert result["meta"]["elapsed"] >= 0


def test_bridgesketch_nebt_bridge_keeps_what_was_sent():
    path = ROOT / "tests" / "bridgesketch" / "nebt-3-spans.quickerbridge.json"
    model = Model.model_validate(validate_project(path.read_text("utf-8"))["model"])
    assert [s.length for s in model.spans] == [30, 35, 30]
    assert all(s.simple for s in model.spans)  # precast girders: no continuity
    assert model.sections[0].kind == "nebt" and model.sections[0].nebt == "NEBT1600"
    ft = model.distribution
    assert ft.enabled and ft.apply and ft.bridge_type == "slab_on_girder"
    assert (ft.girders, ft.spacing, ft.overhang, ft.skew) == (5, 2.4, 1.2, 15)
    assert [load.stage for load in model.dead] == ["steel", "3n"]
    # Fields BridgeSketch 3D does not write take the QuickerBridge defaults.
    assert model.live.vehicle == "CL750QC" and model.self_weight.apply


def test_bridgesketch_haunched_steel_bridge_is_non_prismatic():
    path = ROOT / "tests" / "bridgesketch" / "steel-haunched.quickerbridge.json"
    model = Model.model_validate(validate_project(path.read_text("utf-8"))["model"])
    assert model.nonprismatic and [s.depth for s in model.sections] == [1300, 2200]
    assert model.sections[0].composite.effective_width == 2400
    assert [z.profile for z in model.spans[1].zones] == [
        "parabolic",
        "constant",
        "parabolic",
    ]
    assert not any(s.simple for s in model.spans)
