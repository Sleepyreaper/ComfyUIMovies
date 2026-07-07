"""Tests for the UI->API workflow converter and prompt expansion (offline)."""
import json
import os

import pytest

from comfymovies.convert import convert
from comfymovies.prompts import expand_template

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
UI = os.path.join(ROOT, "workflows", "ltx2_3_t2v_ui.json")
OI = os.path.join(ROOT, "workflows", "object_info.snapshot.json")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(UI) and os.path.exists(OI)),
    reason="template/object_info snapshots not present",
)


def test_convert_produces_runnable_api_graph():
    with open(UI) as f:
        wf = json.load(f)
    with open(OI) as f:
        oi = json.load(f)
    api = convert(wf, oi)

    assert api, "converter produced empty graph"
    ids = set(api)
    # No frontend-only passthrough nodes leak through.
    assert not any(n["class_type"] == "Reroute" for n in api.values())
    # Every reference points at a real node.
    for node in api.values():
        for v in node["inputs"].values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) \
                    and v[0].isdigit():
                assert v[0] in ids, f"dangling ref {v}"
    # A SaveVideo terminates the graph and is fed by a connection, not a literal.
    saves = [n for n in api.values() if n["class_type"] == "SaveVideo"]
    assert saves and isinstance(saves[0]["inputs"]["video"], list)


def test_expand_template_scene_count_and_style():
    scenes = expand_template(
        "GI Joe style 1980s cartoon, soldiers fight the Serpent Empire, "
        "cel animation", 4)
    assert len(scenes) == 4
    # Style hint carried onto every beat.
    assert all("cel animation" in s.prompt for s in scenes)


def test_expand_template_single_scene():
    scenes = expand_template("a lone astronaut on mars, photorealistic", 1)
    assert len(scenes) == 1


def test_load_scene_file_text(tmp_path):
    from comfymovies.prompts import load_scene_file
    p = tmp_path / "m.txt"
    p.write_text("# comment\nscene one\n\nscene two\n")
    scenes = load_scene_file(str(p))
    assert [s.prompt for s in scenes] == ["scene one", "scene two"]


def test_load_scene_file_json_with_weights(tmp_path):
    import json as _json
    from comfymovies.prompts import load_scene_file
    p = tmp_path / "m.json"
    p.write_text(_json.dumps({"scenes": [
        {"prompt": "a", "weight": 2.0}, "b",
    ]}))
    scenes = load_scene_file(str(p))
    assert [s.prompt for s in scenes] == ["a", "b"]
    assert scenes[0].weight == 2.0
