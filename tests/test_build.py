"""Offline unit tests for graph construction — no ComfyUI server required."""
import pytest

from comfymovies.build import (
    AUDIO_MAX_FRAMES, MovieSpec, Scene, build_workflow, frames_for, snap,
)


def _refs(graph):
    """All [node_id, slot] input references in a graph."""
    out = []
    for node in graph.values():
        for v in node["inputs"].values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                out.append(v)
    return out


def test_frames_snap_to_8n_plus_1():
    assert frames_for(2, 24) == 49
    assert (frames_for(5, 24) - 1) % 8 == 0
    assert (frames_for(60, 24) - 1) % 8 == 0


def test_snap_multiple_of_64():
    assert snap(480, 64) == 512
    assert snap(896, 64) == 896
    assert snap(10, 64) == 64


def test_no_dangling_references():
    spec = MovieSpec(scenes=[Scene("a"), Scene("b"), Scene("c")],
                     seconds=4, fps=24)
    g = build_workflow(spec)
    ids = set(g)
    for ref in _refs(g):
        assert ref[0] in ids, f"dangling ref {ref}"


def test_short_clip_has_native_audio():
    g = build_workflow(MovieSpec(scenes=[Scene("a")], seconds=5, fps=24))
    assert "av" in g and "adec" in g
    assert "audio" in g["video"]["inputs"]


def test_long_clip_drops_audio_and_uses_context_windows():
    g = build_workflow(MovieSpec(scenes=[Scene("a")], seconds=60, fps=24))
    assert "av" not in g and "adec" not in g
    assert "ctx" in g  # long clip -> context windows for seamlessness
    assert "audio" not in g["video"]["inputs"]


def test_audio_gate_threshold():
    # ~40s (<=1000 frames) keeps native audio; ~45s (>1000) drops it.
    assert frames_for(40, 24) <= AUDIO_MAX_FRAMES
    assert frames_for(45, 24) > AUDIO_MAX_FRAMES
    short = build_workflow(MovieSpec(scenes=[Scene("a")], seconds=40, fps=24))
    long = build_workflow(MovieSpec(scenes=[Scene("a")], seconds=45, fps=24))
    assert "adec" in short and "adec" not in long


def test_enhancement_nodes_present_by_default():
    g = build_workflow(MovieSpec(scenes=[Scene("a")], seconds=5, fps=24))
    assert any(n["class_type"] == "TextGenerateLTX2Prompt" for n in g.values())


def test_enhancement_can_be_disabled():
    g = build_workflow(MovieSpec(scenes=[Scene("a")], seconds=5, fps=24,
                                 enhance=False))
    assert not any(
        n["class_type"] == "TextGenerateLTX2Prompt" for n in g.values()
    )


def test_scheduled_scenes_create_temporal_regions():
    spec = MovieSpec(scenes=[Scene("a"), Scene("b"), Scene("c")],
                     seconds=6, fps=24, schedule_scenes=True)
    g = build_workflow(spec)
    regions = [n for n in g.values()
               if n["class_type"] == "ConditioningSetAreaPercentageVideo"]
    assert len(regions) == 3
    # z (start offsets) should be monotonically increasing and start at 0.
    starts = sorted(r["inputs"]["z"] for r in regions)
    assert starts[0] == 0.0
    assert starts == sorted(starts)


def test_single_scene_not_scheduled():
    g = build_workflow(MovieSpec(scenes=[Scene("a")], seconds=4, fps=24))
    assert not any(
        n["class_type"] == "ConditioningSetAreaPercentageVideo"
        for n in g.values()
    )


def test_save_node_terminates_graph():
    g = build_workflow(MovieSpec(scenes=[Scene("a")], seconds=4, fps=24))
    save = [n for n in g.values() if n["class_type"] == "SaveVideo"]
    assert len(save) == 1
    assert save[0]["inputs"]["video"][0] == "video"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
