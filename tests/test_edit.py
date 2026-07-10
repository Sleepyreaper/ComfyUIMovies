"""Offline unit tests for the FLUX.2 image-edit graph — no server required."""
import pytest

from comfymovies.edit import CLIP, TURBO_LORA, UNET, VAE, build_edit_workflow


def _by_type(graph, class_type):
    return [n for n in graph.values() if n["class_type"] == class_type]


def test_fast_uses_turbo_lora_and_flux2_assets():
    g = build_edit_workflow(["a.png"], "make it snow", 1024, 768, quality="fast")
    assert _by_type(g, "UNETLoader")[0]["inputs"]["unet_name"] == UNET
    lora = _by_type(g, "LoraLoaderModelOnly")
    assert lora and lora[0]["inputs"]["lora_name"] == TURBO_LORA
    assert _by_type(g, "CLIPLoader")[0]["inputs"]["type"] == "flux2"
    assert _by_type(g, "CLIPLoader")[0]["inputs"]["clip_name"] == CLIP
    assert _by_type(g, "VAELoader")[0]["inputs"]["vae_name"] == VAE
    assert _by_type(g, "Flux2Scheduler")[0]["inputs"]["steps"] == 8
    assert _by_type(g, "FluxGuidance")[0]["inputs"]["guidance"] == 3.0


def test_high_drops_lora_and_bumps_steps():
    g = build_edit_workflow(["a.png"], "make it snow", 1024, 768, quality="high")
    assert not _by_type(g, "LoraLoaderModelOnly")
    assert _by_type(g, "Flux2Scheduler")[0]["inputs"]["steps"] == 28
    assert _by_type(g, "FluxGuidance")[0]["inputs"]["guidance"] == 4.0


def test_scheduler_takes_source_dimensions():
    g = build_edit_workflow(["a.png"], "x", 1360, 752, quality="fast")
    sched = _by_type(g, "Flux2Scheduler")[0]["inputs"]
    assert sched["width"] == 1360 and sched["height"] == 752


def test_source_encode_is_both_reference_and_sampler_canvas():
    """The first image's VAE-encoded latent must feed ReferenceLatent AND be the
    SamplerCustomAdvanced canvas, so edits preserve the source composition."""
    g = build_edit_workflow(["a.png"], "make it night", 1024, 768)
    enc0 = ["enc0", 0]
    # first ReferenceLatent references the source latent
    assert g["ref0"]["inputs"]["latent"] == enc0
    # instruction flows through ReferenceLatent before guidance
    assert g["ref0"]["inputs"]["conditioning"] == ["pos", 0]
    assert g["guid"]["inputs"]["conditioning"] == ["ref0", 0]
    # sampler canvas is the encoded source
    assert g["sample"]["inputs"]["latent_image"] == enc0


def test_multi_reference_chains_all_images():
    g = build_edit_workflow(["a.png", "b.png"], "add the robot", 1024, 768)
    # two encodes, two chained ReferenceLatents
    assert g["ref1"]["inputs"]["latent"] == ["enc1", 0]
    assert g["ref1"]["inputs"]["conditioning"] == ["ref0", 0]
    assert g["guid"]["inputs"]["conditioning"] == ["ref1", 0]


def test_kontext_scale_between_load_and_encode():
    g = build_edit_workflow(["a.png"], "x", 1024, 768)
    assert g["scale0"]["class_type"] == "FluxKontextImageScale"
    assert g["scale0"]["inputs"]["image"] == ["img0", 0]
    assert g["enc0"]["inputs"]["pixels"] == ["scale0", 0]


def test_rejects_empty_and_bad_quality():
    with pytest.raises(ValueError):
        build_edit_workflow([], "x", 1024, 768)
    with pytest.raises(ValueError):
        build_edit_workflow(["a.png"], "x", 1024, 768, quality="ultra")


# --- Qwen-Image-Edit 2511 backend ------------------------------------------
from comfymovies.edit import (  # noqa: E402
    QWEN_ACTION_LORA, QWEN_EDIT_CLIP, QWEN_EDIT_UNET, QWEN_EDIT_VAE,
    QWEN_LIGHTNING_LORA, build_qwen_edit_workflow,
)


def test_qwen_fast_stacks_action_and_lightning_loras():
    g = build_qwen_edit_workflow(["a.png"], "action the scene", quality="fast")
    assert _by_type(g, "UNETLoader")[0]["inputs"]["unet_name"] == QWEN_EDIT_UNET
    loras = {n["inputs"]["lora_name"] for n in _by_type(g, "LoraLoaderModelOnly")}
    assert loras == {QWEN_ACTION_LORA, QWEN_LIGHTNING_LORA}
    ks = _by_type(g, "KSampler")[0]["inputs"]
    assert ks["steps"] == 4 and ks["cfg"] == 1.0
    assert _by_type(g, "CLIPLoader")[0]["inputs"]["type"] == "qwen_image"
    assert _by_type(g, "CLIPLoader")[0]["inputs"]["clip_name"] == QWEN_EDIT_CLIP
    assert _by_type(g, "VAELoader")[0]["inputs"]["vae_name"] == QWEN_EDIT_VAE
    assert _by_type(g, "ModelSamplingAuraFlow")[0]["inputs"]["shift"] == 3.1


def test_qwen_high_drops_lightning_keeps_action():
    g = build_qwen_edit_workflow(["a.png"], "x", quality="high")
    loras = {n["inputs"]["lora_name"] for n in _by_type(g, "LoraLoaderModelOnly")}
    assert loras == {QWEN_ACTION_LORA}
    ks = _by_type(g, "KSampler")[0]["inputs"]
    assert ks["steps"] == 40 and ks["cfg"] == 4.0


def test_qwen_action_strength_zero_removes_action_lora():
    g = build_qwen_edit_workflow(["a.png"], "x", quality="high", action_strength=0)
    assert not _by_type(g, "LoraLoaderModelOnly")


def test_qwen_negative_prompt_is_empty_and_positive_carries_instruction():
    g = build_qwen_edit_workflow(["a.png"], "make it snow")
    encs = _by_type(g, "TextEncodeQwenImageEditPlus")
    prompts = sorted(n["inputs"]["prompt"] for n in encs)
    assert prompts == ["", "make it snow"]


def test_qwen_canvas_is_scaled_source_and_pos_neg_via_multiref():
    g = build_qwen_edit_workflow(["a.png"], "x")
    # VAEEncode canvas comes from the Kontext-scaled source
    assert g["lat"]["inputs"]["pixels"] == ["scale", 0]
    assert g["scale"]["inputs"]["image"] == ["img0", 0]
    # KSampler pos/neg flow through FluxKontextMultiReferenceLatentMethod
    assert g["ks"]["inputs"]["positive"] == ["posr", 0]
    assert g["ks"]["inputs"]["negative"] == ["negr", 0]
    assert g["ks"]["inputs"]["latent_image"] == ["lat", 0]


def test_qwen_extra_refs_feed_image2_image3():
    g = build_qwen_edit_workflow(["a.png", "b.png", "c.png"], "combine")
    pos = [n for nid, n in g.items()
           if n["class_type"] == "TextEncodeQwenImageEditPlus"
           and n["inputs"]["prompt"] == "combine"][0]["inputs"]
    assert pos["image2"] == ["img2", 0]
    assert pos["image3"] == ["img3", 0]


def test_qwen_rejects_empty_and_bad_quality():
    with pytest.raises(ValueError):
        build_qwen_edit_workflow([], "x")
    with pytest.raises(ValueError):
        build_qwen_edit_workflow(["a.png"], "x", quality="ultra")
