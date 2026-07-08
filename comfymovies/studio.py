"""The "studio" pipeline: image-model keyframes animated by WAN 2.2, chained.

This combines the best tool for each job into one ComfyUI graph:

1. **Z-Image** (fast, razor-sharp image model) paints an art-directed keyframe
   for the opening beat.
2. **WAN 2.2 14B** (mixture-of-experts, best sustained motion) animates it via
   ``WanImageToVideo`` (the t2v experts accept a ``start_image``).
3. Each following segment starts from the **previous segment's last frame** with
   the current scene's motion prompt, so the movie is one continuous, seamless
   take that stays sharp and keeps moving (no LTX-style motion decay).

All segments render at full WAN quality (multi-step, no speed LoRA) since GPU
time is cheap here. Output is 16 fps video-only (score with ElevenLabs).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .build import Scene, _scene_spans
from .prompts import MOTION_DIRECTIVE
from .wan import (
    CLIP, LORA_HIGH, LORA_LOW, UNET_HIGH, UNET_LOW, VAE, WAN_FPS, WAN_NEGATIVE,
    WAN_SHIFT, snap, wan_frames,
)

# Z-Image (turbo) keyframe assets, confirmed installed.
ZIMG_UNET = "z_image_turbo_bf16.safetensors"
ZIMG_CLIP = "qwen_3_4b.safetensors"        # loaded as CLIP type "lumina2"
ZIMG_VAE = "ae.safetensors"
ZIMG_STEPS = 8
ZIMG_NEG = "blurry, low quality, deformed, extra limbs, text, watermark, ugly"

# WAN 2.2 is trained around ~81-frame (5s @16fps) clips; keep each segment there.
SEGMENT_FRAMES = 81


@dataclass
class StudioSpec:
    scenes: list[Scene]
    width: int = 832
    height: int = 480
    seconds: float = 60.0
    fps: int = WAN_FPS
    seed: int = 0
    # WAN quality: full multi-step schedule (no lightx2v speed LoRA).
    wan_steps: int = 20
    wan_cfg: float = 3.5
    boundary_fraction: float = 0.5
    lightx2v: bool = False
    # Re-anchor to a fresh keyframe at each scene change (keeps art direction
    # from drifting over a long movie); within a scene, chain by last frame.
    reanchor_per_scene: bool = True
    style_suffix: str = ""
    filename_prefix: str = "ComfyUIMovies/studio"

    def normalized(self) -> "StudioSpec":
        self.width = snap(self.width, 16)
        self.height = snap(self.height, 16)
        return self


def plan_segment_scenes(spec: StudioSpec) -> list[Scene]:
    """Assign each ~5s segment the scene whose timeline slice it falls in."""
    total_frames = wan_frames(spec.seconds, spec.fps)
    # segments overlap by 1 frame at each join, so N segments -> N*80 + 1 frames.
    n = max(1, round((total_frames - 1) / (SEGMENT_FRAMES - 1)))
    scenes = [s for s in spec.scenes if s.prompt.strip()] or [Scene("a cinematic scene")]
    spans = _scene_spans(scenes)
    out: list[Scene] = []
    for i in range(n):
        mid = (i + 0.5) / n
        chosen = scenes[-1]
        for sc, (a, b) in zip(scenes, spans):
            if a <= mid < b:
                chosen = sc
                break
        out.append(chosen)
    return out


def _wan_experts(node, spec: StudioSpec):
    node("w_clip", "CLIPLoader", {"clip_name": CLIP, "type": "wan", "device": "default"})
    node("w_vae", "VAELoader", {"vae_name": VAE})
    node("w_hi", "UNETLoader", {"unet_name": UNET_HIGH, "weight_dtype": "default"})
    node("w_lo", "UNETLoader", {"unet_name": UNET_LOW, "weight_dtype": "default"})
    hi, lo = ["w_hi", 0], ["w_lo", 0]
    if spec.lightx2v:
        node("w_lhi", "LoraLoaderModelOnly", {
            "model": ["w_hi", 0], "lora_name": LORA_HIGH, "strength_model": 1.0})
        node("w_llo", "LoraLoaderModelOnly", {
            "model": ["w_lo", 0], "lora_name": LORA_LOW, "strength_model": 1.0})
        hi, lo = ["w_lhi", 0], ["w_llo", 0]
    node("ms_hi", "ModelSamplingSD3", {"model": hi, "shift": WAN_SHIFT})
    node("ms_lo", "ModelSamplingSD3", {"model": lo, "shift": WAN_SHIFT})
    return ["ms_hi", 0], ["ms_lo", 0]


def build_studio_workflow(spec: StudioSpec) -> dict:
    """Return a single ComfyUI graph for the full keyframe->WAN->chain movie."""
    spec = spec.normalized()
    seg_scenes = plan_segment_scenes(spec)
    W, H = spec.width, spec.height
    steps = 4 if spec.lightx2v else spec.wan_steps
    cfg = 1.0 if spec.lightx2v else spec.wan_cfg
    boundary = max(1, min(steps - 1, round(steps * spec.boundary_fraction)))
    suffix = (", " + spec.style_suffix) if spec.style_suffix else ""

    g: dict[str, dict] = {}

    def node(nid, class_type, inputs):
        g[nid] = {"class_type": class_type, "inputs": inputs}
        return nid

    # Shared loaders.
    hi_model, lo_model = _wan_experts(node, spec)
    node("zi_unet", "UNETLoader", {"unet_name": ZIMG_UNET, "weight_dtype": "default"})
    node("zi_clip", "CLIPLoader", {"clip_name": ZIMG_CLIP, "type": "lumina2", "device": "default"})
    node("zi_vae", "VAELoader", {"vae_name": ZIMG_VAE})

    def make_keyframe(key: str, prompt: str, seed: int) -> list:
        node(f"zip_{key}", "CLIPTextEncode", {"text": prompt + suffix, "clip": ["zi_clip", 0]})
        node(f"zin_{key}", "CLIPTextEncode", {"text": ZIMG_NEG, "clip": ["zi_clip", 0]})
        node(f"zil_{key}", "EmptySD3LatentImage", {"width": W, "height": H, "batch_size": 1})
        node(f"zik_{key}", "KSampler", {
            "model": ["zi_unet", 0], "seed": seed, "steps": ZIMG_STEPS, "cfg": 1.0,
            "sampler_name": "res_multistep", "scheduler": "simple",
            "positive": [f"zip_{key}", 0], "negative": [f"zin_{key}", 0],
            "latent_image": [f"zil_{key}", 0], "denoise": 1.0})
        node(f"zid_{key}", "VAEDecode", {"samples": [f"zik_{key}", 0], "vae": ["zi_vae", 0]})
        return [f"zid_{key}", 0]

    concat_ref: list | None = None
    prev_last: list | None = None
    prev_scene_prompt: str | None = None

    for i, scene in enumerate(seg_scenes):
        k = str(i)
        motion_prompt = f"{scene.prompt.strip()}. {MOTION_DIRECTIVE}"

        # Decide the start image for this segment.
        is_scene_change = scene.prompt != prev_scene_prompt
        if i == 0 or (spec.reanchor_per_scene and is_scene_change):
            start_ref = make_keyframe(k, scene.prompt.strip(), spec.seed + i)
        else:
            start_ref = prev_last  # continue from previous segment's last frame

        node(f"wp_{k}", "CLIPTextEncode", {"text": motion_prompt, "clip": ["w_clip", 0]})
        node(f"wn_{k}", "CLIPTextEncode", {"text": WAN_NEGATIVE, "clip": ["w_clip", 0]})
        node(f"i2v_{k}", "WanImageToVideo", {
            "positive": [f"wp_{k}", 0], "negative": [f"wn_{k}", 0], "vae": ["w_vae", 0],
            "width": W, "height": H, "length": SEGMENT_FRAMES, "batch_size": 1,
            "start_image": start_ref})
        node(f"khi_{k}", "KSamplerAdvanced", {
            "model": hi_model, "add_noise": "enable", "noise_seed": spec.seed + i,
            "steps": steps, "cfg": cfg, "sampler_name": "euler", "scheduler": "simple",
            "positive": [f"i2v_{k}", 0], "negative": [f"i2v_{k}", 1],
            "latent_image": [f"i2v_{k}", 2], "start_at_step": 0,
            "end_at_step": boundary, "return_with_leftover_noise": "enable"})
        node(f"klo_{k}", "KSamplerAdvanced", {
            "model": lo_model, "add_noise": "disable", "noise_seed": spec.seed + i,
            "steps": steps, "cfg": cfg, "sampler_name": "euler", "scheduler": "simple",
            "positive": [f"i2v_{k}", 0], "negative": [f"i2v_{k}", 1],
            "latent_image": [f"khi_{k}", 0], "start_at_step": boundary,
            "end_at_step": steps, "return_with_leftover_noise": "disable"})
        node(f"dec_{k}", "VAEDecode", {"samples": [f"klo_{k}", 0], "vae": ["w_vae", 0]})
        images_ref = [f"dec_{k}", 0]

        node(f"last_{k}", "ImageFromBatch", {
            "image": images_ref, "batch_index": SEGMENT_FRAMES - 1, "length": 1})
        prev_last = [f"last_{k}", 0]
        prev_scene_prompt = scene.prompt

        if concat_ref is None:
            concat_ref = images_ref
        else:
            node(f"trim_{k}", "ImageFromBatch", {
                "image": images_ref, "batch_index": 1, "length": SEGMENT_FRAMES - 1})
            node(f"cat_{k}", "ImageBatch", {
                "image1": concat_ref, "image2": [f"trim_{k}", 0]})
            concat_ref = [f"cat_{k}", 0]

    node("video", "CreateVideo", {"images": concat_ref, "fps": float(spec.fps)})
    node("save", "SaveVideo", {
        "video": ["video", 0], "filename_prefix": spec.filename_prefix,
        "format": "auto", "codec": "auto"})
    return g
