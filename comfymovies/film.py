"""Shot-based cinematic film builder: FLUX 2 keyframe -> WAN 2.2 I2V per shot.

Real anime/Hollywood films are sequences of individually composed shots, not one
continuous take. This module builds one gorgeous shot at a time:

* **FLUX 2** paints a film-grade keyframe (best local image model).
* **WAN 2.2** (20-step MoE) animates it with subtle, natural motion.

Render keyframes first (cheap, ~1 min) to check composition and character
consistency, then animate the keepers (~13-15 min each at 720p). Assemble the
finished shots into a film with :mod:`comfymovies.post` (concat + music).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .wan import (
    CLIP as WAN_CLIP, UNET_I2V_HIGH as UNET_HIGH, UNET_I2V_LOW as UNET_LOW,
    VAE as WAN_VAE, WAN_NEGATIVE, WAN_SHIFT, snap, wan_frames,
)

# FLUX 2 keyframe assets (confirmed installed).
FLUX_UNET = "flux2_dev_fp8mixed.safetensors"
FLUX_TURBO_LORA = "Flux_2-Turbo-LoRA_comfyui.safetensors"
FLUX_CLIP = "mistral_3_small_flux2_bf16.safetensors"
FLUX_VAE = "full_encoder_small_decoder.safetensors"
FLUX_STEPS = 8
FLUX_GUIDANCE = 4.0
FLUX_NEG = "blurry, low quality, deformed, extra limbs, bad anatomy, text, watermark, ugly, jpeg artifacts"

# WAN 2.2 is trained mostly on real footage and will drift an anime keyframe
# toward photorealism as it animates. We fight that by (a) anchoring the anime
# style in WAN's OWN prompt and (b) pushing realism terms into its negative.
ANIME_MOTION_ANCHOR = (
    "2D hand-drawn Studio Ghibli cel anime, flat anime shading, painterly anime "
    "background, bold clean linework, consistent traditional anime art style, "
    "NOT photorealistic, NOT 3d"
)
ANIME_NEGATIVE = (
    WAN_NEGATIVE + ", photorealistic, realistic, photograph, real person, "
    "live action, 3d, 3d render, cgi, realistic skin texture, uncanny, "
    "film still of real people, plastic render"
)


@dataclass
class Shot:
    """One film shot. ``keyframe`` describes the composed still; ``motion`` the
    (subtle) movement WAN adds. Character/style strings are prepended so a cast
    stays consistent across shots."""
    keyframe: str
    motion: str
    seconds: float = 6.0
    seed: int = 0


@dataclass
class FilmSpec:
    shots: list[Shot]
    character: str = ""     # consistent subject description, prepended to every shot
    style: str = ""         # consistent art-direction, appended to every shot
    width: int = 1280
    height: int = 720
    fps: int = 16
    wan_steps: int = 20
    wan_cfg: float = 3.5
    boundary_fraction: float = 0.5
    prefix: str = "ComfyUIMovies/film"
    reference_image: str = ""   # filename in ComfyUI /input; locks character+creature

    def normalized(self) -> "FilmSpec":
        self.width = snap(self.width, 16)
        self.height = snap(self.height, 16)
        return self

    def keyframe_prompt(self, shot: Shot) -> str:
        parts = [p for p in (self.character, shot.keyframe, self.style) if p]
        return ". ".join(parts)

    def motion_prompt(self, shot: Shot) -> str:
        # Keep the anime style anchored during animation so WAN doesn't drift to
        # photorealism; also restate the locked character so WAN's text guidance
        # reinforces (rather than fights) the reference-locked keyframe.
        parts = [p for p in (self.character, shot.motion) if p]
        return f"{'. '.join(parts)}. {ANIME_MOTION_ANCHOR}"


def _flux_keyframe(node, key: str, prompt: str, seed: int, W: int, H: int,
                   ref_image: str = "") -> list:
    """FLUX 2 keyframe subgraph; returns an IMAGE ref.

    When ``ref_image`` (a filename in ComfyUI's /input) is given, it is
    VAE-encoded and chained into the conditioning via ``ReferenceLatent`` so the
    locked character + creature design is reproduced in every shot (FLUX.2's
    native multi-reference mechanism)."""
    node(f"fu_{key}", "UNETLoader", {"unet_name": FLUX_UNET, "weight_dtype": "default"})
    node(f"fl_{key}", "LoraLoaderModelOnly", {
        "model": [f"fu_{key}", 0], "lora_name": FLUX_TURBO_LORA, "strength_model": 1.0})
    node(f"fc_{key}", "CLIPLoader", {"clip_name": FLUX_CLIP, "type": "flux2", "device": "default"})
    node(f"fv_{key}", "VAELoader", {"vae_name": FLUX_VAE})
    node(f"fp_{key}", "CLIPTextEncode", {"text": prompt, "clip": [f"fc_{key}", 0]})
    cond = [f"fp_{key}", 0]
    if ref_image:
        node(f"fli_{key}", "LoadImage", {"image": ref_image})
        node(f"fis_{key}", "FluxKontextImageScale", {"image": [f"fli_{key}", 0]})
        node(f"fre_{key}", "VAEEncode", {"pixels": [f"fis_{key}", 0], "vae": [f"fv_{key}", 0]})
        node(f"frl_{key}", "ReferenceLatent", {"conditioning": cond, "latent": [f"fre_{key}", 0]})
        cond = [f"frl_{key}", 0]
    node(f"fg_{key}", "FluxGuidance", {"conditioning": cond, "guidance": FLUX_GUIDANCE})
    node(f"fgd_{key}", "BasicGuider", {"model": [f"fl_{key}", 0], "conditioning": [f"fg_{key}", 0]})
    node(f"fn_{key}", "RandomNoise", {"noise_seed": seed})
    node(f"fs_{key}", "Flux2Scheduler", {"steps": FLUX_STEPS, "width": W, "height": H})
    node(f"fk_{key}", "KSamplerSelect", {"sampler_name": "euler"})
    node(f"flat_{key}", "EmptyFlux2LatentImage", {"width": W, "height": H, "batch_size": 1})
    node(f"fsa_{key}", "SamplerCustomAdvanced", {
        "noise": [f"fn_{key}", 0], "guider": [f"fgd_{key}", 0], "sampler": [f"fk_{key}", 0],
        "sigmas": [f"fs_{key}", 0], "latent_image": [f"flat_{key}", 0]})
    node(f"fd_{key}", "VAEDecode", {"samples": [f"fsa_{key}", 0], "vae": [f"fv_{key}", 0]})
    return [f"fd_{key}", 0]


def build_keyframe_workflow(spec: FilmSpec, index: int) -> dict:
    """Graph that renders ONLY the FLUX 2 keyframe for one shot (fast preview)."""
    spec = spec.normalized()
    shot = spec.shots[index]
    g: dict[str, dict] = {}

    def node(nid, ct, inp):
        g[nid] = {"class_type": ct, "inputs": inp}
        return nid

    img = _flux_keyframe(node, str(index), spec.keyframe_prompt(shot), shot.seed,
                         spec.width, spec.height, spec.reference_image)
    node("save", "SaveImage", {"images": img, "filename_prefix": f"{spec.prefix}_kf{index:02d}"})
    return g


def build_shot_workflow(spec: FilmSpec, index: int) -> dict:
    """Graph that renders one full shot: FLUX 2 keyframe -> WAN 2.2 I2V clip."""
    spec = spec.normalized()
    shot = spec.shots[index]
    W, H = spec.width, spec.height
    length = wan_frames(shot.seconds, spec.fps)
    steps = spec.wan_steps
    cfg = spec.wan_cfg
    boundary = max(1, min(steps - 1, round(steps * spec.boundary_fraction)))

    g: dict[str, dict] = {}

    def node(nid, ct, inp):
        g[nid] = {"class_type": ct, "inputs": inp}
        return nid

    keyframe = _flux_keyframe(node, str(index), spec.keyframe_prompt(shot), shot.seed, W, H,
                              spec.reference_image)

    node("wclip", "CLIPLoader", {"clip_name": WAN_CLIP, "type": "wan", "device": "default"})
    node("wvae", "VAELoader", {"vae_name": WAN_VAE})
    node("whi", "UNETLoader", {"unet_name": UNET_HIGH, "weight_dtype": "default"})
    node("wlo", "UNETLoader", {"unet_name": UNET_LOW, "weight_dtype": "default"})
    node("mshi", "ModelSamplingSD3", {"model": ["whi", 0], "shift": WAN_SHIFT})
    node("mslo", "ModelSamplingSD3", {"model": ["wlo", 0], "shift": WAN_SHIFT})
    node("wpos", "CLIPTextEncode", {"text": spec.motion_prompt(shot), "clip": ["wclip", 0]})
    node("wneg", "CLIPTextEncode", {"text": ANIME_NEGATIVE, "clip": ["wclip", 0]})
    node("i2v", "WanImageToVideo", {
        "positive": ["wpos", 0], "negative": ["wneg", 0], "vae": ["wvae", 0],
        "width": W, "height": H, "length": length, "batch_size": 1, "start_image": keyframe})
    node("khi", "KSamplerAdvanced", {
        "model": ["mshi", 0], "add_noise": "enable", "noise_seed": shot.seed,
        "steps": steps, "cfg": cfg, "sampler_name": "euler", "scheduler": "simple",
        "positive": ["i2v", 0], "negative": ["i2v", 1], "latent_image": ["i2v", 2],
        "start_at_step": 0, "end_at_step": boundary, "return_with_leftover_noise": "enable"})
    node("klo", "KSamplerAdvanced", {
        "model": ["mslo", 0], "add_noise": "disable", "noise_seed": shot.seed,
        "steps": steps, "cfg": cfg, "sampler_name": "euler", "scheduler": "simple",
        "positive": ["i2v", 0], "negative": ["i2v", 1], "latent_image": ["khi", 0],
        "start_at_step": boundary, "end_at_step": steps, "return_with_leftover_noise": "disable"})
    node("dec", "VAEDecode", {"samples": ["klo", 0], "vae": ["wvae", 0]})
    node("vid", "CreateVideo", {"images": ["dec", 0], "fps": float(spec.fps)})
    node("save", "SaveVideo", {
        "video": ["vid", 0], "filename_prefix": f"{spec.prefix}_shot{index:02d}",
        "format": "auto", "codec": "auto"})
    return g
