"""Build WAN 2.2 14B text-to-video / image-to-video graphs (API format).

WAN 2.2 is a mixture-of-experts video model: a *high-noise* expert denoises the
early steps and a *low-noise* expert finishes. With the lightx2v 4-step LoRAs it
renders in 4 steps; without them it runs a longer, higher-quality schedule. WAN
has notably stronger, more sustained subject/camera motion and sharper detail
than LTX-2, at the cost of a native ~16 fps / 81-frame clip length (chain for
longer). Frame counts follow the ``4n + 1`` rule; dims snap to /16.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Assets confirmed present on the target ComfyUI (RTX 5090 box).
CLIP = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
VAE = "wan_2.1_vae.safetensors"
UNET_HIGH = "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"
UNET_LOW = "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"
# Proper image-to-video UNETs: unlike the t2v models above, these actually
# honor the start_image, so a locked keyframe design survives the animation.
UNET_I2V_HIGH = "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
UNET_I2V_LOW = "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
LORA_HIGH = "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors"
LORA_LOW = "wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors"

WAN_FPS = 16
WAN_SHIFT = 5.0
# WAN's standard quality-negative (kept in English for readability).
WAN_NEGATIVE = (
    "static, still, frozen, motionless, overexposed, blurry details, "
    "low quality, worst quality, jpeg artifacts, subtitles, text, watermark, "
    "deformed, disfigured, extra limbs, gray washed-out colors"
)


def snap(value: int, multiple: int) -> int:
    return int(max(multiple, round(value / multiple) * multiple))


def wan_frames(seconds: float, fps: int = WAN_FPS) -> int:
    """Frame count for a duration, snapped to WAN's ``4n + 1`` rule."""
    raw = max(1, round(seconds * fps))
    n = max(1, round((raw - 1) / 4))
    return n * 4 + 1


@dataclass
class WanSpec:
    """Parameters for one WAN 2.2 clip."""
    prompt: str
    negative: str = WAN_NEGATIVE
    width: int = 832
    height: int = 480
    seconds: float = 5.0
    fps: int = WAN_FPS
    seed: int = 0
    # 4-step lightx2v (fast) vs a full multi-step schedule (slower, top quality).
    lightx2v: bool = True
    steps: int = 20            # used when lightx2v is False
    cfg: float = 3.5           # used when lightx2v is False
    boundary_fraction: float = 0.5   # where the high->low expert handoff happens
    filename_prefix: str = "ComfyUIMovies/wan"
    start_image_node: str | None = None   # for chained I2V (advanced)

    def normalized(self) -> "WanSpec":
        self.width = snap(self.width, 16)
        self.height = snap(self.height, 16)
        return self


def _experts(g, node, spec):
    """Wire the two MoE experts (high/low noise) with optional lightx2v LoRAs."""
    node("clip", "CLIPLoader", {"clip_name": CLIP, "type": "wan", "device": "default"})
    node("vae", "VAELoader", {"vae_name": VAE})
    node("unet_hi", "UNETLoader", {"unet_name": UNET_HIGH, "weight_dtype": "default"})
    node("unet_lo", "UNETLoader", {"unet_name": UNET_LOW, "weight_dtype": "default"})

    hi_ref = ["unet_hi", 0]
    lo_ref = ["unet_lo", 0]
    if spec.lightx2v:
        node("lora_hi", "LoraLoaderModelOnly", {
            "model": ["unet_hi", 0], "lora_name": LORA_HIGH, "strength_model": 1.0})
        node("lora_lo", "LoraLoaderModelOnly", {
            "model": ["unet_lo", 0], "lora_name": LORA_LOW, "strength_model": 1.0})
        hi_ref, lo_ref = ["lora_hi", 0], ["lora_lo", 0]

    node("ms_hi", "ModelSamplingSD3", {"model": hi_ref, "shift": WAN_SHIFT})
    node("ms_lo", "ModelSamplingSD3", {"model": lo_ref, "shift": WAN_SHIFT})
    return ["ms_hi", 0], ["ms_lo", 0]


def build_wan_workflow(spec: WanSpec) -> dict:
    """Return a ComfyUI ``/prompt`` graph for a single WAN 2.2 T2V clip."""
    spec = spec.normalized()
    length = wan_frames(spec.seconds, spec.fps)
    steps = 4 if spec.lightx2v else spec.steps
    cfg = 1.0 if spec.lightx2v else spec.cfg
    boundary = max(1, min(steps - 1, round(steps * spec.boundary_fraction)))

    g: dict[str, dict] = {}

    def node(nid, class_type, inputs):
        g[nid] = {"class_type": class_type, "inputs": inputs}
        return nid

    hi_model, lo_model = _experts(g, node, spec)

    node("pos", "CLIPTextEncode", {"text": spec.prompt, "clip": ["clip", 0]})
    node("neg", "CLIPTextEncode", {"text": spec.negative, "clip": ["clip", 0]})

    if spec.start_image_node:
        # Image-to-video continuation (for chaining): seed from a start frame.
        node("i2v", "WanImageToVideo", {
            "positive": ["pos", 0], "negative": ["neg", 0], "vae": ["vae", 0],
            "width": spec.width, "height": spec.height, "length": length,
            "batch_size": 1, "start_image": [spec.start_image_node, 0],
        })
        pos_ref, neg_ref, latent_ref = ["i2v", 0], ["i2v", 1], ["i2v", 2]
    else:
        node("latent", "EmptyHunyuanLatentVideo", {
            "width": spec.width, "height": spec.height,
            "length": length, "batch_size": 1})
        pos_ref, neg_ref, latent_ref = ["pos", 0], ["neg", 0], ["latent", 0]

    # High-noise expert: steps 0..boundary, keep leftover noise for the handoff.
    node("k_hi", "KSamplerAdvanced", {
        "model": hi_model, "add_noise": "enable", "noise_seed": spec.seed,
        "steps": steps, "cfg": cfg, "sampler_name": "euler", "scheduler": "simple",
        "positive": pos_ref, "negative": neg_ref, "latent_image": latent_ref,
        "start_at_step": 0, "end_at_step": boundary,
        "return_with_leftover_noise": "enable",
    })
    # Low-noise expert: finishes steps boundary..steps.
    node("k_lo", "KSamplerAdvanced", {
        "model": lo_model, "add_noise": "disable", "noise_seed": spec.seed,
        "steps": steps, "cfg": cfg, "sampler_name": "euler", "scheduler": "simple",
        "positive": pos_ref, "negative": neg_ref, "latent_image": ["k_hi", 0],
        "start_at_step": boundary, "end_at_step": steps,
        "return_with_leftover_noise": "disable",
    })

    node("decode", "VAEDecode", {"samples": ["k_lo", 0], "vae": ["vae", 0]})
    node("video", "CreateVideo", {"images": ["decode", 0], "fps": float(spec.fps)})
    node("save", "SaveVideo", {
        "video": ["video", 0], "filename_prefix": spec.filename_prefix,
        "format": "auto", "codec": "auto"})
    return g
