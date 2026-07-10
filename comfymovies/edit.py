"""FLUX.2 dev — instruction-based image editing graph.

FLUX.2 dev (installed on the box) is a proper *instruction* editor: give it an
image plus a plain-language instruction ("make the background Prague", "turn her
into an anime character", "make this person a gummy bear", "add sunglasses") and
it edits *only* what you ask while preserving everything else. The edit works by
feeding the source image into the conditioning through ``ReferenceLatent`` (the
FLUX.2 "Kontext" mechanism) *and* reusing the VAE-encoded source as the sampler
latent, so the output keeps the source composition/size and changes just the
instructed part.

NOTE: the on-box ``qwen_image_2512`` UNET is the BASE text-to-image Qwen model,
NOT Qwen-Image-Edit — it has no edit weights and just reconstructs the input.
FLUX.2 dev is the correct (and best-in-class) local instruction editor; no extra
download is required.

Two quality modes:
  * ``fast``  — Turbo LoRA, 8 steps, guidance 3.0  (~seconds, great for iterating)
  * ``high``  — no LoRA, 28 steps, guidance 4.0     (max fidelity for the keeper)

Up to three reference images can be supplied (``images[0..2]``): the first is the
image being edited; extras act as content references (chained ReferenceLatents),
e.g. "put the character from the second image into this scene".

Assets (all confirmed installed via /object_info):
  UNET  flux2_dev_fp8mixed.safetensors
  LoRA  Flux_2-Turbo-LoRA_comfyui.safetensors   (fast mode only)
  CLIP  mistral_3_small_flux2_bf16.safetensors   (type ``flux2``)
  VAE   flux2-vae.safetensors
"""
from __future__ import annotations

UNET = "flux2_dev_fp8mixed.safetensors"
TURBO_LORA = "Flux_2-Turbo-LoRA_comfyui.safetensors"
CLIP = "mistral_3_small_flux2_bf16.safetensors"
VAE = "flux2-vae.safetensors"

PRESETS = {
    "fast": {"steps": 8, "guidance": 3.0, "lora": True},
    "high": {"steps": 28, "guidance": 4.0, "lora": False},
}


def build_edit_workflow(
    images: list[str],
    instruction: str,
    width: int,
    height: int,
    *,
    quality: str = "fast",
    seed: int = 0,
    guidance: float | None = None,
    prefix: str = "ComfyUIMovies/edit",
) -> dict:
    """Build a FLUX.2 instruction-edit graph.

    ``images`` are filenames already present in ComfyUI's ``/input`` (the first
    is the edit target; up to two more are extra references). ``instruction`` is
    the plain-language edit. ``width``/``height`` come from the source image (read
    locally with PIL) and drive FLUX.2's resolution-dependent sigma shift.
    Returns a prompt-API graph dict.
    """
    if not images:
        raise ValueError("at least one input image is required")
    if quality not in PRESETS:
        raise ValueError(f"quality must be one of {list(PRESETS)}")
    p = PRESETS[quality]
    guid = p["guidance"] if guidance is None else guidance
    imgs = images[:3]

    g: dict[str, dict] = {}

    def node(nid, ct, inp):
        g[nid] = {"class_type": ct, "inputs": inp}
        return nid

    # --- model path (UNET -> optional Turbo LoRA) ---------------------------
    node("unet", "UNETLoader", {"unet_name": UNET, "weight_dtype": "default"})
    model = ["unet", 0]
    if p["lora"]:
        node("lora", "LoraLoaderModelOnly",
             {"model": model, "lora_name": TURBO_LORA, "strength_model": 1.0})
        model = ["lora", 0]

    node("clip", "CLIPLoader",
         {"clip_name": CLIP, "type": "flux2", "device": "default"})
    node("vae", "VAELoader", {"vae_name": VAE})

    # --- load each input, Kontext-scale, VAE-encode -------------------------
    # The first image's encoded latent is BOTH the first reference and the
    # sampler canvas (so the output preserves the source composition + size).
    reflat = None
    encoded = []
    for i, name in enumerate(imgs):
        node(f"img{i}", "LoadImage", {"image": name})
        node(f"scale{i}", "FluxKontextImageScale", {"image": [f"img{i}", 0]})
        node(f"enc{i}", "VAEEncode",
             {"pixels": [f"scale{i}", 0], "vae": ["vae", 0]})
        encoded.append([f"enc{i}", 0])
    reflat = encoded[0]

    # --- conditioning: instruction + chained image references ---------------
    node("pos", "CLIPTextEncode", {"text": instruction, "clip": ["clip", 0]})
    cond = ["pos", 0]
    for i, lat in enumerate(encoded):
        node(f"ref{i}", "ReferenceLatent", {"conditioning": cond, "latent": lat})
        cond = [f"ref{i}", 0]
    node("guid", "FluxGuidance", {"conditioning": cond, "guidance": guid})
    node("guider", "BasicGuider", {"model": model, "conditioning": ["guid", 0]})

    # --- sampler (canvas = VAE-encoded source) ------------------------------
    node("noise", "RandomNoise", {"noise_seed": seed})
    node("sched", "Flux2Scheduler",
         {"steps": p["steps"], "width": width, "height": height})
    node("sampler", "KSamplerSelect", {"sampler_name": "euler"})
    node("sample", "SamplerCustomAdvanced", {
        "noise": ["noise", 0], "guider": ["guider", 0],
        "sampler": ["sampler", 0], "sigmas": ["sched", 0],
        "latent_image": reflat})

    node("dec", "VAEDecode", {"samples": ["sample", 0], "vae": ["vae", 0]})
    node("save", "SaveImage", {"images": ["dec", 0], "filename_prefix": prefix})
    return g


# ===========================================================================
# Qwen-Image-Edit 2511 backend
# ===========================================================================
# This mirrors the official ComfyUI template
# ``template_qwen_image_edit_2511_systms_action`` (the one the user downloaded).
# Unlike the *base* ``qwen_image_2512`` UNET (which has no edit weights and just
# reconstructs the input), ``qwen_image_edit_2511`` is a true instruction
# editor: the Qwen2.5-VL text encoder actually looks at the image and edits only
# what the instruction asks. The template also stacks a "systms_action" LoRA
# (``QWEN_EDIT_ACTION_V1``) that injects dynamic action/motion into the scene.
#
# Assets (from the template graph):
#   UNET  qwen_image_edit_2511_bf16.safetensors      (~40GB; download the model)
#   CLIP  qwen_2.5_vl_7b_fp8_scaled.safetensors      (type ``qwen_image``)
#   VAE   qwen_image_vae.safetensors
#   LoRA  QWEN_EDIT_ACTION_V1.safetensors            (always on; the "action")
#   LoRA  Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors (fast only)
#
# Two quality modes (the template toggles these with a boolean switch):
#   fast — Lightning-4step LoRA on,  steps 4,  cfg 1   (seconds; great to iterate)
#   high — Lightning off,            steps 40, cfg 4   (max fidelity for keepers)
QWEN_EDIT_UNET = "qwen_image_edit_2511_bf16.safetensors"
QWEN_EDIT_CLIP = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
QWEN_EDIT_VAE = "qwen_image_vae.safetensors"
QWEN_ACTION_LORA = "QWEN_EDIT_ACTION_V1.safetensors"
QWEN_LIGHTNING_LORA = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
QWEN_SHIFT = 3.1  # ModelSamplingAuraFlow shift, per the template

QWEN_PRESETS = {
    "fast": {"steps": 4, "cfg": 1.0, "lightning": True},
    "high": {"steps": 40, "cfg": 4.0, "lightning": False},
}


def build_qwen_edit_workflow(
    images: list[str],
    instruction: str,
    *,
    quality: str = "fast",
    seed: int = 0,
    action_strength: float = 1.0,
    prefix: str = "ComfyUIMovies/qwenedit",
) -> dict:
    """Build a Qwen-Image-Edit 2511 instruction-edit graph (the downloaded
    ``systms_action`` template, generalized to take any input + instruction).

    ``images`` are filenames already present in ComfyUI's ``/input`` (the first
    is the edit target; up to two more are extra references passed to
    ``TextEncodeQwenImageEditPlus`` image2/image3). ``instruction`` is the
    plain-language edit. ``action_strength`` scales the QWEN_EDIT_ACTION_V1 LoRA
    (set 0 to effectively disable the action bias for static edits).
    Returns a prompt-API graph dict.
    """
    if not images:
        raise ValueError("at least one input image is required")
    if quality not in QWEN_PRESETS:
        raise ValueError(f"quality must be one of {list(QWEN_PRESETS)}")
    p = QWEN_PRESETS[quality]
    imgs = images[:3]

    g: dict[str, dict] = {}

    def node(nid, ct, inp):
        g[nid] = {"class_type": ct, "inputs": inp}
        return nid

    # --- model chain: UNET -> ACTION lora -> AuraFlow shift -> CFGNorm -------
    node("unet", "UNETLoader", {"unet_name": QWEN_EDIT_UNET, "weight_dtype": "default"})
    model = ["unet", 0]
    if action_strength:
        node("action", "LoraLoaderModelOnly",
             {"model": model, "lora_name": QWEN_ACTION_LORA,
              "strength_model": action_strength})
        model = ["action", 0]
    node("ms", "ModelSamplingAuraFlow", {"model": model, "shift": QWEN_SHIFT})
    node("cfgn", "CFGNorm", {"model": ["ms", 0], "strength": 1.0})
    model = ["cfgn", 0]
    if p["lightning"]:
        node("light", "LoraLoaderModelOnly",
             {"model": model, "lora_name": QWEN_LIGHTNING_LORA,
              "strength_model": 1.0})
        model = ["light", 0]

    node("clip", "CLIPLoader",
         {"clip_name": QWEN_EDIT_CLIP, "type": "qwen_image", "device": "default"})
    node("vae", "VAELoader", {"vae_name": QWEN_EDIT_VAE})

    # --- load images; scale the edit target (image1) via Kontext scale ------
    node("img0", "LoadImage", {"image": imgs[0]})
    node("scale", "FluxKontextImageScale", {"image": ["img0", 0]})
    edit_img = ["scale", 0]
    extra = {}
    for i, name in enumerate(imgs[1:], start=2):
        node(f"img{i}", "LoadImage", {"image": name})
        extra[f"image{i}"] = [f"img{i}", 0]

    # --- positive/negative conditioning (Qwen2.5-VL sees the image) ---------
    pos_in = {"clip": ["clip", 0], "vae": ["vae", 0], "prompt": instruction,
              "image1": edit_img}
    pos_in.update(extra)
    node("pos", "TextEncodeQwenImageEditPlus", pos_in)
    node("posr", "FluxKontextMultiReferenceLatentMethod",
         {"conditioning": ["pos", 0], "reference_latents_method": "index_timestep_zero"})

    neg_in = {"clip": ["clip", 0], "vae": ["vae", 0], "prompt": "",
              "image1": edit_img}
    neg_in.update(extra)
    node("neg", "TextEncodeQwenImageEditPlus", neg_in)
    node("negr", "FluxKontextMultiReferenceLatentMethod",
         {"conditioning": ["neg", 0], "reference_latents_method": "index_timestep_zero"})

    # --- latent canvas = VAE-encoded edit target ----------------------------
    node("lat", "VAEEncode", {"pixels": edit_img, "vae": ["vae", 0]})

    node("ks", "KSampler", {
        "model": model, "positive": ["posr", 0], "negative": ["negr", 0],
        "latent_image": ["lat", 0], "seed": seed, "steps": p["steps"],
        "cfg": p["cfg"], "sampler_name": "euler", "scheduler": "simple",
        "denoise": 1.0})

    node("dec", "VAEDecode", {"samples": ["ks", 0], "vae": ["vae", 0]})
    node("save", "SaveImage", {"images": ["dec", 0], "filename_prefix": prefix})
    return g
