"""Build parameterized LTX-2.3 text-to-video (+audio) workflows in API format.

Rather than shipping the heavyweight two-pass UI template (which relies on a
spatial-upscaler model that isn't installed and an LLM prompt-enhancer with a
compound dynamic widget), we assemble a clean single-pass joint audio+video
graph directly from the verified node schemas. Every knob the movie pipeline
needs — prompt(s), resolution, length, fps, seed, seamless context windows — is
a first-class parameter here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Model assets confirmed present on the target ComfyUI (RTX 5090 box).
CKPT = "ltx-2.3-22b-dev-fp8.safetensors"
TEXT_ENCODER = "gemma_3_12B_it_fp4_mixed.safetensors"
DISTILLED_LORA = "ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors"
# Few-step sigma schedule for the distilled base pass (from the official template).
BASE_SIGMAS = "0.85, 0.7250, 0.4219, 0.0"
DEFAULT_NEGATIVE = (
    "pc game, console game, video game, cartoon childish, ugly, deformed, "
    "watermark, text overlay, low quality, blurry, jpeg artifacts"
)
# LTXVEmptyLatentAudio caps frames_number at 1000, so native joint audio is only
# available for clips at/under this many frames (~40s at 24fps).
AUDIO_MAX_FRAMES = 1000


def snap(value: int, multiple: int) -> int:
    """Round ``value`` to the nearest positive multiple of ``multiple``."""
    v = max(multiple, round(value / multiple) * multiple)
    return int(v)


def frames_for(seconds: float, fps: int) -> int:
    """Frame count for a duration, snapped to LTX's ``8n + 1`` requirement."""
    raw = max(1, round(seconds * fps))
    n = max(1, round((raw - 1) / 8))
    return n * 8 + 1


@dataclass
class Scene:
    """One narrative beat. ``weight`` controls its share of total frames."""
    prompt: str
    weight: float = 1.0


@dataclass
class MovieSpec:
    """Everything needed to render one continuous clip."""
    scenes: list[Scene]
    width: int = 896
    height: int = 512
    fps: int = 24
    seconds: float = 5.0
    seed: int = 0
    negative: str = DEFAULT_NEGATIVE
    lora_strength: float = 0.5
    cfg: float = 1.0
    # Denoising steps. 0 = fast distilled path (fixed few-step ManualSigmas).
    # > 0 = quality path via LTXVScheduler (typically with lora_strength=0 and
    # a higher cfg for sharper, more prompt-faithful output).
    steps: int = 0
    # Seamless long-form controls (context windows). Enabled automatically for
    # clips longer than ``context_length`` frames unless forced off.
    use_context_windows: bool | None = None
    context_length: int = 145      # 8n+1
    context_overlap: int = 40      # 8n
    # Temporal scene scheduling: assign each scene its own slice of the timeline
    # so beats flow into one another. Auto-on when there is more than one scene.
    schedule_scenes: bool | None = None
    # On-box prompt enhancement via the ComfyUI gemma LLM (TextGenerateLTX2Prompt).
    # Keeps everything on the single 5090 workflow — no external LLM needed.
    enhance: bool = True
    enhance_max_length: int = 512
    # Native LTX-2 joint audio. None = auto (on for short clips, off for long
    # ones that exceed the audio latent's 1000-frame cap).
    audio: bool | None = None
    filename_prefix: str = "ComfyUIMovies/movie"
    extra_negative: str = ""

    def normalized(self) -> "MovieSpec":
        self.width = snap(self.width, 64)
        self.height = snap(self.height, 64)
        return self


def _combined_prompt(scenes: list[Scene]) -> str:
    """Single-prompt fallback: join beats into one flowing description."""
    return " ".join(s.prompt.strip() for s in scenes if s.prompt.strip())


def _scene_spans(scenes: list[Scene]) -> list[tuple[float, float]]:
    """Cumulative [start, end) timeline fractions for each scene by weight."""
    weights = [max(0.0, s.weight) for s in scenes]
    total = sum(weights) or 1.0
    spans: list[tuple[float, float]] = []
    cursor = 0.0
    for w in weights:
        span = w / total
        spans.append((cursor, min(1.0, cursor + span)))
        cursor += span
    return spans


def _encode_scene(node, key: str, text: str, spec: "MovieSpec") -> list:
    """Encode one scene's text to CONDITIONING, optionally enhancing on-box.

    When ``spec.enhance`` is set, the raw text is first rewritten by the ComfyUI
    gemma LLM (``TextGenerateLTX2Prompt``) so the whole thing stays on the single
    5090 workflow — no external prompt service required.
    """
    clip_text_ref: object = text
    if spec.enhance:
        gen = node(f"enh_{key}", "TextGenerateLTX2Prompt", {
            "clip": ["clip", 0],
            "prompt": text,
            "max_length": spec.enhance_max_length,
            "sampling_mode": "off",  # deterministic; "on" needs namespaced fields
            "thinking": False,
            "use_default_template": True,
        })
        clip_text_ref = [gen, 0]
    enc = node(f"enc_{key}", "CLIPTextEncode", {
        "text": clip_text_ref, "clip": ["clip", 0],
    })
    return [enc, 0]


def _build_positive(node, scenes: list[Scene], schedule: bool,
                    spec: "MovieSpec") -> list:
    """Return a CONDITIONING ref for the positive prompt.

    When ``schedule`` is set and there are multiple scenes, each scene is encoded
    and pinned to its slice of the video timeline via
    ``ConditioningSetAreaPercentageVideo`` (z = start, temporal = span), then all
    slices are combined so the beats play back-to-back in one seamless clip.
    """
    scenes = [s for s in scenes if s.prompt.strip()] or [Scene("a cinematic scene")]

    if not schedule or len(scenes) == 1:
        text = _combined_prompt(scenes)
        return _encode_scene(node, "single", text, spec)

    spans = _scene_spans(scenes)
    region_refs: list[list] = []
    for i, (scene, (start, end)) in enumerate(zip(scenes, spans)):
        enc_ref = _encode_scene(node, str(i), scene.prompt.strip(), spec)
        region = node(f"region{i}", "ConditioningSetAreaPercentageVideo", {
            "conditioning": enc_ref,
            "width": 1.0, "height": 1.0,
            "temporal": round(max(1e-3, end - start), 6),
            "x": 0.0, "y": 0.0, "z": round(start, 6),
            "strength": 1.0,
        })
        region_refs.append([region, 0])

    combined = region_refs[0]
    for i, ref in enumerate(region_refs[1:], start=1):
        cid = node(f"combine{i}", "ConditioningCombine", {
            "conditioning_1": combined, "conditioning_2": ref,
        })
        combined = [cid, 0]
    return combined


def build_workflow(spec: MovieSpec) -> dict:
    """Return a ComfyUI ``/prompt`` graph for the given :class:`MovieSpec`.

    Uses a joint audio+video latent so LTX-2 generates a synced soundtrack in
    the same pass. For multi-scene specs the beats are concatenated into one
    continuous prompt (seamless by construction); when context windows are
    enabled the model is wrapped so long clips render without seams.
    """
    spec = spec.normalized()
    length = frames_for(spec.seconds, spec.fps)
    negative = (spec.negative + " " + spec.extra_negative).strip()

    schedule = spec.schedule_scenes
    if schedule is None:
        schedule = len(spec.scenes) > 1

    want_ctx = spec.use_context_windows
    if want_ctx is None:
        want_ctx = length > spec.context_length

    # Per-window temporal conditioning (ConditioningSetAreaPercentageVideo +
    # split_conds_to_windows) is incompatible with LTXVContextWindows on this
    # build (triggers a negative-length narrow()). For long clips we instead run
    # "continuous mode": one combined, gemma-enhanced prompt whose narrative arc
    # evolves across the context windows. Scene scheduling stays available for
    # short (non-windowed) clips.
    if want_ctx:
        schedule = False

    g: dict[str, dict] = {}

    def node(nid: str, class_type: str, inputs: dict) -> str:
        g[nid] = {"class_type": class_type, "inputs": inputs}
        return nid

    node("ckpt", "CheckpointLoaderSimple", {"ckpt_name": CKPT})
    model_ref: list = ["ckpt", 0]

    if spec.lora_strength and spec.lora_strength > 0:
        node("lora", "LoraLoaderModelOnly", {
            "model": ["ckpt", 0],
            "lora_name": DISTILLED_LORA,
            "strength_model": spec.lora_strength,
        })
        model_ref = ["lora", 0]

    if want_ctx:
        node("ctx", "LTXVContextWindows", {
            "model": model_ref,
            "context_length": _ensure_8n1(spec.context_length),
            "context_overlap": snap(spec.context_overlap, 8),
            "context_schedule": "standard_uniform",
            "context_stride": 1,
            "closed_loop": False,
            "fuse_method": "pyramid",
            "freenoise": True,
            "retain_first_frame": False,
            # Route each scheduled scene's conditioning to its own windows.
            "split_conds_to_windows": bool(schedule),
        })
        model_ref = ["ctx", 0]

    node("clip", "LTXAVTextEncoderLoader", {
        "text_encoder": TEXT_ENCODER,
        "ckpt_name": CKPT,
        "device": "default",
    })

    positive_ref = _build_positive(node, spec.scenes, schedule, spec)
    node("neg", "CLIPTextEncode", {"text": negative, "clip": ["clip", 0]})
    node("cond", "LTXVConditioning", {
        "positive": positive_ref, "negative": ["neg", 0],
        "frame_rate": float(spec.fps),
    })

    node("vlatent", "EmptyLTXVLatentVideo", {
        "width": spec.width, "height": spec.height,
        "length": length, "batch_size": 1,
    })

    # Native LTX-2 joint audio is only valid up to AUDIO_MAX_FRAMES; longer clips
    # render video-only (seamless via context windows) and are scored separately
    # (e.g. ElevenLabs). Auto-gate unless the caller forces it.
    with_audio = spec.audio
    if with_audio is None:
        with_audio = length <= AUDIO_MAX_FRAMES

    if with_audio:
        node("avae", "LTXVAudioVAELoader", {"ckpt_name": CKPT})
        node("alatent", "LTXVEmptyLatentAudio", {
            "frames_number": length, "frame_rate": spec.fps,
            "batch_size": 1, "audio_vae": ["avae", 0],
        })
        node("av", "LTXVConcatAVLatent", {
            "video_latent": ["vlatent", 0], "audio_latent": ["alatent", 0],
        })
        latent_ref: list = ["av", 0]
    else:
        latent_ref = ["vlatent", 0]

    node("noise", "RandomNoise", {"noise_seed": spec.seed})
    node("sampler", "KSamplerSelect", {"sampler_name": "euler"})

    # Sigma schedule: the distilled few-step path (fast, softer) uses a fixed
    # ManualSigmas; quality mode (steps > 0, typically with the LoRA disabled)
    # derives a proper multi-step schedule from LTXVScheduler for sharper output.
    if spec.steps and spec.steps > 0:
        node("sigmas", "LTXVScheduler", {
            "steps": spec.steps, "max_shift": 2.05, "base_shift": 0.95,
            "stretch": True, "terminal": 0.1, "latent": latent_ref,
        })
    else:
        node("sigmas", "ManualSigmas", {"sigmas": BASE_SIGMAS})

    node("guider", "CFGGuider", {
        "model": model_ref, "positive": ["cond", 0],
        "negative": ["cond", 1], "cfg": spec.cfg,
    })
    node("sample", "SamplerCustomAdvanced", {
        "noise": ["noise", 0], "guider": ["guider", 0],
        "sampler": ["sampler", 0], "sigmas": ["sigmas", 0],
        "latent_image": latent_ref,
    })

    if with_audio:
        node("split", "LTXVSeparateAVLatent", {"av_latent": ["sample", 0]})
        node("vdec", "VAEDecode", {"samples": ["split", 0], "vae": ["ckpt", 2]})
        node("adec", "LTXVAudioVAEDecode", {
            "samples": ["split", 1], "audio_vae": ["avae", 0],
        })
        node("video", "CreateVideo", {
            "images": ["vdec", 0], "audio": ["adec", 0], "fps": float(spec.fps),
        })
    else:
        node("vdec", "VAEDecode", {"samples": ["sample", 0], "vae": ["ckpt", 2]})
        node("video", "CreateVideo", {
            "images": ["vdec", 0], "fps": float(spec.fps),
        })

    node("save", "SaveVideo", {
        "video": ["video", 0], "filename_prefix": spec.filename_prefix,
        "format": "auto", "codec": "auto",
    })
    return g


def _ensure_8n1(value: int) -> int:
    n = max(1, round((value - 1) / 8))
    return n * 8 + 1
