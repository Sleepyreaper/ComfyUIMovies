"""Quality proof: FLUX2 keyframe -> LTX-2.3 image-to-video.

Fixes the two NEON RUN problems in one graph:
  * consistency  -> every cut starts from the SAME crisp FLUX2 hero keyframe
  * blur         -> calm/controlled camera prompts + native res, no upscale

Renders one shared hero keyframe, then a few short i2v cuts from it. All on the
single 5090, native 1280x704, no ffmpeg upscaling afterwards.
"""
import json, sys
from comfymovies.film import _flux_keyframe
from comfymovies.build import CKPT, TEXT_ENCODER, DISTILLED_LORA, BASE_SIGMAS, DEFAULT_NEGATIVE
from comfymovies.wan import wan_frames  # 8n+1-ish helper not used; keep simple
from comfymovies.comfy import ComfyClient

HOST, PORT = "192.168.1.90", 8188
W, H, FPS = 1280, 704, 24

HERO_KF_PROMPT = (
    "A sleek wild red fox with a bright white-tipped tail and amber eyes, standing "
    "alert in a narrow rain-soaked neon Tokyo alley at night, glowing pink and blue "
    "signage, wet reflective pavement, drifting steam, cinematic anamorphic "
    "photograph, shallow depth of field, sharp crisp focus, highly detailed fur, "
    "moody volumetric neon light, film still, 35mm"
)

# Calm, controlled motion prompts (no whip-pans, no 'full speed').
CUTS = [
    ("head", "the red fox slowly turns its head and flicks its ears, blinking, "
             "gentle rain falling, faint steam drifting, the camera holds steady, "
             "very subtle slow push-in, calm and crisp"),
    ("walk", "the red fox takes a few slow deliberate steps forward toward the "
             "camera, paws placing gently on the wet pavement, tail low, a smooth "
             "steady slow dolly, controlled cinematic motion, sharp focus"),
]

STEPS, CFG = 24, 3.5


def frames_8n1(seconds: float, fps: int) -> int:
    raw = max(1, round(seconds * fps))
    n = max(1, round((raw - 1) / 8))
    return n * 8 + 1


def keyframe_graph(seed: int, prefix: str) -> dict:
    g = {}
    def node(nid, ct, inp): g[nid] = {"class_type": ct, "inputs": inp}; return nid
    img = _flux_keyframe(node, "hero", HERO_KF_PROMPT, seed, W, H, "")
    node("save", "SaveImage", {"images": img, "filename_prefix": prefix})
    return g


def i2v_graph(kf_image_name: str, motion: str, seconds: float, seed: int, prefix: str) -> dict:
    length = frames_8n1(seconds, FPS)
    g = {}
    def node(nid, ct, inp): g[nid] = {"class_type": ct, "inputs": inp}; return nid
    node("ckpt", "CheckpointLoaderSimple", {"ckpt_name": CKPT})
    node("clip", "LTXAVTextEncoderLoader",
         {"text_encoder": TEXT_ENCODER, "ckpt_name": CKPT, "device": "default"})
    node("img", "LoadImage", {"image": kf_image_name})
    node("pos", "CLIPTextEncode", {"text": motion, "clip": ["clip", 0]})
    node("neg", "CLIPTextEncode", {"text": DEFAULT_NEGATIVE, "clip": ["clip", 0]})
    node("cond", "LTXVConditioning",
         {"positive": ["pos", 0], "negative": ["neg", 0], "frame_rate": float(FPS)})
    node("i2v", "LTXVImgToVideo", {
        "positive": ["cond", 0], "negative": ["cond", 1], "vae": ["ckpt", 2],
        "image": ["img", 0], "width": W, "height": H, "length": length,
        "batch_size": 1, "strength": 1.0})
    node("noise", "RandomNoise", {"noise_seed": seed})
    node("sig", "LTXVScheduler", {"steps": STEPS, "max_shift": 2.05, "base_shift": 0.95,
                                  "stretch": True, "terminal": 0.1, "latent": ["i2v", 2]})
    node("sampler", "KSamplerSelect", {"sampler_name": "euler"})
    node("guider", "CFGGuider",
         {"model": ["ckpt", 0], "positive": ["i2v", 0], "negative": ["i2v", 1], "cfg": CFG})
    node("samp", "SamplerCustomAdvanced", {
        "noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0],
        "sigmas": ["sig", 0], "latent_image": ["i2v", 2]})
    node("dec", "VAEDecode", {"samples": ["samp", 0], "vae": ["ckpt", 2]})
    node("vid", "CreateVideo", {"images": ["dec", 0], "fps": float(FPS)})
    node("save", "SaveVideo", {"video": ["vid", 0], "filename_prefix": prefix,
                               "format": "auto", "codec": "auto"})
    return g


if __name__ == "__main__":
    c = ComfyClient(host=HOST, port=PORT)
    mode = sys.argv[1] if len(sys.argv) > 1 else "kf"
    if mode == "kf":
        pid = c.submit(keyframe_graph(seed=7, prefix="ComfyUIMovies/qt_hero"))
        json.dump({"kf": pid}, open("/tmp/qt_kf_pid.json", "w"))
        print("queued hero keyframe ->", pid)
    else:
        kf_name = sys.argv[2]  # filename of the downloaded+uploaded hero keyframe in /input
        pids = {}
        for name, motion in CUTS:
            pid = c.submit(i2v_graph(kf_name, motion, 3.0, 21, f"ComfyUIMovies/qt_{name}"))
            pids[name] = pid
            print(f"queued i2v {name} ->", pid)
        json.dump(pids, open("/tmp/qt_i2v_pids.json", "w"))
