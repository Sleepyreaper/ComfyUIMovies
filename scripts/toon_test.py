"""80s-cartoon style + motion gate for the SERPENT EMPIRE trailer.

Proves BEFORE the full build:
  1. cel-shaded 80s-cartoon look (FLUX2 keyframes)
  2. that SHORT clips with RELIABLE motions (camera push / simple action) avoid
     the "snake" warping that killed the fox chase.

Usage:
  PYTHONPATH=. python3 scripts/toon_test.py kf                 # 3 style keyframes
  PYTHONPATH=. python3 scripts/toon_test.py i2v <hero_png>     # 2 motion tests
"""
import json
import os
import sys

from comfymovies.film import _flux_keyframe
from comfymovies.build import CKPT, TEXT_ENCODER
from comfymovies.comfy import ComfyClient

HOST, PORT = "192.168.1.90", 8188
W, H, FPS = 1280, 704, 24
STEPS, CFG = 24, 3.5
OUTDIR = "output/toon_test"

STYLE = ("1980s Saturday-morning cartoon cel animation still, retro American action "
         "cartoon in the style of G.I. Joe and Robotech, bold thick clean ink outlines, "
         "flat cel shading, bright saturated colors, hand-painted background, subtle "
         "film grain, 1985")

KEYS = {
    "hero": (f"{STYLE}. A heroic square-jawed American commando soldier in tan and "
             "green tactical gear, red beret, confident determined expression, "
             "dramatic low-angle hero shot, an American flag patch on the shoulder, "
             "dynamic cinematic composition"),
    "villain": (f"{STYLE}. A menacing warlord of the Serpent Empire, sleek cobra-hooded "
                "helmet, dark green and purple armor with golden snake emblems, glowing "
                "yellow eyes, cruel sneer, throne room with green serpent banners, "
                "dramatic villain composition"),
    "battle": (f"{STYLE}. Dynamic battle, heroic American soldiers firing bright BLUE "
               "energy laser rifles from cover while masked Serpent Empire troopers fire "
               "RED laser bolts back, glowing laser streaks across the frame, bursts of "
               "light, dramatic action, no blood, no injuries"),
}
SEEDS = {"hero": 101, "villain": 202, "battle": 303}

# Motion gate: only reliable motions (camera push-in + a simple rigid action).
MOTIONS = [
    ("push", "slow dramatic camera push-in on the hero soldier, he narrows his eyes "
             "with determination, the cel-animated flag and background hold steady, "
             "limited 1980s cartoon animation, subtle motion, no warping"),
    ("aim", "the hero soldier raises his blue energy laser rifle and takes aim, a "
            "confident sharp movement, bright blue muzzle glow, limited 1980s cartoon "
            "animation, clean and stable, no warping"),
]


def frames_8n1(seconds, fps):
    raw = max(1, round(seconds * fps)); n = max(1, round((raw - 1) / 8)); return n * 8 + 1


def kf_graph(prompt, seed, prefix):
    g = {}
    def node(nid, ct, inp): g[nid] = {"class_type": ct, "inputs": inp}; return nid
    img = _flux_keyframe(node, "kf", prompt, seed, W, H, "")
    node("save", "SaveImage", {"images": img, "filename_prefix": prefix})
    return g


def i2v_graph(kf_name, motion, seconds, seed, prefix):
    length = frames_8n1(seconds, FPS)
    g = {}
    def node(nid, ct, inp): g[nid] = {"class_type": ct, "inputs": inp}; return nid
    node("ckpt", "CheckpointLoaderSimple", {"ckpt_name": CKPT})
    node("clip", "LTXAVTextEncoderLoader",
         {"text_encoder": TEXT_ENCODER, "ckpt_name": CKPT, "device": "default"})
    node("img", "LoadImage", {"image": kf_name})
    node("pos", "CLIPTextEncode", {"text": motion, "clip": ["clip", 0]})
    node("neg", "CLIPTextEncode",
         {"text": "blurry, warped, melting, deformed, distorted, morphing, extra limbs, "
                  "photorealistic, 3d render, live action", "clip": ["clip", 0]})
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
    os.makedirs(OUTDIR, exist_ok=True)
    c = ComfyClient(host=HOST, port=PORT)
    mode = sys.argv[1] if len(sys.argv) > 1 else "kf"
    if mode == "kf":
        for name, prompt in KEYS.items():
            pid = c.submit(kf_graph(prompt, SEEDS[name], f"ComfyUIMovies/toon_{name}"))
            entry = c.wait(pid, timeout=900)
            imgs = c.find_outputs(entry)
            c.download(imgs[0], os.path.join(OUTDIR, f"{name}.png"))
            print(f"keyframe {name} -> {OUTDIR}/{name}.png", flush=True)
    else:
        kf_local = sys.argv[2]
        server = c.upload_image(kf_local, name="toon_hero.png")
        for name, motion in MOTIONS:
            pid = c.submit(i2v_graph(server, motion, 2.0, 501, f"ComfyUIMovies/toon_m_{name}"))
            entry = c.wait(pid, timeout=1200)
            vids = [f for f in c.find_outputs(entry)
                    if f["filename"].lower().endswith((".mp4", ".webm", ".mov"))]
            c.download(vids[0], os.path.join(OUTDIR, f"motion_{name}.mp4"))
            print(f"motion {name} -> {OUTDIR}/motion_{name}.mp4", flush=True)
