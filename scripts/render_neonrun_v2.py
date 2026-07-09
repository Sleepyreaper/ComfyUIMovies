"""NEON RUN v2 — the keyframe-locked, controlled-camera rebuild.

Per shot in prompts/chase.json:
  1. FLUX2 keyframe (reference-locked to the hero fox via qt_hero.png when
     shot.ref is true) at native 1280x704  -> download -> upload to /input.
  2. LTX-2.3 image-to-video from that keyframe with a CALM/controlled-camera
     motion prompt (action from CONTENT, not frantic camera).
No ffmpeg upscaling — everything is rendered at native res so the fox stays
razor-sharp and pixel-consistent shot to shot.

Usage:
  PYTHONPATH=. python3 scripts/render_neonrun_v2.py            # all shots
  PYTHONPATH=. python3 scripts/render_neonrun_v2.py 2 3 8      # only these ids
Outputs land in output/neon_run_v2/shotNN.mp4 (+ kfNN.png).
"""
import json
import os
import sys
import time

from comfymovies.film import _flux_keyframe
from comfymovies.build import CKPT, TEXT_ENCODER
from comfymovies.comfy import ComfyClient

HOST, PORT = "192.168.1.90", 8188
W, H, FPS = 1280, 704, 24
STEPS, CFG = 24, 3.5
REF_IMAGE = "qt_hero.png"           # locked hero fox, already in ComfyUI /input
STORYBOARD = "prompts/chase.json"
OUTDIR = "output/neon_run_v2"


def frames_8n1(seconds: float, fps: int) -> int:
    raw = max(1, round(seconds * fps))
    n = max(1, round((raw - 1) / 8))
    return n * 8 + 1


def keyframe_graph(prompt: str, seed: int, ref: str, prefix: str) -> dict:
    g = {}
    def node(nid, ct, inp): g[nid] = {"class_type": ct, "inputs": inp}; return nid
    img = _flux_keyframe(node, "kf", prompt, seed, W, H, ref)
    node("save", "SaveImage", {"images": img, "filename_prefix": prefix})
    return g


def i2v_graph(kf_name: str, motion: str, neg: str, seconds: float, seed: int,
              prefix: str) -> dict:
    length = frames_8n1(seconds, FPS)
    g = {}
    def node(nid, ct, inp): g[nid] = {"class_type": ct, "inputs": inp}; return nid
    node("ckpt", "CheckpointLoaderSimple", {"ckpt_name": CKPT})
    node("clip", "LTXAVTextEncoderLoader",
         {"text_encoder": TEXT_ENCODER, "ckpt_name": CKPT, "device": "default"})
    node("img", "LoadImage", {"image": kf_name})
    node("pos", "CLIPTextEncode", {"text": motion, "clip": ["clip", 0]})
    node("neg", "CLIPTextEncode", {"text": neg, "clip": ["clip", 0]})
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


def render_shot(c: ComfyClient, sb: dict, shot: dict) -> str:
    sid = shot["id"]
    neg = sb.get("neg", "")
    ref = REF_IMAGE if shot.get("ref") else ""
    kf_prefix = f"ComfyUIMovies/nrv2_kf{sid:02d}"
    kf_local = os.path.join(OUTDIR, f"kf{sid:02d}.png")
    out_local = os.path.join(OUTDIR, f"shot{sid:02d}.mp4")

    # 1) keyframe
    t0 = time.time()
    print(f"[shot {sid:02d}] keyframe (ref={'yes' if ref else 'no'}) ...", flush=True)
    pid = c.submit(keyframe_graph(shot["keyframe"], shot["seed"], ref, kf_prefix))
    entry = c.wait(pid, timeout=900)
    imgs = c.find_outputs(entry)
    if not imgs:
        raise RuntimeError(f"shot {sid}: no keyframe produced")
    c.download(imgs[0], kf_local)
    server_name = c.upload_image(kf_local, name=f"nrv2_kf{sid:02d}.png")
    print(f"[shot {sid:02d}] keyframe done in {time.time()-t0:.0f}s -> {server_name}", flush=True)

    # 2) image-to-video
    t1 = time.time()
    print(f"[shot {sid:02d}] i2v {shot['seconds']}s ...", flush=True)
    pid = c.submit(i2v_graph(server_name, shot["motion"], neg, shot["seconds"],
                             shot["seed"] + 1000, f"ComfyUIMovies/nrv2_shot{sid:02d}"))
    entry = c.wait(pid, timeout=1800)
    vids = c.find_outputs(entry)
    vids = [f for f in vids if f["filename"].lower().endswith((".mp4", ".webm", ".mov"))] or vids
    if not vids:
        raise RuntimeError(f"shot {sid}: no video produced")
    c.download(vids[0], out_local)
    print(f"[shot {sid:02d}] i2v done in {time.time()-t1:.0f}s -> {out_local}", flush=True)
    return out_local


if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    sb = json.load(open(STORYBOARD))
    only = {int(a) for a in sys.argv[1:]} if len(sys.argv) > 1 else None
    shots = [s for s in sb["shots"] if only is None or s["id"] in only]
    c = ComfyClient(host=HOST, port=PORT)
    done = []
    for shot in shots:
        try:
            done.append(render_shot(c, sb, shot))
        except Exception as e:
            print(f"[shot {shot['id']:02d}] FAILED: {e}", flush=True)
    print("DONE:", done, flush=True)
