"""EAGLE FORCE vs THE SERPENT EMPIRE — 80s cartoon trailer renderer.

Pass 1: render the HERO and VILLAIN keyframes (no ref) and upload them to
        /input as toon_hero.png / toon_villain.png.
Pass 2: per shot -> FLUX2 keyframe (reference-locked to hero/villain when
        shot.ref is set) at native 1280x704 -> short LTX i2v with a RELIABLE
        motion (camera push / simple action). No ffmpeg upscale.

Lasers, flashes, title cards and grain are added later in build_trailer.py.

Usage:
  PYTHONPATH=. python3 scripts/render_serpent.py            # everything
  PYTHONPATH=. python3 scripts/render_serpent.py refs       # just hero+villain
  PYTHONPATH=. python3 scripts/render_serpent.py 5 6 10     # only these shot ids
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
STORYBOARD = "prompts/serpent.json"
OUTDIR = "output/serpent"


def frames_8n1(seconds, fps):
    raw = max(1, round(seconds * fps)); n = max(1, round((raw - 1) / 8)); return n * 8 + 1


def kf_graph(prompt, seed, ref, prefix):
    g = {}
    def node(nid, ct, inp): g[nid] = {"class_type": ct, "inputs": inp}; return nid
    img = _flux_keyframe(node, "kf", prompt, seed, W, H, ref)
    node("save", "SaveImage", {"images": img, "filename_prefix": prefix})
    return g


def i2v_graph(kf_name, motion, neg, seconds, seed, prefix):
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


def sub_style(text, style):
    return text.replace("STYLE", style)


def render_ref(c, sb, name):
    """Render a character reference keyframe and upload it to /input."""
    style = sb["style"]
    seeds = {"hero": 101, "villain": 202}
    prompts = {
        "hero": (f"{style}. A heroic square-jawed American commando soldier in tan and "
                 "green tactical gear, red beret, confident determined expression, "
                 "clean neutral hero portrait, dynamic cinematic composition"),
        "villain": (f"{style}. A menacing warlord of the Serpent Empire, sleek cobra-hooded "
                    "helmet, dark green and purple armor with golden snake emblems, glowing "
                    "yellow eyes, cruel sneer, clean neutral villain portrait"),
    }
    prefix = f"ComfyUIMovies/serpent_ref_{name}"
    local = os.path.join(OUTDIR, f"ref_{name}.png")
    print(f"[ref {name}] rendering ...", flush=True)
    pid = c.submit(kf_graph(prompts[name], seeds[name], "", prefix))
    entry = c.wait(pid, timeout=900)
    c.download(c.find_outputs(entry)[0], local)
    server = c.upload_image(local, name=sb["refs"][name])
    print(f"[ref {name}] -> {server}", flush=True)
    return server


def render_shot(c, sb, shot, takes=1):
    sid = shot["id"]; style = sb["style"]; neg = sb.get("neg", "")
    ref = sb["refs"].get(shot["ref"], "") if shot.get("ref") else ""
    kf_prefix = f"ComfyUIMovies/serpent_kf{sid:02d}"
    kf_local = os.path.join(OUTDIR, f"kf{sid:02d}.png")
    out_local = os.path.join(OUTDIR, f"shot{sid:02d}.mp4")

    t0 = time.time()
    print(f"[shot {sid:02d}] keyframe (ref={shot.get('ref') or 'none'}) ...", flush=True)
    pid = c.submit(kf_graph(sub_style(shot["keyframe"], style), shot["seed"], ref, kf_prefix))
    entry = c.wait(pid, timeout=900)
    c.download(c.find_outputs(entry)[0], kf_local)
    server = c.upload_image(kf_local, name=f"serpent_kf{sid:02d}.png")
    print(f"[shot {sid:02d}] keyframe {time.time()-t0:.0f}s", flush=True)

    # best-of-N: render `takes` i2v variants (different noise seeds), auto-pick
    # the highest-scoring (healthy motion + sharp, no static/warp).
    motion = sub_style(shot["motion"], style)
    take_paths = []
    for k in range(takes):
        t1 = time.time()
        seed = shot["seed"] + 5000 + k * 137
        pid = c.submit(i2v_graph(server, motion, neg, shot["seconds"], seed,
                                 f"ComfyUIMovies/serpent_shot{sid:02d}_t{k}"))
        entry = c.wait(pid, timeout=1200)
        vids = [f for f in c.find_outputs(entry)
                if f["filename"].lower().endswith((".mp4", ".webm", ".mov"))]
        tp = os.path.join(OUTDIR, f"shot{sid:02d}_t{k}.mp4")
        c.download(vids[0], tp)
        take_paths.append(tp)
        print(f"[shot {sid:02d}] take {k} {time.time()-t1:.0f}s", flush=True)

    if takes == 1:
        os.replace(take_paths[0], out_local)
        print(f"[shot {sid:02d}] -> {out_local}", flush=True)
        return out_local

    try:
        from comfymovies.takes import pick_best
        best, bscore, allscores = pick_best(take_paths)
        for i, s in enumerate(allscores):
            print(f"    take {i}: motion={s.motion} sharp={s.sharpness} "
                  f"jerk={s.jerk} score={s.score} [{s.verdict}]", flush=True)
        os.replace(best, out_local)
        for p in take_paths:
            if os.path.exists(p) and os.path.abspath(p) != os.path.abspath(out_local):
                os.remove(p)
        print(f"[shot {sid:02d}] BEST -> {out_local} (score {bscore.score} {bscore.verdict})",
              flush=True)
    except Exception as e:
        os.replace(take_paths[0], out_local)
        print(f"[shot {sid:02d}] scoring failed ({e}); kept take 0", flush=True)
    return out_local


if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    sb = json.load(open(STORYBOARD))
    c = ComfyClient(host=HOST, port=PORT)
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"

    if arg in ("refs", "all"):
        render_ref(c, sb, "hero")
        render_ref(c, sb, "villain")
    if arg == "refs":
        sys.exit(0)

    only = {int(a) for a in sys.argv[1:]} if arg not in ("all", "refs") else None
    shots = [s for s in sb["shots"] if only is None or s["id"] in only]
    takes = int(os.environ.get("TAKES", "1"))
    print(f"rendering {len(shots)} shot(s), {takes} take(s) each", flush=True)
    done = []
    for shot in shots:
        try:
            done.append(render_shot(c, sb, shot, takes=takes))
        except Exception as e:
            print(f"[shot {shot['id']:02d}] FAILED: {e}", flush=True)
    print("DONE:", done, flush=True)
