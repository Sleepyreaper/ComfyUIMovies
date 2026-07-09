"""Queue every NEON RUN chase cut as an independent LTX-2.3 quality job.

Each cut is its own single-pass text-to-video+audio render (hard cuts, not a
scheduled continuous clip), locked to a shared hero + world description so the
fox and the neon-Tokyo look stay consistent across the whole piece.
"""
import json
from comfymovies.build import MovieSpec, Scene, build_workflow
from comfymovies.comfy import ComfyClient

HOST, PORT = "192.168.1.90", 8188
W, H, FPS = 1280, 704, 24
STORY = json.load(open("prompts/chase.json"))
HERO, WORLD = STORY["hero"], STORY["world"]


def cut_spec(cut: dict) -> MovieSpec:
    prompt = f"{HERO}. {cut['prompt']}. {WORLD}"
    spec = MovieSpec(
        scenes=[Scene(prompt)], width=W, height=H, fps=FPS,
        seconds=cut["seconds"], seed=cut["seed"],
        enhance=True, schedule_scenes=False,
        # quality path: multi-step scheduler, no distill LoRA, higher cfg
        steps=24, cfg=3.5, lora_strength=0.0,
        filename_prefix=f"ComfyUIMovies/chase_cut{cut['id']:02d}",
    )
    return spec


if __name__ == "__main__":
    c = ComfyClient(host=HOST, port=PORT)
    pids = {}
    for cut in STORY["cuts"]:
        pid = c.submit(build_workflow(cut_spec(cut)))
        pids[str(cut["id"])] = pid
        print(f"queued cut{cut['id']:02d} ({cut['seconds']}s) -> {pid}")
    json.dump(pids, open("/tmp/chase_pids.json", "w"))
    print(f"\nqueued {len(pids)} cuts")
