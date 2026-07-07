"""Command-line entry point: one line of text -> a seamless movie.

Examples
--------
    python -m comfymovies "GI Joe style 1980s cartoon, American soldiers fight \
        the evil Serpent Empire, cel animation 80s style" --duration 90 --fps 24 --res 480p

    python -m comfymovies "cyberpunk city chase" --duration 60 --scenes 5 \
        --music-eleven "driving synthwave, 120 bpm, tense"
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from .build import MovieSpec, Scene, build_workflow, frames_for
from .comfy import ComfyClient, ComfyError
from .config import Config
from .prompts import expand_concept

# Named resolutions snapped to LTX's /64 requirement, ~16:9 where possible.
RES_PRESETS = {
    "480p": (896, 512),
    "512": (768, 512),
    "576p": (1024, 576),
    "720p": (1280, 704),
}


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="comfymovies",
        description="Generate a seamless LTX-2 movie from a text concept.",
    )
    p.add_argument("concept", help="One-line description of the movie")
    p.add_argument("--duration", type=float, default=60.0,
                   help="Target length in seconds (default 60)")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--res", default="480p",
                   help="480p|512|576p|720p or WxH (default 480p)")
    p.add_argument("--scenes", type=int, default=0,
                   help="Number of scheduled scene beats (0 = auto by duration)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-enhance", action="store_true",
                   help="Skip on-box gemma prompt enhancement")
    p.add_argument("--no-schedule", action="store_true",
                   help="One continuous prompt instead of temporal scene scheduling")
    p.add_argument("--negative", default="")
    p.add_argument("--out", default="", help="Output .mp4 path")
    p.add_argument("--music-eleven", metavar="PROMPT", default="",
                   help="Also score the movie with ElevenLabs music (merged copy)")
    p.add_argument("--music-keep-native", action="store_true",
                   help="Mix ElevenLabs music over the native audio (duck native)")
    p.add_argument("--dry-run", action="store_true",
                   help="Build + validate the workflow, then stop")
    p.add_argument("--timeout", type=float, default=5400,
                   help="Max seconds to wait for render (default 5400 = 90m)")
    return p.parse_args(argv)


def resolve_res(res: str) -> tuple[int, int]:
    if res in RES_PRESETS:
        return RES_PRESETS[res]
    if "x" in res.lower():
        w, h = res.lower().split("x", 1)
        return int(w), int(h)
    raise SystemExit(f"Unknown --res '{res}'. Use {list(RES_PRESETS)} or WxH.")


def auto_scene_count(duration: float) -> int:
    """Roughly one beat per ~15s, clamped to a sensible range."""
    return max(1, min(5, round(duration / 15)))


def build_spec(args: argparse.Namespace, cfg: Config) -> MovieSpec:
    width, height = resolve_res(args.res)
    n_scenes = args.scenes or auto_scene_count(args.duration)
    scenes = (
        [Scene(args.concept)] if n_scenes <= 1
        else expand_concept(args.concept, n_scenes, cfg)
    )
    spec = MovieSpec(
        scenes=scenes, width=width, height=height, fps=args.fps,
        seconds=args.duration, seed=args.seed,
        enhance=not args.no_enhance,
        schedule_scenes=None if not args.no_schedule else False,
    )
    if args.negative:
        spec.negative = args.negative
    return spec


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    cfg = Config.from_env()

    spec = build_spec(args, cfg)
    graph = build_workflow(spec)
    frames = frames_for(spec.seconds, spec.fps)

    print(f"» {len(spec.scenes)} scene(s), {spec.width}x{spec.height}, "
          f"{spec.fps}fps, ~{spec.seconds:.0f}s ({frames} frames), "
          f"enhance={spec.enhance}, nodes={len(graph)}")
    for i, s in enumerate(spec.scenes, 1):
        print(f"    scene {i}: {s.prompt[:90]}")

    client = ComfyClient(host=cfg.comfy_host, port=cfg.comfy_port)
    try:
        prompt_id = client.submit(graph)
    except ComfyError as e:
        print(f"✗ ComfyUI rejected the workflow:\n{e}", file=sys.stderr)
        return 2
    print(f"✓ validated & queued: prompt_id={prompt_id}")

    if args.dry_run:
        client.interrupt()
        print("dry-run: interrupted after validation.")
        return 0

    os.makedirs(cfg.output_dir, exist_ok=True)
    start = time.time()
    print("… rendering (this can take a while for long clips)…")
    try:
        entry = client.wait(prompt_id, timeout=args.timeout, poll=5)
    except ComfyError as e:
        print(f"✗ render failed: {e}", file=sys.stderr)
        return 3

    files = ComfyClient.find_outputs(entry)
    if not files:
        print("✗ no output files produced", file=sys.stderr)
        return 4

    out = args.out or os.path.join(cfg.output_dir, files[0]["filename"])
    client.download(files[0], out)
    print(f"✓ movie ready in {time.time()-start:.0f}s: {out}")

    if args.music_eleven:
        from .music import MusicError, score_movie
        scored = os.path.splitext(out)[0] + "_eleven.mp4"
        try:
            score_movie(out, args.music_eleven, scored, cfg=cfg,
                        keep_native=args.music_keep_native)
            print(f"✓ ElevenLabs-scored copy: {scored}")
        except MusicError as e:
            print(f"! ElevenLabs scoring skipped: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
