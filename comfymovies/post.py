"""Local post-processing to make finished clips "killer": smooth + sharp + HD.

Everything here runs on the machine with ffmpeg (your Mac), so it needs no extra
ComfyUI models on the render box:

* **Motion-compensated frame interpolation** (ffmpeg ``minterpolate``) retimes
  WAN's native 16 fps to a smooth 24/30/60 fps by synthesizing in-between frames
  along real motion vectors — much smoother than frame duplication.
* **High-quality upscale + light sharpen** (lanczos + ``unsharp``) lifts 480p to
  720p/1080p with crisp cel edges.

For the *best* result, render larger natively (WAN does 720p) — but this polish
pass turns the existing 480p/16 fps movies into HD/24 fps without re-rendering.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass


class PostError(RuntimeError):
    pass


@dataclass
class PolishSpec:
    fps: int = 24                 # target frame rate (interpolated)
    height: int = 720             # target height; width scales to keep aspect
    sharpen: float = 0.6          # unsharp amount (0 = off)
    crf: int = 16                 # x264 quality (lower = better/bigger)
    interpolate: bool = True      # motion-compensated retime vs simple fps set


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise PostError("ffmpeg not found on PATH")
    return exe


def build_filter(spec: PolishSpec) -> str:
    parts: list[str] = []
    if spec.interpolate:
        parts.append(
            f"minterpolate=fps={spec.fps}:mi_mode=mci:mc_mode=aobmc:"
            f"me_mode=bidir:vsbmc=1"
        )
    else:
        parts.append(f"fps={spec.fps}")
    # -2 keeps width even and preserves aspect ratio.
    parts.append(f"scale=-2:{spec.height}:flags=lanczos")
    if spec.sharpen > 0:
        parts.append(f"unsharp=5:5:{spec.sharpen}")
    return ",".join(parts)


def polish(src: str, dst: str, spec: PolishSpec | None = None,
           audio: str | None = None) -> str:
    """Interpolate + upscale + sharpen ``src`` into ``dst``.

    If ``audio`` is given (e.g. an ElevenLabs track), it is muxed in and the
    output is trimmed to the shorter stream.
    """
    spec = spec or PolishSpec()
    exe = _ffmpeg()
    cmd = [exe, "-y", "-v", "error", "-i", src]
    if audio:
        cmd += ["-i", audio]
    cmd += ["-vf", build_filter(spec),
            "-c:v", "libx264", "-crf", str(spec.crf), "-pix_fmt", "yuv420p"]
    if audio:
        cmd += ["-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", "192k",
                "-shortest"]
    cmd += [dst]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise PostError(f"ffmpeg polish failed: {proc.stderr[-800:]}")
    return dst


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys
    p = argparse.ArgumentParser(
        prog="comfymovies.post",
        description="Polish a clip: motion-interpolate + upscale + sharpen.")
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--sharpen", type=float, default=0.6)
    p.add_argument("--crf", type=int, default=16)
    p.add_argument("--no-interpolate", action="store_true")
    p.add_argument("--audio", default="", help="optional audio track to mux in")
    a = p.parse_args(argv if argv is not None else sys.argv[1:])
    spec = PolishSpec(fps=a.fps, height=a.height, sharpen=a.sharpen, crf=a.crf,
                      interpolate=not a.no_interpolate)
    out = polish(a.src, a.dst, spec, audio=a.audio or None)
    print(f"✓ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
