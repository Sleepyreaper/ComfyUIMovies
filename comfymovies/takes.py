"""Automatic clip scoring for best-of-N take selection.

The single biggest quality lever vs one-shot rendering: render several takes of
each shot (varying seed / motion strength) and keep the BEST, the way polished
AI-video demos are actually made. This module scores a clip on:

  * motion    — mean absolute frame-to-frame luma change (too low => "no
                animation at all"; too high/chaotic => warping/melting)
  * sharpness — mean variance-of-Laplacian across frames (low => blur)
  * jerk      — std of per-frame motion (high => unstable / snakey warping)

`score()` combines them: reward sharpness + motion inside a healthy band,
penalize static, blur, and chaotic jerk. `pick_best()` chooses among takes.

Pure numpy + Pillow + ffmpeg (no GPU) so it runs during a render or offline.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass

import numpy as np


# healthy motion band (mean abs luma delta, 0-255 scale, on 320px frames)
MOTION_MIN = 1.2      # below => basically static ("no animation")
MOTION_LOW = 2.5      # start of the good band
MOTION_HIGH = 14.0    # end of the good band
MOTION_MAX = 24.0     # above => likely warping / chaos
SHARP_MIN = 45.0      # below => blurry


def _frames(path: str, width: int = 320, every: int = 2) -> np.ndarray:
    """Decode a clip to a stack of grayscale frames (T, H, W) via ffmpeg."""
    # get height for the target width preserving aspect
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", path],
        capture_output=True, text=True)
    try:
        w0, h0 = (int(x) for x in probe.stdout.strip().split(",")[:2])
        height = max(2, int(round(width * h0 / w0)) // 2 * 2)
    except Exception:
        height = 176
    cmd = ["ffmpeg", "-v", "error", "-i", path,
           "-vf", f"select=not(mod(n\\,{every})),scale={width}:{height}",
           "-vsync", "0", "-pix_fmt", "gray", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    n = len(raw) // (width * height)
    if n == 0:
        return np.zeros((0, height, width), dtype=np.uint8)
    return np.frombuffer(raw[: n * width * height], dtype=np.uint8).reshape(n, height, width)


def _laplacian_var(frame: np.ndarray) -> float:
    f = frame.astype(np.float32)
    lap = (-4 * f
           + np.roll(f, 1, 0) + np.roll(f, -1, 0)
           + np.roll(f, 1, 1) + np.roll(f, -1, 1))
    return float(lap[1:-1, 1:-1].var())


@dataclass
class ClipScore:
    motion: float
    sharpness: float
    jerk: float
    score: float
    verdict: str


def _band(m: float) -> float:
    """1.0 inside the healthy motion band, ramping to 0 outside it."""
    if m < MOTION_MIN or m > MOTION_MAX:
        return 0.0
    if m < MOTION_LOW:
        return (m - MOTION_MIN) / (MOTION_LOW - MOTION_MIN)
    if m > MOTION_HIGH:
        return max(0.0, (MOTION_MAX - m) / (MOTION_MAX - MOTION_HIGH))
    return 1.0


def score_clip(path: str) -> ClipScore:
    fr = _frames(path)
    if len(fr) < 3:
        return ClipScore(0, 0, 0, 0, "unreadable")
    diffs = np.abs(np.diff(fr.astype(np.float32), axis=0)).mean(axis=(1, 2))
    motion = float(diffs.mean())
    jerk = float(diffs.std())
    sharpness = float(np.mean([_laplacian_var(f) for f in fr[:: max(1, len(fr) // 8)]]))

    band = _band(motion)
    sharp_n = min(1.0, sharpness / 250.0)
    # penalize chaotic motion relative to its mean (warping/snakey)
    jerk_pen = 1.0 / (1.0 + max(0.0, jerk / max(motion, 0.1) - 0.9))
    score = 100.0 * band * (0.35 + 0.65 * sharp_n) * jerk_pen

    if motion < MOTION_MIN:
        verdict = "STATIC (no animation)"
    elif motion > MOTION_MAX:
        verdict = "CHAOTIC (warp risk)"
    elif sharpness < SHARP_MIN:
        verdict = "BLURRY"
    elif band >= 1.0 and sharp_n > 0.4:
        verdict = "GOOD"
    else:
        verdict = "ok"
    return ClipScore(round(motion, 2), round(sharpness, 1), round(jerk, 2),
                     round(score, 1), verdict)


def pick_best(paths: list[str]) -> tuple[str, ClipScore, list[ClipScore]]:
    scored = [(p, score_clip(p)) for p in paths]
    best = max(scored, key=lambda t: t[1].score)
    return best[0], best[1], [s for _, s in scored]


if __name__ == "__main__":
    import glob
    import sys
    pat = sys.argv[1] if len(sys.argv) > 1 else "output/serpent/shot*.mp4"
    rows = sorted(glob.glob(pat))
    print(f"{'clip':<34} {'motion':>7} {'sharp':>7} {'jerk':>6} {'score':>6}  verdict")
    for p in rows:
        s = score_clip(p)
        print(f"{p:<34} {s.motion:>7} {s.sharpness:>7} {s.jerk:>6} {s.score:>6}  {s.verdict}")
