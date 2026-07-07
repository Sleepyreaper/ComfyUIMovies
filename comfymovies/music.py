"""ElevenLabs music generation and audio muxing via ffmpeg.

The LTX-2 pipeline already produces a native soundtrack. This module is for the
*optional* workflow the user wants: generate a music track with ElevenLabs and
merge it into the rendered movie, so they can A/B the native audio against an
ElevenLabs score and keep whichever they prefer.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import urllib.request

from .config import Config


class MusicError(RuntimeError):
    pass


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise MusicError("ffmpeg not found on PATH")
    return exe


def _ffprobe_duration(path: str) -> float:
    exe = shutil.which("ffprobe")
    if not exe:
        return 0.0
    out = subprocess.run(
        [exe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True,
    )
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def generate_music(
    prompt: str, out_path: str, *, length_ms: int, cfg: Config | None = None,
) -> str:
    """Generate an MP3 with the ElevenLabs Music API. Returns ``out_path``.

    ``length_ms`` should roughly match the movie duration. Raises if no API key
    is configured.
    """
    cfg = cfg or Config.from_env()
    if not cfg.elevenlabs_api_key:
        raise MusicError(
            "ELEVENLABS_API_KEY is not set — cannot generate ElevenLabs music"
        )
    body = json.dumps({
        "prompt": prompt,
        "music_length_ms": int(length_ms),
        "model_id": cfg.elevenlabs_model,
    }).encode()
    req = urllib.request.Request(
        "https://api.elevenlabs.io/v1/music",
        data=body,
        headers={
            "xi-api-key": cfg.elevenlabs_api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        audio = r.read()
    with open(out_path, "wb") as f:
        f.write(audio)
    return out_path


def merge_music(
    video_path: str, music_path: str, out_path: str, *,
    music_volume: float = 1.0, keep_native: bool = False,
    native_volume: float = 0.35,
) -> str:
    """Mux a music track onto a video with ffmpeg.

    * ``keep_native=False`` (default): replace the movie's audio with the music,
      trimmed/padded to the video length.
    * ``keep_native=True``: duck the native LTX audio under the music and mix.
    """
    exe = _ffmpeg()
    if keep_native:
        # Mix native (ducked) + music, ending with the shortest stream (video).
        filt = (
            f"[0:a]volume={native_volume}[a0];"
            f"[1:a]volume={music_volume}[a1];"
            f"[a0][a1]amix=inputs=2:duration=first:dropout_transition=0[aout]"
        )
        cmd = [
            exe, "-y", "-i", video_path, "-i", music_path,
            "-filter_complex", filt,
            "-map", "0:v:0", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest", out_path,
        ]
    else:
        cmd = [
            exe, "-y", "-i", video_path, "-i", music_path,
            "-filter_complex", f"[1:a]volume={music_volume}[aout]",
            "-map", "0:v:0", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest", out_path,
        ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise MusicError(f"ffmpeg merge failed: {proc.stderr[-800:]}")
    return out_path


def score_movie(
    video_path: str, prompt: str, out_path: str, *,
    cfg: Config | None = None, keep_native: bool = False,
    music_volume: float = 1.0,
) -> str:
    """Generate ElevenLabs music sized to the movie and merge it in one call."""
    length_ms = int(max(1.0, _ffprobe_duration(video_path)) * 1000)
    music_tmp = out_path + ".music.mp3"
    generate_music(prompt, music_tmp, length_ms=length_ms, cfg=cfg)
    return merge_music(
        video_path, music_tmp, out_path,
        music_volume=music_volume, keep_native=keep_native,
    )
