"""Runtime configuration, resolved from environment with sane defaults.

Env vars (all optional):
  COMFY_HOST / COMFY_PORT       — ComfyUI server (default 192.168.1.90:8188)
  COMFYMOVIES_OUTPUT            — local output directory (default ./output)
  LLM_BASE_URL / LLM_API_KEY / LLM_MODEL
                                — optional OpenAI-compatible endpoint used to
                                  expand a one-liner into scene beats
  ELEVENLABS_API_KEY / ELEVENLABS_MODEL / ELEVENLABS_MUSIC_LENGTH_MS
                                — ElevenLabs music generation
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Config:
    comfy_host: str = "192.168.1.90"
    comfy_port: int = 8188
    output_dir: str = "output"

    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = "qwen3:14b"

    elevenlabs_api_key: str = ""
    elevenlabs_model: str = "music_v1"

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            comfy_host=os.environ.get("COMFY_HOST", "192.168.1.90"),
            comfy_port=int(os.environ.get("COMFY_PORT", "8188")),
            output_dir=os.environ.get("COMFYMOVIES_OUTPUT", "output"),
            llm_base_url=os.environ.get("LLM_BASE_URL", ""),
            llm_api_key=os.environ.get("LLM_API_KEY", ""),
            llm_model=os.environ.get("LLM_MODEL", "qwen3:14b"),
            elevenlabs_api_key=os.environ.get("ELEVENLABS_API_KEY", ""),
            elevenlabs_model=os.environ.get("ELEVENLABS_MODEL", "music_v1"),
        )
