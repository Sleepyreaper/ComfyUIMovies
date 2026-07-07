"""Turn a one-line concept into a list of cinematic scene beats.

Two backends:

* **template** (default, offline) — deterministically derives a shared style
  suffix from the concept and expands it across a classic narrative arc. Always
  available, no network.
* **llm** (optional) — an OpenAI-compatible chat endpoint (e.g. a local model
  behind ``LLM_BASE_URL``) that writes richer, concept-aware beats.
"""
from __future__ import annotations

import json
import re
import urllib.request

from .build import Scene
from .config import Config

# Style cues we lift verbatim from the concept so every beat stays consistent.
_STYLE_HINTS = [
    "cel animation", "cel-shaded", "1980s cartoon", "80s cartoon", "80s style",
    "anime", "cartoon", "photorealistic", "photorealism", "claymation",
    "stop motion", "pixar", "watercolor", "noir", "cyberpunk", "retro",
    "vhs", "film grain", "hand-drawn", "comic book",
]

_ARC = [
    "Establishing shot: {subject}. Wide cinematic view that sets the scene.",
    "Rising action: {subject}. The conflict builds, energy rising.",
    "Confrontation: {subject}. The forces clash head-on, dramatic and intense.",
    "Climax: {subject}. The decisive moment, maximum spectacle and motion.",
    "Resolution: {subject}. The aftermath settles, a final memorable image.",
]


def _extract_style(concept: str) -> str:
    low = concept.lower()
    found = [h for h in _STYLE_HINTS if h in low]
    # De-dup while preserving order and collapsing overlapping 80s hints.
    seen: list[str] = []
    for h in found:
        if h not in seen:
            seen.append(h)
    return ", ".join(seen)


def _core_subject(concept: str) -> str:
    """Strip trailing style clauses so the arc focuses on the action."""
    subject = concept.strip().rstrip(".")
    for hint in sorted(_STYLE_HINTS, key=len, reverse=True):
        subject = re.sub(rf",?\s*{re.escape(hint)}\b", "", subject,
                         flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", subject).strip(" ,")


def expand_template(concept: str, n_scenes: int) -> list[Scene]:
    subject = _core_subject(concept)
    style = _extract_style(concept)
    n = max(1, min(n_scenes, len(_ARC)))
    # Pick an evenly spaced subset of the arc for the requested scene count.
    if n == 1:
        beats = [subject]
    else:
        idx = [round(i * (len(_ARC) - 1) / (n - 1)) for i in range(n)]
        beats = [_ARC[i].format(subject=subject) for i in idx]
    suffix = f", {style}" if style else ""
    return [Scene(prompt=f"{b}{suffix}") for b in beats]


_LLM_SYSTEM = (
    "You are a film director writing shot lists for a text-to-video model. "
    "Given a concept, output a JSON array of {n} vivid, self-contained scene "
    "prompts that flow as one continuous, seamless story with NO hard cuts. "
    "Keep a single consistent art style across every scene. Each prompt is one "
    "or two sentences describing subject, action, camera and lighting. "
    "Return ONLY the JSON array of strings."
)


def expand_llm(concept: str, n_scenes: int, cfg: Config) -> list[Scene]:
    body = json.dumps({
        "model": cfg.llm_model,
        "messages": [
            {"role": "system",
             "content": _LLM_SYSTEM.format(n=n_scenes)},
            {"role": "user", "content": concept},
        ],
        "temperature": 0.8,
        "stream": False,
    }).encode()
    headers = {"Content-Type": "application/json"}
    if cfg.llm_api_key:
        headers["Authorization"] = f"Bearer {cfg.llm_api_key}"
    url = cfg.llm_base_url.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read())
    content = data["choices"][0]["message"]["content"]
    prompts = _parse_json_array(content)
    if not prompts:
        raise ValueError("LLM returned no usable scene list")
    return [Scene(prompt=p) for p in prompts[:n_scenes]]


def _parse_json_array(text: str) -> list[str]:
    text = text.strip()
    # Strip code fences and any prose around the array.
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        text = match.group(0)
    try:
        arr = json.loads(text)
        return [str(x).strip() for x in arr if str(x).strip()]
    except json.JSONDecodeError:
        return []


def expand_concept(
    concept: str, n_scenes: int, cfg: Config | None = None
) -> list[Scene]:
    """Expand a concept into ``n_scenes`` beats, preferring the LLM if configured.

    Falls back to the offline template expander on any LLM failure so movie
    generation never hard-depends on a network service.
    """
    cfg = cfg or Config.from_env()
    if cfg.llm_base_url:
        try:
            return expand_llm(concept, n_scenes, cfg)
        except Exception:
            pass
    return expand_template(concept, n_scenes)
