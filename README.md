# ComfyUIMovies

Turn a single line of text into a **seamless 60–120 second, 24 fps, ~480p movie**,
rendered on a remote **ComfyUI** instance (RTX 5090) using the **LTX‑2.3** video
model — with native audio on short clips and an optional **ElevenLabs** music
score merged in.

```bash
python -m comfymovies \
  "GI Joe style 1980s cartoon, American soldiers fight the evil Serpent Empire, cel animation 80s style" \
  --duration 90 --fps 24 --res 480p
```

Everything runs on **one GPU, one ComfyUI workflow**: prompt enhancement, scene
progression, video, and (for short clips) synchronized audio are all produced
on the 5090 via ComfyUI's built‑in gemma LLM node — no external prompt service
required.

> **New here?** See **[docs/SETUP.md](docs/SETUP.md)** for the full architecture
> and a step‑by‑step install (NVIDIA **and** AMD), and
> **[prompts/README.md](prompts/README.md)** for how to write your own prompts.

---

## How it works

```
concept ──► scene split ──► [per scene] gemma enhance ──► CLIP encode ──┐
(one line)  (offline arc     (TextGenerateLTX2Prompt on the 5090)        │
             template)                                                   ▼
                                                    LTX‑2.3 (distilled LoRA)
                                                    + context windows (long)
                                                    + joint audio (≤ ~40 s)
                                                                         │
                                                                         ▼
                                                     CreateVideo ──► SaveVideo ──► .mp4
                                                                         │
                                              (optional) ElevenLabs music ┘  merged via ffmpeg
```

**Seamless, no hard cuts.** Instead of stitching separate clips, the movie is a
single continuous latent:

- **Long clips (> context length):** *continuous mode* — one gemma‑enhanced
  prompt whose narrative arc (establish → rise → clash → climax → resolve)
  evolves across `LTXVContextWindows` with FreeNoise blending.
- **Short clips:** optional *scene scheduling* pins each beat to its slice of the
  timeline with `ConditioningSetAreaPercentageVideo` so beats flow into each
  other in one take.

**Audio.** LTX‑2's joint audio path generates a synced soundtrack for clips up to
~40 s (the audio latent caps at 1000 frames). Longer movies render video‑only and
are meant to be scored — e.g. with the ElevenLabs option below, which is why the
default long‑form soundtrack comes from ElevenLabs.

---

## Requirements

- Python ≥ 3.10 (standard library only — **no pip dependencies** for core use).
- A reachable ComfyUI **0.27+** with the LTX‑2.3 assets installed:
  - checkpoint `ltx-2.3-22b-dev-fp8.safetensors`
  - text encoder `gemma_3_12B_it_fp4_mixed.safetensors`
  - distilled LoRA `ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors`
- `ffmpeg` on `PATH` (only for the ElevenLabs music merge).

Configure the server (defaults target the PC at `192.168.1.90:8188`):

```bash
export COMFY_HOST=192.168.1.90
export COMFY_PORT=8188
export COMFYMOVIES_OUTPUT=./output
# optional ElevenLabs scoring:
export ELEVENLABS_API_KEY=sk_...
```

---

## CLI

```
python -m comfymovies "<concept>" [options]

  --duration SECONDS     target length (default 60)
  --fps N                frames per second (default 24)
  --res 480p|512|576p|720p|WxH
  --scenes N             number of narrative beats (0 = auto by duration)
  --scene-file PATH      explicit scene beats from a JSON/txt/md file
  --seed N
  --quality              full model + multi-step schedule + higher CFG (sharper)
  --steps N              denoising steps (0 = fast distilled; >0 = quality)
  --cfg FLOAT            guidance scale (default 1.0 fast / 3.5 quality)
  --lora-strength FLOAT  distilled LoRA strength (default 0.5 fast / 0 quality)
  --no-enhance           skip on-box gemma prompt enhancement
  --no-schedule          one continuous prompt (no temporal scene scheduling)
  --negative "..."       override the default negative prompt
  --out PATH             output .mp4 path
  --music-eleven "PROMPT"    also render an ElevenLabs-scored copy (_eleven.mp4)
  --music-keep-native        duck native audio under the ElevenLabs music
  --dry-run              build + validate the workflow, then stop
  --timeout SECONDS      max wait for the render (default 5400)
```

### Examples

```bash
# 60s seamless GI Joe short (video-only + score with ElevenLabs)
python -m comfymovies "GI Joe style 1980s cartoon, American soldiers fight \
  the evil Serpent Empire, cel animation 80s style" \
  --duration 60 --music-eleven "heroic 1980s cartoon theme, brass, driving drums"

# 20s clip with native LTX audio, 5 scheduled beats
python -m comfymovies "a neon cyberpunk street chase in the rain" \
  --duration 20 --scenes 5

# Just validate the graph against the server
python -m comfymovies "a lone astronaut on mars, photorealistic" --dry-run
```

You will **A/B the audio**: keep the native LTX soundtrack, or swap in an
ElevenLabs score you generated separately and merge it:

```python
from comfymovies.music import merge_music
merge_music("output/movie.mp4", "my_elevenlabs_track.mp3",
            "output/movie_scored.mp4")            # replace audio
# or mix over the native track:
merge_music("output/movie.mp4", "my_track.mp3", "output/movie_mixed.mp4",
            keep_native=True, native_volume=0.35)
```

---

## Library

```python
from comfymovies.build import MovieSpec, Scene, build_workflow
from comfymovies.comfy import ComfyClient

spec = MovieSpec(
    scenes=[Scene("soldiers charge at dawn"), Scene("a serpent mech rises")],
    width=896, height=512, fps=24, seconds=60, seed=1980,
)
graph = build_workflow(spec)          # ComfyUI /prompt API graph

client = ComfyClient(host="192.168.1.90", port=8188)
pid = client.submit(graph)            # validates + queues
entry = client.wait(pid)              # blocks until done
client.download(client.find_outputs(entry)[0], "output/movie.mp4")
```

### Modules

| module | purpose |
|--------|---------|
| `comfymovies/build.py`   | Assemble parameterized LTX‑2.3 API graphs (audio gating, context windows, scene scheduling, on‑box enhancement). |
| `comfymovies/comfy.py`   | ComfyUI HTTP client: submit, poll, download, interrupt. |
| `comfymovies/prompts.py` | One‑liner → scene beats (offline arc template; optional OpenAI‑compatible LLM). |
| `comfymovies/music.py`   | ElevenLabs music generation + ffmpeg muxing. |
| `comfymovies/convert.py` | Convert exported ComfyUI **UI/subgraph** workflows → **API** format. |
| `comfymovies/cli.py`     | Command‑line entry point. |

`convert.py` is a general utility: drop any workflow you exported from the
ComfyUI web UI (including new subgraph‑based templates) into `workflows/` and
convert it to the `/prompt` API format, resolving links, widget values,
`Reroute` passthroughs, and single top‑level subgraph flattening.

---

## Docker (run on the sleepycore 3090 server)

The pipeline is a thin client — it can run anywhere that can reach the ComfyUI
box. To run it as a container on the home server:

```bash
docker build -t comfymovies docker/     # or: docker compose -f docker/docker-compose.yml build
docker run --rm \
  -e COMFY_HOST=192.168.1.90 -e COMFY_PORT=8188 \
  -e ELEVENLABS_API_KEY=$ELEVENLABS_API_KEY \
  -v "$PWD/output:/app/output" \
  comfymovies "cyberpunk city chase" --duration 60
```

---

## Tests

```bash
python -m pytest -q
```

Tests are **offline** — they exercise graph construction, audio gating, scene
scheduling, and the UI→API converter without touching the ComfyUI server.

---

## Notes & limits

- Dimensions snap to multiples of 64; frame counts to `8n + 1` (LTX‑2 rules).
- The 22B two‑pass spatial upscaler from the official template is **not** used
  (that upscaler model isn't installed on the target box); a clean single‑pass
  distilled graph is generated instead.
- On‑box prompt enhancement uses `sampling_mode="off"` (deterministic); the
  `"on"` sampling branch requires namespaced sub‑fields the flat API can't set.
- Long renders are heavy: a 60 s / 1441‑frame clip at 896×512 takes on the order
  of an hour on an RTX 5090.
