# Architecture & Setup

This guide explains **how ComfyUIMovies works** and **everything you need to
install** to run it yourself — on NVIDIA or AMD GPUs.

---

## 1. Architecture

ComfyUIMovies is a **thin orchestration client**. It does no ML itself; it builds
a ComfyUI graph, sends it to a **ComfyUI server** (where the GPU lives), waits,
and downloads the result. This means the client can run anywhere (your laptop, a
different server, a container) while the heavy lifting happens on the GPU box.

```
┌─────────────────────────────┐         HTTP /prompt          ┌──────────────────────────────┐
│  ComfyUIMovies (this repo)  │  ─────────────────────────▶   │      ComfyUI server (GPU)     │
│  Python, stdlib only        │                               │  ComfyUI 0.27+  ·  LTX-2.3     │
│                             │   ◀───────────────────────    │                              │
│  build.py  → API graph      │      /history, /view (mp4)    │  ┌────────────────────────┐  │
│  comfy.py  → submit/poll    │                               │  │ CheckpointLoaderSimple │  │
│  prompts.py→ scenes         │                               │  │ + distilled LoRA       │  │
│  cli.py    → UX             │                               │  │ LTXAVTextEncoderLoader │  │
│  music.py  → ElevenLabs     │                               │  │  (gemma LLM enhance)   │  │
└─────────────────────────────┘                               │  │ LTXVContextWindows     │  │
             │                                                 │  │ SamplerCustomAdvanced  │  │
             │  ffmpeg (local)                                 │  │ VAEDecode / Audio VAE  │  │
             ▼                                                 │  │ CreateVideo → SaveVideo│  │
     merged .mp4 with music                                    │  └────────────────────────┘  │
                                                               └──────────────────────────────┘
```

### The generation graph (built by `build.py`)

1. **CheckpointLoaderSimple** loads the LTX‑2.3 model (MODEL + VAE).
2. **LoraLoaderModelOnly** (fast mode) applies the distilled few‑step LoRA; in
   `--quality` mode this is skipped for the full model.
3. **LTXAVTextEncoderLoader** loads the gemma text encoder (LTX‑2's CLIP).
4. **TextGenerateLTX2Prompt** *(on the GPU)* rewrites each scene into a rich
   cinematic prompt — the "prompt enhancement" runs on the same box, no external
   LLM.
5. **CLIPTextEncode** → **LTXVConditioning** builds positive/negative
   conditioning at the target frame rate.
6. Latents: **EmptyLTXVLatentVideo** (+ **LTXVEmptyLatentAudio** →
   **LTXVConcatAVLatent** for clips ≤ ~40 s that get native audio).
7. **LTXVContextWindows** (long clips) wraps the model so a single continuous
   latent renders seamlessly in overlapping windows with FreeNoise blending.
8. Sigmas: **ManualSigmas** (fast distilled) or **LTXVScheduler** (`--quality`,
   multi‑step) → **CFGGuider** → **SamplerCustomAdvanced**.
9. **VAEDecode** (+ **LTXVAudioVAEDecode**) → **CreateVideo** → **SaveVideo**.

### Design choices worth knowing

- **Seamless, not stitched.** One continuous latent. Long clips use *continuous
  mode* (one enhanced prompt whose arc evolves across context windows); short
  clips can *temporally schedule* scenes with
  `ConditioningSetAreaPercentageVideo`.
- **Audio cap.** LTX's audio latent maxes at 1000 frames (~40 s @ 24 fps). Longer
  movies render **video‑only** and are scored with ElevenLabs.
- **Fast vs. quality.** Default = distilled LoRA + ~3 steps (≈2 min / 60 s, soft).
  `--quality` = full model + `LTXVScheduler` multi‑step + higher CFG (sharp,
  slower).
- **Converter.** `convert.py` turns any workflow you export from the ComfyUI web
  UI (including new subgraph templates) into the `/prompt` API format — handy for
  adapting other pipelines.

---

## 2. Hardware requirements

| | Minimum | Comfortable | Notes |
|---|---|---|---|
| **GPU VRAM** | ~16 GB (fast mode, short clips) | 24–32 GB | LTX‑2.3 22B is large; context windows bound VRAM for long clips |
| **System RAM** | 32 GB | 64 GB | |
| **Disk** | ~60 GB free | 100 GB+ | model weights are tens of GB |

The reference box for this project is an **RTX 5090 (32 GB)**; a 60 s / 1441‑frame
quality render takes on the order of tens of minutes. Smaller GPUs work best in
fast mode and/or at lower resolution and shorter durations. If you hit
out‑of‑memory: lower `--res`, shorten `--duration`, reduce `--steps`, or (advanced)
tune context‑window size in `MovieSpec`.

---

## 3. Install the ComfyUI server (the GPU side)

### 3a. NVIDIA (recommended — the models are fp8/fp4)

The default model files are **fp8/fp4 quantized**, which need a recent NVIDIA GPU
(Ada/Blackwell — RTX 40xx/50xx — for real fp8 throughput; Ampere RTX 30xx runs
them with emulation).

```bash
git clone https://github.com/comfyanonymous/ComfyUI
cd ComfyUI
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python main.py --listen 0.0.0.0                       # expose on the LAN
```

The easiest path on Windows is the **ComfyUI portable / Desktop** build — it ships
its own Python and CUDA. Launch it with network access (`--listen 0.0.0.0`) and
open TCP **8188** in the firewall so the client can reach it.

### 3b. AMD (Radeon) — works, with caveats

> **Important:** the *default* fp8/fp4 weights generally **do not run on AMD**
> (fp8 compute isn't exposed on consumer Radeon via ROCm/DirectML/ZLUDA as of
> 2025). On AMD you must use **bf16/fp16** model variants (see §4) and expect
> higher VRAM use and slower speed.

| Backend | OS | How | Notes |
|---|---|---|---|
| **ROCm** | Linux | `pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2` then standard ComfyUI install | **Best AMD option.** Supported GPUs: RX 7900 XT/XTX, RX 9070, Radeon PRO / MI‑series. FP16/BF16 only. |
| **ZLUDA** | Windows | Community ComfyUI‑ZLUDA forks | CUDA‑on‑AMD shim; experimental, slower, some nodes fail. |
| **DirectML** | Windows | `pip install torch-directml`, run `python main.py --directml` | Widest Windows AMD compatibility; slowest; no fp8. |

Practical AMD recommendation: **Linux + ROCm on an RX 7900 XTX / 9070 (or
MI‑series)** with **bf16** weights. Because 22B bf16 is very heavy (~44 GB), most
Radeon users should prefer the smaller **LTX‑2 19B bf16** and/or lower resolution.

### 3c. Apple Silicon / CPU
ComfyUI runs on Apple Silicon (MPS) but LTX‑2 video at these sizes is impractically
slow; treat it as unsupported for this project.

---

## 4. Download the models

Place files under `ComfyUI/models/`. Sources are on Hugging Face
([Lightricks](https://huggingface.co/Lightricks)) and the ComfyUI‑distributed
mirrors; the official **LTX‑2 ComfyUI templates** (Templates → Video → *LTX‑2*)
include one‑click download links and are the most reliable source of the exact
filenames.

| Role | Default file (NVIDIA, fp8/fp4) | Folder |
|---|---|---|
| Checkpoint | `ltx-2.3-22b-dev-fp8.safetensors` | `models/checkpoints/` |
| Text encoder | `gemma_3_12B_it_fp4_mixed.safetensors` | `models/text_encoders/` (a.k.a. `clip/`) |
| Distilled LoRA | `ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors` | `models/loras/` |
| Audio VAE | *(bundled in the checkpoint on this build)* | — |

**AMD / non‑fp8 users:** substitute the **bf16/fp16** equivalents from the same
Lightricks repo (e.g. an `ltx-2-19b` bf16 checkpoint + a bf16 gemma encoder), then
point the client at them by editing the constants at the top of
[`comfymovies/build.py`](../comfymovies/build.py):

```python
CKPT = "your-ltx-2-bf16.safetensors"
TEXT_ENCODER = "your-gemma-bf16.safetensors"
DISTILLED_LORA = "your-distilled-lora.safetensors"   # or run --quality (LoRA off)
```

The fastest way to confirm the exact names your server sees:

```bash
curl -s http://<COMFY_HOST>:8188/object_info | \
  python -c "import json,sys; d=json.load(sys.stdin); \
  print(d['CheckpointLoaderSimple']['input']['required']['ckpt_name'][0])"
```

Use whatever that prints as `CKPT` (and likewise inspect `UNETLoader`,
`CLIPLoader`, `LoraLoader` for the others).

---

## 5. Install & configure the client (this repo)

```bash
git clone https://github.com/Sleepyreaper/ComfyUIMovies
cd ComfyUIMovies
pip install -e .          # or: python -m comfymovies ...  (no install needed; stdlib only)
```

Point it at your ComfyUI server and (optionally) ElevenLabs:

```bash
export COMFY_HOST=192.168.1.90      # your GPU box
export COMFY_PORT=8188
export COMFYMOVIES_OUTPUT=./output
export ELEVENLABS_API_KEY=sk_...    # optional, for --music-eleven
```

`ffmpeg` must be on `PATH` for the ElevenLabs merge (`brew install ffmpeg`,
`apt install ffmpeg`, or `choco install ffmpeg`).

---

## 6. First run

```bash
# validate everything end-to-end without a long render:
python -m comfymovies "a lone astronaut on mars, photorealistic" --dry-run

# a real short clip (native audio, fast):
python -m comfymovies "a neon cyberpunk street chase in the rain" --duration 12

# the flagship: 60s quality movie
python -m comfymovies "GI Joe style 1980s cartoon, American soldiers fight \
  the evil Serpent Empire, cel animation 80s style" \
  --duration 60 --res 480p --quality --seed 1980
```

Bring your own prompts (no code changes): see
[`prompts/README.md`](../prompts/README.md) for concept strings, scene files, and
music prompts.

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `Prompt rejected … value_not_in_list` for a model name | The filename doesn't match what your server has — inspect `/object_info` (§4) and update `build.py` constants. |
| `narrow(): length must be non-negative` | Don't combine context windows with per‑window temporal scheduling (the code already gates this; only relevant if you hand‑edit). |
| `Value … bigger than max of 1000` on audio | Clip > ~40 s — audio is auto‑disabled; score with ElevenLabs. |
| Out of memory | Lower `--res`, `--duration`, or `--steps`; use bf16 19B on smaller GPUs. |
| Output is soft/hazy | Use `--quality` (full model, more steps, higher CFG). |
| Can't reach the server | Start ComfyUI with `--listen 0.0.0.0` and open TCP 8188 in the firewall. |
