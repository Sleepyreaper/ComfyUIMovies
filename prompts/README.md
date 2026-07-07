# Prompts

This folder holds **your movie prompts** — the easiest thing to fork and make
your own. Nothing here affects the code; add files freely.

## 1. Just pass a concept (simplest)

```bash
python -m comfymovies "a neon cyberpunk street chase in the rain" \
  --duration 30 --res 480p --quality
```

The concept is auto‑split into narrative beats and enhanced on the GPU. Add
`--scenes N` to control how many beats, or `--no-schedule` for one continuous
prompt.

## 2. Bring your own scenes (full control)

Write the exact beats and pass a **scene file**. Two formats:

**Plain text / markdown** — one scene per line (`#` lines are comments):

```text
# my_movie.txt
a lone astronaut steps onto the red martian dunes at sunrise, photorealistic
a dust storm rolls in as the astronaut runs for the lander, cinematic
the lander lifts off through swirling dust, triumphant wide shot
```

**JSON** — optional per‑scene `weight` (share of the timeline):

```json
{
  "scenes": [
    { "prompt": "establishing shot ...", "weight": 1.0 },
    { "prompt": "the confrontation ...", "weight": 1.5 }
  ]
}
```

Run it:

```bash
python -m comfymovies --scene-file prompts/my_movie.txt --duration 30 --res 480p --quality
# or the included example:
python -m comfymovies --scene-file prompts/example_scenes.json --duration 30 --res 480p --quality
```

Scene files **bypass** auto‑expansion — you get exactly the beats you wrote
(still enhanced on‑GPU unless you add `--no-enhance`).

## 3. Music prompts

For long clips (> ~40 s, which render video‑only), pair the movie with an
ElevenLabs score. See `gijoe_serpent_empire.md` for a worked example (a ready‑to‑
paste ElevenLabs Music prompt + CLI usage). Copy it as a template for your own.

## Tips

- **Consistency:** keep one art‑style phrase in every beat (e.g. `cel animation,
  1980s cartoon`) so the look doesn't drift.
- **Seamlessness:** beats flow into one another — write them as a continuous
  story, not disconnected shots.
- **Length vs. audio:** ≤ ~40 s keeps native LTX audio; longer is video‑only →
  score with ElevenLabs.
