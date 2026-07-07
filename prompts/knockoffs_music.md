# Disguised Robots & Rumble Dogs — audio prompts

ElevenLabs Music prompts for the two original 80s knock-off shorts. Both render
video‑only at 60 s (over the native‑audio cap), so score them with ElevenLabs.

---

## DISGUISED ROBOTS (Transformers‑style)

> Epic 1980s cartoon robots action theme, fully instrumental, ~60 seconds.
> Heavy heroic synth‑brass fanfare and a driving electronic rock beat with
> chugging electric guitars, metallic percussion and soaring 80s synth leads.
> Arc: (0–15s) heroic transforming‑robots main theme; (15–35s) dark menacing
> villain motif with distorted low synths as the enemy arrives; (35–50s)
> full‑throttle battle with pounding drums and screaming guitar; (50–60s)
> triumphant victory fanfare, big final power chord. Powerful, mechanical,
> cinematic, no vocals. ~140 BPM, key of E minor.

**Length:** 60000 ms · No vocals.

---

## RUMBLE DOGS (ThunderCats‑style)

> Epic 1980s fantasy cartoon adventure theme, fully instrumental, ~60 seconds.
> Sweeping orchestral brass and strings blended with bold 80s synths, heroic and
> mystical. Arc: (0–15s) majestic heroic main theme with noble French horns;
> (15–35s) dark ominous sorcerer motif, choir‑like pads, tense low strings and
> tribal drums; (35–50s) soaring battle climax, full orchestra and synth,
> galloping percussion; (50–60s) triumphant sunrise resolution with a proud,
> ringing brass finale. Adventurous, magical, cinematic, no vocals. ~132 BPM,
> key of C minor to C major.

**Length:** 60000 ms · No vocals.

---

## Auto‑score from the CLI

Set `ELEVENLABS_API_KEY`, then either add `--music-eleven "<prompt above>"` to a
render, or score an existing file:

```bash
python - <<'PY'
from comfymovies.music import merge_music
merge_music("output/disguised_robots_60s.mp4",
            "path/to/your_track.mp3",
            "output/disguised_robots_60s_scored.mp4")
PY
```
