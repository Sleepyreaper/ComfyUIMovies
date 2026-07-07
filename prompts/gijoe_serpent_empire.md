# GI Joe vs. the Serpent Empire — audio prompts

Companion soundtrack prompts for `output/gijoe_serpent_60s_quality.mp4`
(60 s, 24 fps, 896×512, 4‑beat arc: establish → rise → clash → resolve).
The movie is rendered **video‑only** (over the 40 s native‑audio cap), so the
score comes from ElevenLabs.

## ElevenLabs Music prompt (paste into the ElevenLabs Music UI)

> Heroic 1980s Saturday‑morning cartoon action theme, fully instrumental, ~60
> seconds. Big triumphant brass fanfare and a bold march‑style main melody over
> driving military snare and toms, punchy 80s synth brass stabs, electric guitar
> power chords, and a soaring lead synth. Structure it as a mini score: (0–15s)
> heroic wake‑up fanfare and marching build; (15–35s) rising tension with
> staccato strings, ticking percussion and a menacing low synth for the villain;
> (35–50s) full‑throttle battle climax, double‑time drums, blazing brass and
> guitar; (50–60s) victorious resolution with a proud, ringing final brass chord.
> Energetic, patriotic, adventurous, cinematic. No vocals, no lyrics. Tempo ~132
> BPM, key of D major.

**Length:** 60000 ms · **Vibe:** heroic / patriotic / adventurous · **No vocals.**

### Shorter tag line (if the UI wants something terse)
> Heroic 80s cartoon military march, instrumental, brass + synth + driving
> snare, building to a triumphant battle climax, ~60s, 132 BPM.

## Villain‑focused alt (darker cut)
> Ominous 1980s cartoon villain theme for a snake‑empire army, instrumental,
> minor key, hissing synth pads, low brass menace, tribal war drums and metallic
> percussion, rising to a sinister climax. Cinematic, tense, ~60s, no vocals.

## Auto‑score from the CLI
Once `ELEVENLABS_API_KEY` is set, generate + merge in one shot:

```bash
python -m comfymovies \
  "GI Joe style 1980s cartoon, American soldiers fight the evil Serpent Empire, cel animation 80s style" \
  --duration 60 --res 480p --quality --seed 1980 \
  --music-eleven "Heroic 1980s Saturday-morning cartoon action theme, fully instrumental, ~60s, triumphant brass fanfare and a bold march melody over driving military snare, 80s synth brass stabs and guitar power chords, rising to a full battle climax and a victorious final brass chord. Energetic, patriotic, cinematic. No vocals. ~132 BPM, D major."
```

If you generate a track by hand in the ElevenLabs UI, merge it directly:

```bash
python - <<'PY'
from comfymovies.music import merge_music
merge_music("output/gijoe_serpent_60s_quality.mp4",
            "path/to/your_elevenlabs_track.mp3",
            "output/gijoe_serpent_60s_scored.mp4")   # add keep_native=True to mix
PY
```
