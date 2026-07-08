# THE FOREST FRIEND — audio (Hisaishi-style score + voice)

Companion audio for the 8-shot cinematic-anime short (~48-50s). Music is the
star; see the voice section for the dub decision.

---

## ElevenLabs Music prompt (cinematic-anime orchestral)

Paste into ElevenLabs Music, length **50000 ms**, instrumental. (Note: ElevenLabs
blocks named artists, so this describes the *style* rather than naming a composer.)

> An epic, tender orchestral film score inspired by classic cinematic anime
> music, fully instrumental, about 50 seconds, in a gentle 3/4 waltz feel, key of
> D major. A simple, memorable piano melody leads, joined by lush warm strings,
> soaring French horns, and delicate woodwinds (oboe and flute). Structure it as
> a mini score that follows a story of wonder and friendship: (0-12s) intimate
> solo piano, soft and curious, a sense of quiet magic awakening as a child
> enters an enchanted forest; (12-30s) warm strings and a lyrical oboe enter and
> the theme blooms, tender and hopeful as an unlikely friendship forms; (30-42s)
> the full orchestra swells into a soaring, magical crescendo, sweeping strings
> and bright horns, breathtaking awe and joy; (42-50s) it resolves back to a
> warm, gentle piano and strings, nostalgic and peaceful, fading softly like a
> sunset. Rich, cinematic, heartfelt, nostalgic, emotional. No drums, no
> percussion, no vocals, no electronic sounds.

**Shorter tagline (if the UI wants terse):**
> Cinematic-anime orchestral waltz, solo piano building to soaring strings and
> horns, wonder and friendship, ~50s, D major, instrumental, no vocals.

**Darker/awe alt (for the magic swell, shot 7):**
> Cinematic-anime orchestral crescendo, shimmering strings, glockenspiel and
> harp, choir-like "aahs" swelling with wonder, magical and vast, ~15s.

---

## Voice / dubbing — the honest call

Our shots are **not lip-synced** (the characters' mouths don't move to speech),
so putting *character dialogue* on them looks off — the classic AI-video tell.
Three good options that avoid that:

1. **Wordless (recommended, most Ghibli).** Many of Ghibli's best moments have no
   dialogue at all — just music + ambient. Let the score and visuals carry it.
2. **Storybook narration.** A warm narrator over the cutaways (no lip-sync needed,
   because we never hold on a talking mouth). ElevenLabs TTS, a gentle voice.
3. **Reactions + creature SFX only.** A soft gasp of wonder, a giggle, the
   spirit's chimes/coos — sparse, no full sentences.

Note: ElevenLabs **Dubbing** is for translating *existing* dialogue tracks; for
original narration we'd use ElevenLabs **Text-to-Speech**. Either way it's a
mergeable audio track via `comfymovies.post` (`polish(..., audio=track)`).

### Optional narration script (if you pick #2)
> "Deep in the old forest, where the light falls like gold... a girl found a
> door that no one else could see. And waiting there — small, and glowing, and
> kind — was a friend she would never forget."

Warm, unhurried, storybook cadence; leave gaps for the music to breathe.

---

## Assembly
Render the 8 shots -> concat in order -> lay the Hisaishi track under the whole
thing (and narration if chosen) -> optional light color grade. All via ffmpeg
(`comfymovies.post`), no extra models on the render box.
