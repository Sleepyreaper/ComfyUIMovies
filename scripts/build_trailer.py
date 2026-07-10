"""EAGLE FORCE trailer — 80s post treatment + assembly (no GPU needed).

Turns the raw output/serpent/shotNN.mp4 clips into an addictive 80s cartoon
trailer:
  * per-shot 80s look: saturation pop, grain, scanlines, vignette
  * animated laser bolts (blue = heroes, red = serpent) overlaid per shot.fx
  * white/colored flash-ins on hero beats
  * chrome/neon 80s TITLE CARDS interspersed (villain tease, tagline, LOGO)
  * fast hard-cut assembly + a big logo reveal
Assets (title cards, laser bolts, scanlines) are generated with PIL/ffmpeg and
can be built offline before the render exists (use `assets` mode to preview).

Usage:
  python3 scripts/build_trailer.py assets    # build/preview overlays+cards only
  python3 scripts/build_trailer.py           # full assemble (needs shot clips)
"""
import json
import os
import subprocess
import sys
import tempfile

W, H, FPS = 1280, 704, 24
SHOTDIR = "output/serpent"
ASSETDIR = "output/serpent/assets"
STORYBOARD = "prompts/serpent.json"
FINAL = "output/eagle_force_trailer_silent.mp4"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Georgia Bold.ttf"
FONT_IMPACT = "/System/Library/Fonts/Supplemental/Impact.ttf"


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(" ".join(cmd[:6]) + " ...\n" + p.stderr[-900:])
    return p


# ---------------------------------------------------------------- assets (PIL)
def _font(path, size):
    from PIL import ImageFont
    for p in (path, FONT_BOLD, "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def make_laser_bolt(path, color, w=460, h=20):
    """A horizontal energy bolt PNG with a bright core + glow, transparent bg."""
    from PIL import Image, ImageDraw, ImageFilter
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cy = h // 2
    d.line([(6, cy), (w - 6, cy)], fill=color + (255,), width=6)
    glow = img.filter(ImageFilter.GaussianBlur(5))
    core = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    dc = ImageDraw.Draw(core)
    dc.line([(10, cy), (w - 10, cy)], fill=(255, 255, 255, 255), width=2)
    out = Image.alpha_composite(glow, core)
    out.save(path)


def make_scanlines(path):
    from PIL import Image, ImageDraw
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for y in range(0, H, 3):
        d.line([(0, y), (W, y)], fill=(0, 0, 0, 40), width=1)
    img.save(path)


def _chrome_text(d, text, font, cx, cy):
    """Draw metallic-chrome 80s title text centered at (cx, cy)."""
    from PIL import ImageFont
    bb = d.textbbox((0, 0), text, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    x, y = cx - tw // 2 - bb[0], cy - th // 2 - bb[1]
    # magenta/cyan offset shadows (chromatic 80s pop)
    for dx, dy, c in [(5, 5, (10, 10, 30)), (-3, -3, (255, 40, 180)), (3, 3, (40, 210, 255))]:
        d.text((x + dx, y + dy), text, font=font, fill=c)
    # chrome-ish fill (light top): approximate with near-white + gold underline
    d.text((x, y), text, font=font, fill=(245, 240, 210))
    return (x, y, tw, th)


def make_title_card(path, main, sub="", accent=(255, 200, 40), size=120):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (W, H), (6, 5, 14))
    d = ImageDraw.Draw(img)
    # retro horizon grid
    for i in range(1, 9):
        y = H // 2 + i * i * 3
        if y < H:
            d.line([(0, y), (W, y)], fill=(40, 10, 60), width=2)
    for gx in range(-6, 7):
        d.line([(W // 2 + gx * 120, H // 2 + 6), (W // 2 + gx * 420, H)],
               fill=(40, 10, 60), width=2)
    f = _font(FONT_IMPACT, size)
    x, y, tw, th = _chrome_text(d, main, f, W // 2, H // 2 - (20 if sub else 0))
    d.line([(W // 2 - tw // 2, y + th + 16), (W // 2 + tw // 2, y + th + 16)],
           fill=accent, width=5)
    if sub:
        fs = _font(FONT_BOLD, 40)
        sb = d.textbbox((0, 0), sub, font=fs)
        d.text((W // 2 - (sb[2] - sb[0]) // 2, y + th + 34), sub, font=fs, fill=(210, 210, 230))
    img.save(path)


def build_assets():
    os.makedirs(ASSETDIR, exist_ok=True)
    make_laser_bolt(f"{ASSETDIR}/bolt_blue.png", (80, 170, 255))
    make_laser_bolt(f"{ASSETDIR}/bolt_red.png", (255, 60, 60))
    make_scanlines(f"{ASSETDIR}/scanlines.png")
    make_title_card(f"{ASSETDIR}/card_tease.png", "THEY CAME FROM THE SHADOWS",
                    "the Serpent Empire has risen", accent=(120, 255, 120), size=70)
    make_title_card(f"{ASSETDIR}/card_tagline.png", "ONE FORCE STRIKES BACK",
                    "armed with the power of light", accent=(80, 170, 255), size=80)
    make_title_card(f"{ASSETDIR}/card_logo.png", "EAGLE FORCE",
                    "vs THE SERPENT EMPIRE", accent=(255, 200, 40), size=150)
    make_title_card(f"{ASSETDIR}/card_fall.png", "THIS FALL", "", accent=(255, 200, 40), size=110)
    print("assets ->", ASSETDIR)


# ------------------------------------------------------------- ffmpeg pipeline
LOOK = "eq=saturation=1.28:contrast=1.08:brightness=0.01,noise=alls=9:allf=t,vignette=PI/5"


def _card_clip(png, dst, dur, flash=True):
    fin = "fade=t=in:st=0:d=0.15" if flash else "fade=t=in:st=0:d=0.3"
    vf = f"fps={FPS},format=yuv420p,{fin},fade=t=out:st={dur-0.3:.2f}:d=0.3"
    run(["ffmpeg", "-y", "-v", "error", "-loop", "1", "-t", f"{dur}", "-i", png,
         "-vf", vf, "-r", str(FPS), "-c:v", "libx264", "-crf", "16",
         "-pix_fmt", "yuv420p", dst])


def _process_shot(src, dst, fx, scan, rescue=False):
    """Apply 80s look + flash + laser overlays to one shot clip.

    ``rescue`` adds a slow camera drift over a slightly enlarged frame so a
    STATIC/low-motion take never sits dead on screen (camera-on-cel feel).
    """
    blue, red = f"{ASSETDIR}/bolt_blue.png", f"{ASSETDIR}/bolt_red.png"
    inputs = ["-i", src, "-i", scan]
    pre = ""
    if rescue:
        zw, zh = (int(W * 1.14) // 2 * 2), (int(H * 1.14) // 2 * 2)
        drift = (f"scale={zw}:{zh},"
                 "crop=%d:%d:x='(in_w-out_w)/2+(in_w-out_w)/2*sin(t*0.55)':"
                 "y='(in_h-out_h)/2+(in_h-out_h)/2*sin(t*0.42)'," % (W, H))
        pre = drift
    fg = [f"[0:v]{pre}{LOOK}[base]", "[1:v]format=rgba,colorchannelmixer=aa=0.5[sl]",
          "[base][sl]overlay=0:0[v0]"]
    last = "v0"
    idx = 2
    bolts = []
    if fx in ("laser-blue", "laser-both"):
        bolts.append((blue, "L"))
    if fx in ("laser-red", "laser-both"):
        bolts.append((red, "R"))
    if fx == "laser-blue":
        bolts.append((blue, "R"))
    for j, (png, direction) in enumerate(bolts):
        inputs += ["-i", png]
        y = 180 + j * 220
        if direction == "L":
            x = f"-460+(t*{W+900})"
        else:
            x = f"{W}-(t*{W+900})"
        fg.append(f"[{idx}:v]format=rgba[b{j}]")
        fg.append(f"[{last}][b{j}]overlay=x='{x}':y={y}:enable='between(t,0,2.5)'[v{j+1}]")
        last = f"v{j+1}"
        idx += 1
    # flash-in: brief white lift at the head
    flash = "flash-in" in fx or "flash" in fx
    if flash:
        fg.append(f"[{last}]fade=t=in:st=0:d=0.12:color=white[vf]")
        last = "vf"
    filt = ";".join(fg)
    run(["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", filt,
         "-map", f"[{last}]", "-r", str(FPS), "-an", "-c:v", "libx264",
         "-crf", "16", "-pix_fmt", "yuv420p", dst])


# assembly order: interleave title cards with shot groups
SEQUENCE = [
    ("card", "card_tease.png", 1.6),
    ("shots", [0, 1, 2]),
    ("card", "card_tagline.png", 1.4),
    ("shots", [3, 4, 5, 6, 7]),
    ("shots", [8, 9, 10, 11, 12]),
    ("card", "card_fall.png", 1.2),
    ("shots", [13]),
    ("card", "card_logo.png", 2.6),
]


def assemble():
    sb = json.load(open(STORYBOARD))
    fx_by_id = {s["id"]: s.get("fx", "") for s in sb["shots"]}
    scan = f"{ASSETDIR}/scanlines.png"
    tmp = tempfile.mkdtemp(prefix="trailer_")
    clips = []
    n = 0
    for kind, *rest in SEQUENCE:
        if kind == "card":
            png, dur = f"{ASSETDIR}/{rest[0]}", rest[1]
            out = f"{tmp}/{n:02d}_card.mp4"
            _card_clip(png, out, dur, flash=True)
            clips.append(out); n += 1
        else:
            for sid in rest[0]:
                src = f"{SHOTDIR}/shot{sid:02d}.mp4"
                if not os.path.exists(src):
                    print("MISSING", src); continue
                out = f"{tmp}/{n:02d}_shot{sid:02d}.mp4"
                try:
                    from comfymovies.takes import score_clip, MOTION_LOW
                    sc = score_clip(src)
                    rescue = sc.motion < MOTION_LOW
                    if rescue:
                        print(f"  rescue shot{sid:02d} (motion {sc.motion} -> camera drift)")
                except Exception:
                    rescue = False
                _process_shot(src, out, fx_by_id.get(sid, ""), scan, rescue=rescue)
                clips.append(out); n += 1
    listf = f"{tmp}/list.txt"
    with open(listf, "w") as f:
        for c in clips:
            f.write(f"file '{c}'\n")
    run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", listf,
         "-c", "copy", FINAL])
    dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", FINAL], capture_output=True, text=True).stdout.strip()
    print(f"TRAILER -> {FINAL} ({dur}s)")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    build_assets()
    if mode != "assets":
        assemble()
