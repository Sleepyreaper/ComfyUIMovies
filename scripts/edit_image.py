"""Edit an image with a plain-language instruction (FLUX.2 or Qwen-Image-Edit).

Usage:
  PYTHONPATH=. python3 scripts/edit_image.py INPUT.png "make it snow at night"
  PYTHONPATH=. python3 scripts/edit_image.py INPUT.png "turn her jacket red" OUT.png

  # multi-image reference (put subject of ref2 into the scene of the first):
  PYTHONPATH=. python3 scripts/edit_image.py scene.png "add the robot from the \
      second image standing on the left" --ref robot.png

Env / flags:
  ENGINE=flux2|qwen   (default flux2; qwen = Qwen-Image-Edit 2511 systms_action)
  QUALITY=fast|high   (fast=quick iterate; high=max fidelity)
  SEED=<int>          (default random)
  ACTION=<float>      (qwen only: QWEN_EDIT_ACTION_V1 LoRA strength, default 1.0;
                       set 0 for a static edit with no action bias)
  --ref FILE          (extra reference image, repeatable up to 2)

The result is downloaded next to the input as INPUT.edited.png unless an output
path is given.
"""
import os
import random
import sys

from PIL import Image

from comfymovies.comfy import ComfyClient
from comfymovies.edit import build_edit_workflow, build_qwen_edit_workflow

HOST, PORT = "192.168.1.90", 8188


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    src = argv[0]
    instruction = argv[1]
    refs = []
    out = None
    i = 2
    while i < len(argv):
        if argv[i] == "--ref":
            refs.append(argv[i + 1]); i += 2
        else:
            out = argv[i]; i += 1
    if not os.path.exists(src):
        print(f"input not found: {src}"); return 1
    if out is None:
        base, _ = os.path.splitext(src)
        out = f"{base}.edited.png"

    engine = os.environ.get("ENGINE", "flux2").lower()
    quality = os.environ.get("QUALITY", "fast")
    seed = int(os.environ.get("SEED", random.randint(1, 2**31 - 1)))

    with Image.open(src) as im:
        width, height = im.size

    c = ComfyClient(host=HOST, port=PORT)
    print(f"uploading {src} ...", flush=True)
    names = [c.upload_image(src, name=f"edit_src_{os.path.basename(src)}")]
    for r in refs[:2]:
        names.append(c.upload_image(r, name=f"edit_ref_{os.path.basename(r)}"))

    if engine == "qwen":
        action = float(os.environ.get("ACTION", "1.0"))
        g = build_qwen_edit_workflow(names, instruction, quality=quality,
                                     seed=seed, action_strength=action)
    else:
        g = build_edit_workflow(names, instruction, width, height,
                                quality=quality, seed=seed)
    print(f"editing [{engine}/{quality}] seed={seed}: {instruction!r}", flush=True)
    pid = c.submit(g)
    entry = c.wait(pid, timeout=900)
    outs = c.find_outputs(entry)
    imgs = [f for f in outs if f["filename"].lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
    if not imgs:
        print("no image produced"); return 1
    c.download(imgs[0], out)
    print(f"saved -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
