"""Compose up to 3 reference images into ONE new image via a prompt.

Qwen-Image-Edit 2511 reads all three references (image1/image2/image3) and
generates a single image following your instruction — e.g. put the character
from one image and the character from another into the scene/location of a third.

Usage:
  PYTHONPATH=. python3 scripts/compose_images.py \
      city.png hero.png villain.png \
      "the soldier from image 2 and the cobra villain from image 3 face off in \
       this city street, 1980s cartoon cel style"

  # 2 images also fine (drop the third):
  PYTHONPATH=. python3 scripts/compose_images.py scene.png robot.png \
      "add the robot from image 2 standing in this scene"  -o out.png

Positional args:  IMG1 [IMG2 [IMG3]] "INSTRUCTION"
  IMG1 is the canvas/base (its aspect ratio drives the output); refer to the
  others in the prompt as "image 2" / "image 3".

Flags / env:
  -o OUT            output path (default: <IMG1 dir>/composed.png)
  QUALITY=fast|high (fast = 4-step; high = 40-step, sharper composite)
  SEED=<int>        (default random)
  ACTION=<float>    QWEN_EDIT_ACTION_V1 LoRA strength (default 1.0; 0 = static)
"""
import os
import random
import sys

from comfymovies.comfy import ComfyClient
from comfymovies.edit import build_qwen_edit_workflow

HOST, PORT = "192.168.1.90", 8188


def main(argv):
    # Separate output flag, image paths, and the trailing instruction.
    out = None
    args = []
    i = 0
    while i < len(argv):
        if argv[i] in ("-o", "--out"):
            out = argv[i + 1]; i += 2
        else:
            args.append(argv[i]); i += 1

    if len(args) < 2:
        print(__doc__)
        return 1

    instruction = args[-1]
    images = args[:-1]
    if len(images) > 3:
        print("at most 3 images are supported")
        return 1
    missing = [p for p in images if not os.path.exists(p)]
    if missing:
        print("input(s) not found: " + ", ".join(missing))
        return 1

    if out is None:
        out = os.path.join(os.path.dirname(images[0]) or ".", "composed.png")

    quality = os.environ.get("QUALITY", "fast")
    seed = int(os.environ.get("SEED", random.randint(1, 2**31 - 1)))
    action = float(os.environ.get("ACTION", "1.0"))

    c = ComfyClient(host=HOST, port=PORT)
    names = []
    for n, p in enumerate(images, start=1):
        print(f"uploading image {n}: {p}", flush=True)
        names.append(c.upload_image(p, name=f"compose_{n}_{os.path.basename(p)}"))

    g = build_qwen_edit_workflow(names, instruction, quality=quality, seed=seed,
                                 action_strength=action)
    print(f"composing [{len(names)} imgs / {quality}] seed={seed}: {instruction!r}",
          flush=True)
    pid = c.submit(g)
    entry = c.wait(pid, timeout=1200)
    outs = c.find_outputs(entry)
    imgs = [f for f in outs
            if f["filename"].lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
    if not imgs:
        print("no image produced")
        return 1
    c.download(imgs[0], out)
    print(f"saved -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
