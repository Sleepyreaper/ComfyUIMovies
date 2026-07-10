"""Export the image-edit graphs as ComfyUI UI + API workflow files.

Writes into ``workflows/``:
  flux2_image_edit.ui.json / .api.json
  qwen_image_edit_2511.api.json   (UI version is the official downloaded template)

Drag a ``.ui.json`` onto the ComfyUI canvas to load an editable node graph;
``.api.json`` is the /prompt payload (also loadable, auto-arranged). Then set the
LoadImage + instruction and Queue.

Usage:  PYTHONPATH=. python3 scripts/export_workflows.py
"""
import json
import os
import urllib.request

from comfymovies.edit import build_edit_workflow, build_qwen_edit_workflow
from comfymovies.to_ui import to_ui

HOST, PORT = "192.168.1.90", 8188
OUT = "workflows"


def object_info(classes):
    oi = {}
    for c in sorted(classes):
        url = f"http://{HOST}:{PORT}/object_info/{c}"
        oi.update(json.load(urllib.request.urlopen(url, timeout=20)))
    return oi


def main():
    os.makedirs(OUT, exist_ok=True)
    flux = build_edit_workflow(["your_input.png"], "describe your edit here",
                               1024, 1024, quality="fast")
    qwen = build_qwen_edit_workflow(["your_input.png"], "describe your edit here",
                                    quality="fast")

    json.dump(flux, open(f"{OUT}/flux2_image_edit.api.json", "w"), indent=2)
    json.dump(qwen, open(f"{OUT}/qwen_image_edit_2511.api.json", "w"), indent=2)

    classes = {n["class_type"] for n in flux.values()}
    oi = object_info(classes)
    ui = to_ui({str(k): v for k, v in flux.items()}, oi,
               title="FLUX.2 image edit (ComfyUIMovies)")
    json.dump(ui, open(f"{OUT}/flux2_image_edit.ui.json", "w"), indent=2)

    print("wrote:")
    for f in ("flux2_image_edit.ui.json", "flux2_image_edit.api.json",
              "qwen_image_edit_2511.api.json"):
        print("  ", os.path.join(OUT, f))
    print("(qwen UI = workflows/qwen_image_edit_2511.ui.json, the official template)")


if __name__ == "__main__":
    main()
