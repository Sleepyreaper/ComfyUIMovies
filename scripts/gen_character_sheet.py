from comfymovies.comfy import ComfyClient
from comfymovies import film

SHEET_PROMPT = (
    "Studio Ghibli style anime character reference sheet, plain soft cream studio background, "
    "full body, two characters standing side by side facing forward: "
    "LEFT, a young girl about ten years old with a short glossy black bob haircut and a small red side ribbon, "
    "big curious dark eyes, round friendly face, wearing a pale blue short-sleeve dress with a white sailor collar "
    "and red rubber boots; "
    "RIGHT, a small round fluffy pure white forest spirit the size of a melon, perfectly round soft puffball body, "
    "two big round gentle black eyes, tiny rosy pink cheeks, two little green leaf-shaped ears on top, "
    "no wings, no arms, no tail, softly glowing white; "
    "clean flat even lighting, crisp delicate anime linework, painterly, wholesome, model sheet, masterpiece"
)
W, H = 1216, 832
c = ComfyClient(host="192.168.1.90", port=8188)

def build(seed):
    g = {}
    def node(nid, ct, inp):
        g[nid] = {"class_type": ct, "inputs": inp}; return nid
    img = film._flux_keyframe(node, f"sheet{seed}", SHEET_PROMPT, seed, W, H)
    node("save", "SaveImage", {"images": img, "filename_prefix": f"ff_charsheet_{seed}"})
    return g

pids = {}
for seed in (1111, 2222, 3333):
    pids[seed] = c.submit(build(seed))
    print("queued seed", seed, "->", pids[seed])
import json
open("/tmp/sheet_pids.json","w").write(json.dumps(pids))
