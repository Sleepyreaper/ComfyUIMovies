import json, sys
from comfymovies.film import FilmSpec, Shot, build_keyframe_workflow, build_shot_workflow
from comfymovies.comfy import ComfyClient

def load_spec(path, prefix, ref="ff_ref.png"):
    d = json.load(open(path))
    shots = [Shot(keyframe=s["keyframe"], motion=s["motion"],
                  seconds=s.get("seconds", 6.0), seed=s.get("seed", 0))
             for s in d["shots"]]
    return FilmSpec(shots=shots, character=d.get("character",""), style=d.get("style",""),
                    width=d.get("width") or 1280, height=d.get("height") or 720,
                    fps=d.get("fps") or 16, prefix=prefix, reference_image=ref)

if __name__ == "__main__":
    mode = sys.argv[1]            # "kf" or "shot"
    idxs = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv)>2 else list(range(8))
    prefix = sys.argv[3] if len(sys.argv)>3 else "ComfyUIMovies/ffc"
    spec = load_spec("prompts/forest_friend.json", prefix)
    c = ComfyClient(host="192.168.1.90", port=8188)
    build = build_keyframe_workflow if mode=="kf" else build_shot_workflow
    pids = {}
    for i in idxs:
        pids[i] = c.submit(build(spec, i))
        print(f"queued {mode} shot{i:02d} -> {pids[i]}")
    json.dump(pids, open(f"/tmp/ff_{mode}_pids.json","w"))
