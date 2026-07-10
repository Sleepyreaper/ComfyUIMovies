"""Offline round-trip test: to_ui(api) -> convert() reproduces the api graph."""
from comfymovies.convert import convert
from comfymovies.to_ui import to_ui

# Minimal synthetic /object_info covering a loader -> encode -> sampler -> save
# chain, including a seed widget (control_after_generate) and an optional widget.
OI = {
    "UNETLoader": {
        "input": {"required": {
            "unet_name": [["a.safetensors", "b.safetensors"], {}],
            "weight_dtype": [["default", "fp8"], {}],
        }},
        "output": ["MODEL"],
    },
    "CLIPTextEncode": {
        "input": {"required": {
            "text": ["STRING", {"default": ""}],
            "clip": ["CLIP", {}],
        }},
        "output": ["CONDITIONING"],
    },
    "CLIPLoader": {
        "input": {"required": {
            "clip_name": [["x.safetensors"], {}],
            "type": [["flux2"], {}],
            "device": [["default"], {}],
        }},
        "output": ["CLIP"],
    },
    "KSampler": {
        "input": {
            "required": {
                "model": ["MODEL", {}],
                "positive": ["CONDITIONING", {}],
                "seed": ["INT", {"default": 0, "control_after_generate": True}],
                "steps": ["INT", {"default": 20}],
                "cfg": ["FLOAT", {"default": 7.0}],
            },
            "optional": {"denoise": ["FLOAT", {"default": 1.0}]},
        },
        "output": ["LATENT"],
    },
    "SaveImage": {
        "input": {"required": {
            "images": ["IMAGE", {}],
            "filename_prefix": ["STRING", {"default": "ComfyUI"}],
        }},
        "output": [],
    },
}


def _api():
    return {
        "unet": {"class_type": "UNETLoader",
                 "inputs": {"unet_name": "a.safetensors", "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": "x.safetensors", "type": "flux2", "device": "default"}},
        "pos": {"class_type": "CLIPTextEncode",
                "inputs": {"text": "hello", "clip": ["clip", 0]}},
        "ks": {"class_type": "KSampler",
               "inputs": {"model": ["unet", 0], "positive": ["pos", 0],
                          "seed": 42, "steps": 8, "cfg": 3.0, "denoise": 1.0}},
    }


def _iso(a, b):
    """Structural signature multiset equality (ignores id renaming)."""
    def sig(api):
        memo = {}
        def ns(k):
            if k in memo:
                return memo[k]
            memo[k] = "<c>"
            n = api[str(k)]
            parts = []
            for name in sorted(n["inputs"]):
                v = n["inputs"][name]
                if isinstance(v, list) and len(v) == 2 and str(v[0]) in api:
                    parts.append(f"{name}=>({ns(str(v[0]))})[{v[1]}]")
                else:
                    parts.append(f"{name}={v!r}")
            memo[k] = f"{n['class_type']}|" + ",".join(parts)
            return memo[k]
        return sorted(ns(k) for k in api)
    a = {str(k): v for k, v in a.items()}
    b = {str(k): v for k, v in b.items()}
    return sig(a) == sig(b)


def test_round_trip_is_isomorphic():
    api = _api()
    ui = to_ui(api, OI)
    back = convert(ui, OI)
    assert _iso(api, back)


def test_ui_has_positions_and_links():
    ui = to_ui(_api(), OI)
    assert len(ui["nodes"]) == 4
    assert all("pos" in n and len(n["pos"]) == 2 for n in ui["nodes"])
    # three wires: unet->ks, clip->pos, pos->ks
    assert ui["last_link_id"] == len(ui["links"]) == 3


def test_seed_widget_gets_control_after_generate_token():
    ui = to_ui(_api(), OI)
    ks = [n for n in ui["nodes"] if n["type"] == "KSampler"][0]
    # widgets: seed, <control>, steps, cfg, denoise
    assert ks["widgets_values"][0] == 42
    assert ks["widgets_values"][1] == "fixed"
    assert 8 in ks["widgets_values"] and 3.0 in ks["widgets_values"]
