"""Convert exported ComfyUI (UI/subgraph) workflows into the /prompt API graph.

ComfyUI's web UI exports a *graph* format (nodes + links, possibly nested in
subgraphs). The HTTP `/prompt` endpoint wants the *API* format: a flat dict
keyed by node id, each ``{"class_type": str, "inputs": {name: value|[src, slot]}}``.

This module reproduces what the frontend's "Save (API format)" does, driven by
the live ``/object_info`` schema so widget values map to the correct input names.
It flattens a single top-level subgraph instance (the shape the LTX-2 templates
ship in) so the result runs unmodified on the server.
"""
from __future__ import annotations

import copy
import json
from typing import Any

# Widget "control" values the frontend appends after seed/int widgets. These are
# UI-only and must be dropped when mapping widgets_values -> inputs.
_CONTROL_VALUES = {"fixed", "increment", "decrement", "randomize"}

# Input types that are wires (never widgets). Everything else (INT/FLOAT/STRING/
# BOOLEAN/COMBO list) is a widget carrying a value in widgets_values.
_CONNECTION_TYPES = {
    "MODEL", "CLIP", "VAE", "LATENT", "CONDITIONING", "IMAGE", "MASK",
    "AUDIO", "NOISE", "GUIDER", "SAMPLER", "SIGMAS", "CONTROL_NET",
    "STYLE_MODEL", "CLIP_VISION", "CLIP_VISION_OUTPUT", "UPSCALE_MODEL",
    "GLIGEN", "PHOTOMAKER", "WEBCAM", "VIDEO", "UPSCALE_MODEL",
    "LATENT_UPSCALE_MODEL", "LTXV_UPSAMPLER", "*",
}


def _is_widget_type(t: Any) -> bool:
    """Return True if a schema input type is a widget (carries a value)."""
    if isinstance(t, list):  # COMBO options
        return True
    if isinstance(t, str):
        return t not in _CONNECTION_TYPES
    return False


def _schema_inputs(object_info: dict, class_type: str) -> list[tuple[str, Any]]:
    """Ordered ``[(name, type), ...]`` for a class from /object_info."""
    spec = object_info.get(class_type)
    if not spec:
        return []
    out: list[tuple[str, Any]] = []
    inp = spec.get("input", {})
    for section in ("required", "optional"):
        for name, meta in inp.get(section, {}).items():
            out.append((name, meta[0] if isinstance(meta, list) else meta))
    return out


def _widget_names(object_info: dict, class_type: str) -> list[str]:
    """Names of widget inputs, in schema order (used to align widgets_values)."""
    return [n for n, t in _schema_inputs(object_info, class_type) if _is_widget_type(t)]


def _map_widget_values(object_info: dict, node: dict) -> dict[str, Any]:
    """Align a node's ``widgets_values`` array to its widget input names.

    Handles the UI quirk where a ``control_after_generate`` string is inserted
    right after seed-like INT widgets, producing more values than widget slots.
    """
    values = node.get("widgets_values")
    class_type = node["type"]
    names = _widget_names(object_info, class_type)
    result: dict[str, Any] = {}

    if isinstance(values, dict):
        # Some newer nodes store widgets as a dict already.
        return {k: v for k, v in values.items() if k in set(names)}
    if not isinstance(values, list):
        return result

    vi = 0
    for name in names:
        if vi >= len(values):
            break
        result[name] = values[vi]
        vi += 1
        # Skip a trailing control_after_generate token if present.
        if (
            vi < len(values)
            and isinstance(values[vi], str)
            and values[vi] in _CONTROL_VALUES
        ):
            vi += 1
    return result


def _build_link_index(links: list) -> dict[int, tuple[int, int]]:
    """Map link_id -> (origin_node_id, origin_slot) for both link encodings.

    Top-level links are arrays ``[id, origin, oslot, target, tslot, type]``;
    subgraph links are dicts ``{id, origin_id, origin_slot, ...}``.
    """
    idx: dict[int, tuple[int, int]] = {}
    for lk in links:
        if isinstance(lk, dict):
            idx[lk["id"]] = (lk["origin_id"], lk["origin_slot"])
        elif isinstance(lk, list) and len(lk) >= 5:
            idx[lk[0]] = (lk[1], lk[2])
    return idx


def _nodes_to_api(
    object_info: dict,
    nodes: list[dict],
    link_index: dict[int, tuple[int, int]],
    *,
    external: dict[int, Any] | None = None,
    skip_types: set[str] | None = None,
) -> dict[str, dict]:
    """Convert a flat list of graph nodes to API format.

    ``external`` maps a link id to a concrete value (used to resolve links that
    originate from a subgraph input node into a constant/default).
    """
    external = external or {}
    skip_types = skip_types or set()
    api: dict[str, dict] = {}

    for node in nodes:
        class_type = node["type"]
        if class_type in skip_types:
            continue
        inputs: dict[str, Any] = _map_widget_values(object_info, node)

        for slot in node.get("inputs", []):
            link = slot.get("link")
            name = slot.get("name")
            if link is None:
                continue
            if link in external:
                # Unconnected subgraph inputs resolve to None -> keep the node's
                # own widget default rather than clobbering it.
                if external[link] is not None:
                    inputs[name] = external[link]
            elif link in link_index:
                origin_id, origin_slot = link_index[link]
                inputs[name] = [str(origin_id), origin_slot]

        api[str(node["id"])] = {"class_type": class_type, "inputs": inputs}
    return api


def convert(workflow: dict, object_info: dict) -> dict:
    """Convert an exported UI workflow (with optional subgraphs) to API format.

    Supports the LTX-2 template shape: top-level SaveVideo + a single subgraph
    instance. Flattens the subgraph, resolving its unconnected external inputs
    to their default widget values so the graph runs standalone.
    """
    defs = workflow.get("definitions", {}).get("subgraphs", [])
    subgraphs = {sg["id"]: sg for sg in defs}

    top_nodes = workflow.get("nodes", [])
    top_links = _build_link_index(workflow.get("links", []))

    # Identify subgraph instance nodes at the top level.
    instances = [n for n in top_nodes if n["type"] in subgraphs]
    plain_top = [n for n in top_nodes if n["type"] not in subgraphs
                 and n["type"] != "MarkdownNote"]

    api: dict[str, dict] = {}

    # 1) Flatten each subgraph instance.
    for inst in instances:
        sg = subgraphs[inst["type"]]
        sg_links = _build_link_index(sg.get("links", []))

        # Resolve external subgraph inputs -> default values.
        # Each subgraph input has linkIds pointing at internal consumers; the
        # instance node may override via its own inputs, else use the input's
        # default widget value (absent -> None, which ComfyUI treats as unset).
        external: dict[int, Any] = {}
        inst_input_by_name = {
            i.get("name"): i for i in inst.get("inputs", [])
        }
        for sg_in in sg.get("inputs", []):
            label = sg_in.get("label") or sg_in.get("name")
            default = sg_in.get("widget", {}).get("value") if isinstance(
                sg_in.get("widget"), dict) else None
            inst_slot = inst_input_by_name.get(sg_in.get("name")) or \
                inst_input_by_name.get(label)
            if inst_slot and inst_slot.get("link") is not None and \
                    inst_slot["link"] in top_links:
                origin_id, origin_slot = top_links[inst_slot["link"]]
                value: Any = [str(origin_id), origin_slot]
            else:
                value = default
            for lid in sg_in.get("linkIds", []):
                external[lid] = value

        api.update(
            _nodes_to_api(
                object_info,
                sg.get("nodes", []),
                sg_links,
                external=external,
                skip_types={"MarkdownNote", "Note"},
            )
        )

        # Wire subgraph output(s) to top-level consumers. The instance node's
        # output slot i corresponds to the subgraph's outputs[i]; that output's
        # internal producer feeds any top node consuming the instance's output.
        sg_outputs = sg.get("outputs", [])
        for i, inst_out in enumerate(inst.get("outputs", [])):
            producer = None
            if i < len(sg_outputs):
                producer = _resolve_subgraph_output(sg, sg_links, sg_outputs[i])
            if producer is None:
                continue
            for lid in inst_out.get("links", []) or []:
                for tn in plain_top:
                    for slot in tn.get("inputs", []):
                        if slot.get("link") == lid:
                            _ensure_top_node(api, tn, object_info)
                            api[str(tn["id"])]["inputs"][slot["name"]] = producer

    # 2) Add remaining plain top-level nodes (e.g. SaveVideo) not yet added.
    for tn in plain_top:
        _ensure_top_node(api, tn, object_info)

    # 3) Resolve any still-unresolved top-level links between plain nodes.
    for tn in plain_top:
        node = api[str(tn["id"])]
        for slot in tn.get("inputs", []):
            name = slot.get("name")
            if name in node["inputs"]:
                continue
            link = slot.get("link")
            if link is not None and link in top_links:
                origin_id, origin_slot = top_links[link]
                node["inputs"][name] = [str(origin_id), origin_slot]

    return _inline_reroutes({k: v for k, v in api.items() if v is not None})


# Frontend-only passthrough nodes that the server cannot execute. Their single
# input is spliced directly into every consumer.
_PASSTHROUGH_TYPES = {"Reroute", "Reroute (rgthree)"}


def _inline_reroutes(api: dict[str, dict]) -> dict[str, dict]:
    """Remove Reroute passthrough nodes, rewiring consumers to the true source."""
    def source_of(node_id: str) -> Any:
        node = api.get(node_id)
        if not node:
            return None
        # A reroute has exactly one connection input; return its resolved source.
        for val in node["inputs"].values():
            if isinstance(val, list) and len(val) == 2:
                src = val[0]
                if api.get(src, {}).get("class_type") in _PASSTHROUGH_TYPES:
                    return source_of(src)
                return val
        return None

    reroutes = {nid for nid, n in api.items()
                if n["class_type"] in _PASSTHROUGH_TYPES}
    if not reroutes:
        return api

    resolved = {nid: source_of(nid) for nid in reroutes}
    out: dict[str, dict] = {}
    for nid, node in api.items():
        if nid in reroutes:
            continue
        new_inputs = {}
        for name, val in node["inputs"].items():
            if isinstance(val, list) and len(val) == 2 and val[0] in reroutes:
                new_inputs[name] = resolved.get(val[0])
            else:
                new_inputs[name] = val
        out[nid] = {"class_type": node["class_type"], "inputs": new_inputs}
    return out


def _ensure_top_node(api: dict, node: dict, object_info: dict) -> None:
    if api.get(str(node["id"])) is None:
        api[str(node["id"])] = {
            "class_type": node["type"],
            "inputs": _map_widget_values(object_info, node),
        }


def _resolve_subgraph_output(sg: dict, sg_links: dict, out: dict):
    """Find the internal producer [node_id, slot] feeding a subgraph output."""
    out_node = sg.get("outputNode", {})
    # The output node's inputs carry links from the real producer.
    target_name = out.get("name")
    for slot in out_node.get("inputs", []) if isinstance(out_node, dict) else []:
        if slot.get("name") == target_name and slot.get("link") in sg_links:
            oid, oslot = sg_links[slot["link"]]
            return [str(oid), oslot]
    # Fallback: match by the output's own linkIds against internal links.
    for lid in out.get("linkIds", []):
        if lid in sg_links:
            oid, oslot = sg_links[lid]
            return [str(oid), oslot]
    return None


def convert_file(path: str, object_info_path: str) -> dict:
    with open(path) as f:
        wf = json.load(f)
    with open(object_info_path) as f:
        oi = json.load(f)
    return convert(wf, oi)
