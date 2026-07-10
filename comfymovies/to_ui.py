"""Convert a /prompt API graph back into an exported ComfyUI *UI* workflow.

The API format (flat ``{id: {class_type, inputs}}``) runs on the server but does
NOT open as a node graph in the web UI. This module reconstructs a UI/graph
export (nodes with positions + a global ``links`` array + widgets_values) so a
generated workflow can be *loaded and edited in the ComfyUI canvas*, exactly
like a downloaded template.

It is the inverse of :mod:`comfymovies.convert` and is schema-driven off the
same ``/object_info``. Round-trip ``convert(to_ui(api)) == api`` is asserted by
the tests, which guarantees the emitted UI graph is wiring-faithful.
"""
from __future__ import annotations

from typing import Any

from .convert import _is_widget_type, _schema_inputs

_CONTROL_DEFAULT = "fixed"


def _widget_inputs(object_info: dict, class_type: str) -> list[tuple[str, Any]]:
    return [(n, t) for n, t in _schema_inputs(object_info, class_type)
            if _is_widget_type(t)]


def _connection_inputs(object_info: dict, class_type: str) -> list[tuple[str, Any]]:
    return [(n, t) for n, t in _schema_inputs(object_info, class_type)
            if not _is_widget_type(t)]


def _has_control_after_generate(object_info: dict, class_type: str, name: str) -> bool:
    spec = object_info.get(class_type, {}).get("input", {})
    for section in ("required", "optional"):
        meta = spec.get(section, {}).get(name)
        if isinstance(meta, list) and len(meta) > 1 and isinstance(meta[1], dict):
            return bool(meta[1].get("control_after_generate"))
    return False


def _widget_default(object_info: dict, class_type: str, name: str, t: Any) -> Any:
    spec = object_info.get(class_type, {}).get("input", {})
    for section in ("required", "optional"):
        meta = spec.get(section, {}).get(name)
        if isinstance(meta, list):
            if len(meta) > 1 and isinstance(meta[1], dict) and "default" in meta[1]:
                return meta[1]["default"]
            if isinstance(meta[0], list) and meta[0]:  # COMBO -> first option
                return meta[0][0]
    if isinstance(t, list) and t:
        return t[0]
    return ""


def _output_types(object_info: dict, class_type: str) -> list[str]:
    spec = object_info.get(class_type, {})
    outs = spec.get("output", [])
    return list(outs)


def to_ui(api: dict[str, dict], object_info: dict, *,
          title: str = "ComfyUIMovies workflow") -> dict:
    """Build a UI/graph workflow dict from an API graph.

    Nodes are laid out left-to-right by topological depth. Returns a dict ready
    to ``json.dump`` and drag onto the ComfyUI canvas.
    """
    # Stable integer ids (sorted so layout is deterministic).
    ids = list(api.keys())
    nid = {k: i + 1 for i, k in enumerate(ids)}

    # Topological depth for x-position (longest path from a source).
    depth: dict[str, int] = {}

    def compute_depth(k: str, seen: frozenset[str]) -> int:
        if k in depth:
            return depth[k]
        if k in seen:
            return 0
        d = 0
        for v in api[k]["inputs"].values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) \
                    and v[0] in api:
                d = max(d, 1 + compute_depth(v[0], seen | {k}))
        depth[k] = d
        return d

    for k in ids:
        compute_depth(k, frozenset())

    # Group nodes per depth for vertical stacking.
    per_depth: dict[int, list[str]] = {}
    for k in ids:
        per_depth.setdefault(depth[k], []).append(k)

    nodes = []
    links = []
    link_seq = 0
    # Track producers' output "links" lists to fill after we know consumers.
    out_links: dict[tuple[str, int], list[int]] = {}

    # First pass: build nodes + input slots + links.
    node_by_key: dict[str, dict] = {}
    for k in ids:
        ct = api[k]["class_type"]
        d = depth[k]
        row = per_depth[d].index(k)
        x = 60 + d * 320
        y = 60 + row * 260

        conn_inputs = _connection_inputs(object_info, ct)
        input_slots = []
        for name, t in conn_inputs:
            slot: dict[str, Any] = {"name": name, "type": t, "link": None}
            val = api[k]["inputs"].get(name)
            if isinstance(val, list) and len(val) == 2 and isinstance(val[0], str):
                src, sslot = val
                link_seq += 1
                slot["link"] = link_seq
                links.append([link_seq, nid[src], sslot, nid[k],
                              len(input_slots), t])
                out_links.setdefault((src, sslot), []).append(link_seq)
            input_slots.append(slot)

        # widgets_values in schema widget order (+ control token for seeds).
        wvals: list[Any] = []
        for name, t in _widget_inputs(object_info, ct):
            if name in api[k]["inputs"] and not (
                isinstance(api[k]["inputs"][name], list)
                and len(api[k]["inputs"][name]) == 2
                and isinstance(api[k]["inputs"][name][0], str)
            ):
                wvals.append(api[k]["inputs"][name])
            else:
                wvals.append(_widget_default(object_info, ct, name, t))
            if _has_control_after_generate(object_info, ct, name):
                wvals.append(_CONTROL_DEFAULT)

        out_types = _output_types(object_info, ct)
        node = {
            "id": nid[k],
            "type": ct,
            "pos": [x, y],
            "size": [270, 120],
            "flags": {},
            "order": 0,
            "mode": 0,
            "inputs": input_slots,
            "outputs": [
                {"name": ot, "type": ot, "slot_index": i, "links": []}
                for i, ot in enumerate(out_types)
            ],
            "properties": {"Node name for S&R": ct},
            "widgets_values": wvals,
        }
        node_by_key[k] = node
        nodes.append(node)

    # Second pass: fill each producer output slot's links list + order.
    for (src, sslot), lids in out_links.items():
        outs = node_by_key[src]["outputs"]
        if sslot < len(outs):
            outs[sslot]["links"] = lids
    for i, k in enumerate(sorted(ids, key=lambda z: (depth[z], per_depth[depth[z]].index(z)))):
        node_by_key[k]["order"] = i

    return {
        "last_node_id": len(ids),
        "last_link_id": link_seq,
        "nodes": nodes,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {"title": title},
        "version": 0.4,
    }
