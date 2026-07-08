"""Minimal ComfyUI HTTP client: submit a graph, wait, download the result."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass


class ComfyError(RuntimeError):
    """Raised when the server rejects a prompt or execution fails."""


@dataclass
class ComfyClient:
    host: str = "192.168.1.90"
    port: int = 8188
    client_id: str = ""

    def __post_init__(self) -> None:
        self.client_id = self.client_id or uuid.uuid4().hex

    @property
    def base(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _get(self, path: str, timeout: float = 30) -> bytes:
        with urllib.request.urlopen(self.base + path, timeout=timeout) as r:
            return r.read()

    def system_stats(self) -> dict:
        return json.loads(self._get("/system_stats", timeout=10))

    def submit(self, graph: dict) -> str:
        """Queue a graph; return the prompt_id. Raises on validation errors."""
        payload = json.dumps(
            {"prompt": graph, "client_id": self.client_id}
        ).encode()
        req = urllib.request.Request(
            self.base + "/prompt", data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())["prompt_id"]
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            try:
                j = json.loads(detail)
                msg = j.get("error", {}).get("message", detail)
                errs = j.get("node_errors", {})
                if errs:
                    msg += " :: " + json.dumps(errs)[:800]
            except json.JSONDecodeError:
                msg = detail
            raise ComfyError(f"Prompt rejected ({e.code}): {msg}") from None

    def wait(
        self, prompt_id: str, *, timeout: float = 3600, poll: float = 2.0,
        on_progress=None,
    ) -> dict:
        """Block until the prompt finishes; return its history entry."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            hist = json.loads(
                self._get(f"/history/{prompt_id}", timeout=30) or b"{}"
            )
            entry = hist.get(prompt_id)
            if entry:
                status = entry.get("status", {})
                if status.get("completed") or status.get(
                    "status_str"
                ) in ("success", "error"):
                    if status.get("status_str") == "error":
                        raise ComfyError(
                            "Execution failed: "
                            + json.dumps(status.get("messages", []))[:1000]
                        )
                    return entry
            if on_progress:
                try:
                    q = json.loads(self._get("/queue", timeout=10))
                    on_progress(q)
                except Exception:
                    pass
            time.sleep(poll)
        raise ComfyError(f"Timed out after {timeout}s waiting for {prompt_id}")

    @staticmethod
    def find_outputs(entry: dict) -> list[dict]:
        """Collect saved video/image/gif file descriptors from a history entry."""
        files: list[dict] = []
        for node_out in entry.get("outputs", {}).values():
            for key in ("videos", "gifs", "images"):
                for f in node_out.get(key, []):
                    if isinstance(f, dict) and f.get("filename"):
                        files.append(f)
        return files

    def download(self, file_desc: dict, dest_path: str) -> str:
        """Download one output file descriptor to ``dest_path``."""
        params = urllib.parse.urlencode({
            "filename": file_desc["filename"],
            "subfolder": file_desc.get("subfolder", ""),
            "type": file_desc.get("type", "output"),
        })
        data = self._get(f"/view?{params}", timeout=300)
        with open(dest_path, "wb") as f:
            f.write(data)
        return dest_path

    def interrupt(self) -> None:
        req = urllib.request.Request(self.base + "/interrupt", data=b"")
        try:
            urllib.request.urlopen(req, timeout=10)
        except Exception:
            pass

    def free(self, unload_models: bool = True, free_memory: bool = True) -> None:
        """Ask ComfyUI to unload models / free VRAM (only acts when idle).

        Useful before a heavy (e.g. high-res) render so the job starts with a
        clean VRAM slate instead of competing with cached models from a prior
        run — important for fitting WAN's two 14B experts on a 32GB card.
        """
        payload = json.dumps({
            "unload_models": unload_models, "free_memory": free_memory,
        }).encode()
        req = urllib.request.Request(
            self.base + "/free", data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(req, timeout=15)
        except Exception:
            pass

    def vram_free_gb(self) -> float:
        """Current free VRAM in GB on the first CUDA device (0 on failure)."""
        try:
            d = self.system_stats()
            return d["devices"][0]["vram_free"] / 1e9
        except Exception:
            return 0.0

    def cancel(self, prompt_id: str) -> None:
        """Cancel a specific queued/running prompt without touching others.

        Removes it from the pending queue by id; only issues a global
        ``/interrupt`` if that exact prompt is the one currently running (so a
        dry-run never kills an unrelated render already in progress).
        """
        payload = json.dumps({"delete": [prompt_id]}).encode()
        req = urllib.request.Request(
            self.base + "/queue", data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(req, timeout=10)
        except Exception:
            pass
        try:
            q = json.loads(self._get("/queue", timeout=10))
            running = [item[1] for item in q.get("queue_running", [])]
            if running == [prompt_id]:
                self.interrupt()
        except Exception:
            pass
