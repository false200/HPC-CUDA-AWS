"""Local dashboard for the CUDA Monte Carlo risk engine.

Binds to 127.0.0.1 only. Job files and results stay in this repository.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))

import jobtext  # noqa: E402

HOST = "127.0.0.1"
PORT = 8765

# us-east-1 Linux on-demand list prices, checked October 2026.
# Published non-tensor FP32 peaks: T4 8.1 TFLOPS, A10G 31.2 TFLOPS.
CLOUD = [
    {
        "instance": "g4dn.xlarge",
        "gpu": "NVIDIA T4",
        "usd_per_hour": 0.526,
        "fp32_tflops": 8.1,
    },
    {
        "instance": "g5.xlarge",
        "gpu": "NVIDIA A10G",
        "usd_per_hour": 1.006,
        "fp32_tflops": 31.2,
    },
]


def cloud_estimate(result: dict) -> dict:
    timing = result.get("timing") or {}
    gpu = result.get("gpu") or {}
    paths = float(result.get("paths") or 0)
    seconds = float(timing.get("gpu_job_seconds") or 0)
    local_tflops = float(gpu.get("fp32_tflops") or 0)
    peak_ok = 1.0 <= local_tflops <= 40.0
    rows = []
    for item in CLOUD:
        same = item["usd_per_hour"] * seconds / 3600.0
        row = {
            "instance": item["instance"],
            "gpu": item["gpu"],
            "usd_per_hour": item["usd_per_hour"],
            "fp32_tflops": item["fp32_tflops"],
            "seconds_at_this_gpu_speed": seconds,
            "usd_at_this_gpu_speed": same,
        }
        if peak_ok and paths > 0:
            scaled_seconds = seconds * (local_tflops / item["fp32_tflops"])
            row["seconds_if_peak_scales"] = scaled_seconds
            row["usd_if_peak_scales"] = item["usd_per_hour"] * scaled_seconds / 3600.0
            row["seconds_per_100m_if_peak_scales"] = scaled_seconds * (1e8 / paths)
            row["usd_per_100m_if_peak_scales"] = row["usd_if_peak_scales"] * (1e8 / paths)
        rows.append(row)
    return {
        "region": "us-east-1",
        "price_basis": "Linux on-demand list price",
        "local_fp32_tflops": local_tflops,
        "peak_scaling": peak_ok,
        "note": (
            "Dollar figures use the measured GPU job time on this machine. "
            "The peak-scaled column assumes throughput moves with published FP32 peak "
            "and is an estimate, not a run on AWS. Spot instances in the Batch template "
            "are usually cheaper than these on-demand rates and change through the day."
        ),
        "instances": rows,
    }


def engine_binary() -> Path:
    names = ("mc_engine.exe", "mc_engine") if os.name == "nt" else ("mc_engine", "mc_engine.exe")
    for name in names:
        path = ROOT / "build" / name
        if path.exists():
            return path
    return ROOT / "build" / names[0]


def find_nvrtc() -> Path | None:
    env = os.environ.get("NVRTC_PATH")
    if env and Path(env).is_file():
        return Path(env)
    root = ROOT / ".cuda_pkgs"
    if not root.exists():
        return None
    patterns = ("nvrtc64_*.dll", "libnvrtc.so*")
    for pattern in patterns:
        found = sorted(root.rglob(pattern))
        if found:
            return found[0]
    return None


def engine_env() -> dict:
    env = os.environ.copy()
    env["TEMP"] = str(ROOT / "tmp")
    env["TMP"] = str(ROOT / "tmp")
    env["PYTHONPYCACHEPREFIX"] = str(ROOT / ".cache" / "pyc")
    nvrtc = find_nvrtc()
    if nvrtc is not None:
        env["NVRTC_PATH"] = str(nvrtc)
    env["MC_KERNEL"] = str(ROOT / "kernels" / "mc_risk.cu")
    return env


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict) -> None:
        self._send(code, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/":
            page = (ROOT / "dashboard" / "index.html").read_bytes()
            self._send(200, page, "text/html; charset=utf-8")
            return
        if path == "/api/sample":
            self._json(200, jobtext.sample_portfolio())
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path != "/api/run":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0 or length > 1_000_000:
            self._json(400, {"error": "request body is missing or too large"})
            return
        raw = self.rfile.read(length)
        try:
            incoming = json.loads(raw.decode("utf-8"))
            job = jobtext.validate(incoming)
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._json(400, {"error": str(exc)})
            return
        engine = engine_binary()
        if not engine.exists():
            self._json(500, {"error": "The engine binary is missing. Run build.ps1 from the repository root."})
            return
        stamp = uuid.uuid4().hex[:12]
        work = ROOT / "tmp"
        work.mkdir(parents=True, exist_ok=True)
        job_path = work / f"job_{stamp}.txt"
        out_path = work / f"result_{stamp}.json"
        job_path.write_text(jobtext.to_job_text(job), encoding="utf-8")
        try:
            proc = subprocess.run(
                [str(engine), "run", str(job_path), str(out_path)],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=300,
                env=engine_env(),
            )
        except subprocess.TimeoutExpired:
            self._json(504, {"error": "the GPU job exceeded 300 seconds"})
            return
        if proc.returncode != 0 or not out_path.exists():
            detail = (proc.stderr or proc.stdout or "engine failed").strip()
            self._json(500, {"error": detail[-2000:]})
            return
        result = json.loads(out_path.read_text(encoding="utf-8"))
        result["cloud"] = cloud_estimate(result)
        result["log"] = (proc.stderr or "").strip()
        self._json(200, result)


def main() -> None:
    os.environ["PYTHONPYCACHEPREFIX"] = str(ROOT / ".cache" / "pyc")
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Monte Carlo risk desk at http://{HOST}:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
