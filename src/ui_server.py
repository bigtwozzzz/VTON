import argparse
import cgi
import json
import mimetypes
import os
import shutil
import subprocess
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


PROJECT_ROOT = Path("/root/autodl-tmp/VTON")
CLOTH_INPUT_DIR = PROJECT_ROOT / "test_data" / "cloth_inputs"
OUTPUTS_DIR = PROJECT_ROOT / "test_data" / "outputs"
WORK_DIR = PROJECT_ROOT / "test_data" / "work"
PIPELINE_SCRIPT = PROJECT_ROOT / "src" / "run_tryon_pipeline.py"


def _ensure_dirs() -> None:
    CLOTH_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)


def _clear_dir_contents(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for p in path.iterdir():
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()


def _clear_images_only(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for p in path.iterdir():
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            p.unlink()


def _safe_ext(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext in {".jpg", ".jpeg", ".png", ".webp"}:
        return ext
    return ".jpg"


class AppState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.last_upload = None
        self.current_job_id = None
        self.jobs: dict[str, dict] = {}


STATE = AppState()


def _new_cloth_name(ext: str) -> tuple[str, str, str]:
    cloth_id = time.strftime("%Y%m%d%H%M%S") + uuid.uuid4().hex[:6]
    frame_id = "0001"
    filename = f"{cloth_id}_{frame_id}_back{ext}"
    return cloth_id, frame_id, filename


def _infer_output_cloth_id(filename: str) -> str:
    parts = Path(filename).stem.split("_")
    if len(parts) >= 1:
        return parts[0]
    return ""


def _list_images(path: Path) -> list[str]:
    if not path.exists():
        return []
    files: list[str] = []
    for p in path.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        name = p.name
        if name.startswith("cond_"):
            continue
        if "cond" in name or "parse" in name:
            continue
        files.append(name)
    files.sort()
    return files


def _find_cloth_folder(outputs_root: Path, cloth_id: str) -> Path:
    p = outputs_root / cloth_id
    if p.exists() and p.is_dir():
        return p
    if "_" in cloth_id:
        p2 = outputs_root / cloth_id.split("_", 1)[0]
        if p2.exists() and p2.is_dir():
            return p2
    return outputs_root


def _run_pipeline_async(
    job_id: str,
    wonder3d_env: str,
    vton360_env: str,
    args: dict,
) -> None:
    log_path = WORK_DIR / f"job_{job_id}.log"
    cmd = [
        "python",
        str(PIPELINE_SCRIPT),
        "--wonder3d-env",
        wonder3d_env,
        "--vton360-env",
        vton360_env,
        "--collar-module",
        str(args.get("collar_module", "none")),
        "--seam-module",
        str(args.get("seam_module", "none")),
        "--seam-band-width",
        str(int(args.get("seam_band_width", 24))),
        "--neckline-edge-ymax-scale",
        str(float(args.get("neckline_edge_ymax_scale", 0.60))),
        "--neckline-edge-depth-bonus",
        str(float(args.get("neckline_edge_depth_bonus", 0.45))),
        "--neckline-edge-depth-penalty",
        str(float(args.get("neckline_edge_depth_penalty", 0.02))),
        "--neckline-edge-slope-strength",
        str(float(args.get("neckline_edge_slope_strength", 0.8))),
        "--neckline-edge-slope-power",
        str(float(args.get("neckline_edge_slope_power", 1.2))),
        "--neckline-manual-x",
        str(float(args.get("neckline_manual_x", -1.0))),
        "--neckline-manual-y",
        str(float(args.get("neckline_manual_y", -1.0))),
    ]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PAGER"] = "cat"

    with open(log_path, "wb") as f:
        p = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
        )

    with STATE.lock:
        STATE.jobs[job_id]["pid"] = p.pid

    code = p.wait()
    with STATE.lock:
        STATE.jobs[job_id]["status"] = "done"
        STATE.jobs[job_id]["exit_code"] = code
        if STATE.current_job_id == job_id:
            STATE.current_job_id = None


class Handler(BaseHTTPRequestHandler):
    server_version = "VTONUIServer/0.1"

    def _send_json(self, status: int, data: dict) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, status: int, content_type: str, data: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._send_json(200, {"ok": True})
            return
        if parsed.path == "/api/uploaded_image":
            with STATE.lock:
                last = STATE.last_upload
            if not last:
                self._send_json(404, {"error": "no_upload"})
                return
            p = CLOTH_INPUT_DIR / last["filename"]
            if not p.exists():
                self._send_json(404, {"error": "not_found"})
                return
            ct = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
            self._send_bytes(200, ct, p.read_bytes())
            return

        if parsed.path == "/api/status":
            qs = parse_qs(parsed.query)
            job_id = (qs.get("job_id") or [""])[0]
            with STATE.lock:
                job = STATE.jobs.get(job_id)
            if not job:
                self._send_json(404, {"error": "job_not_found"})
                return
            log_path = WORK_DIR / f"job_{job_id}.log"
            tail = ""
            if log_path.exists():
                data = log_path.read_bytes()
                tail = data[-6000:].decode("utf-8", errors="replace")
            self._send_json(
                200,
                {
                    "job_id": job_id,
                    "status": job.get("status"),
                    "exit_code": job.get("exit_code"),
                    "log_tail": tail,
                    "cloth_id": job.get("cloth_id"),
                },
            )
            return

        if parsed.path == "/api/results":
            with STATE.lock:
                last = STATE.last_upload
                job_id = STATE.current_job_id
            if not last:
                self._send_json(200, {"front": [], "back": [], "cloth_id": ""})
                return
            cloth_id = last.get("output_cloth_id") or last.get("cloth_id") or ""
            front_root = OUTPUTS_DIR / "vton360_front"
            back_root = OUTPUTS_DIR / "vton360_back"
            front_dir = _find_cloth_folder(front_root, cloth_id)
            back_dir = _find_cloth_folder(back_root, cloth_id)
            self._send_json(
                200,
                {
                    "cloth_id": cloth_id,
                    "job_running": bool(job_id),
                    "front": _list_images(front_dir),
                    "back": _list_images(back_dir),
                },
            )
            return

        if parsed.path == "/api/image":
            qs = parse_qs(parsed.query)
            which = (qs.get("which") or ["front"])[0]
            name = (qs.get("name") or [""])[0]
            cloth_id = (qs.get("cloth_id") or [""])[0]
            if which not in {"front", "back"} or not name:
                self._send_json(400, {"error": "bad_request"})
                return
            root = OUTPUTS_DIR / ("vton360_front" if which == "front" else "vton360_back")
            p = root / cloth_id / name
            if not p.exists():
                p = root / name
            if not p.exists() or not p.is_file():
                self._send_json(404, {"error": "not_found"})
                return
            ct = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
            self._send_bytes(200, ct, p.read_bytes())
            return

        self._send_json(404, {"error": "not_found"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/upload":
            ctype, pdict = cgi.parse_header(self.headers.get("content-type") or "")
            if ctype != "multipart/form-data":
                self._send_json(400, {"error": "expected_multipart"})
                return
            pdict["boundary"] = pdict["boundary"].encode("utf-8")
            form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST"})
            if "file" not in form:
                self._send_json(400, {"error": "missing_file"})
                return
            file_item = form["file"]
            filename = getattr(file_item, "filename", "") or "upload.jpg"
            ext = _safe_ext(filename)

            _ensure_dirs()
            _clear_images_only(CLOTH_INPUT_DIR)
            _clear_dir_contents(OUTPUTS_DIR)
            _clear_dir_contents(WORK_DIR)

            cloth_id, frame_id, out_name = _new_cloth_name(ext)
            output_cloth_id = _infer_output_cloth_id(out_name)
            out_path = CLOTH_INPUT_DIR / out_name
            data = file_item.file.read()
            out_path.write_bytes(data)

            with STATE.lock:
                STATE.last_upload = {
                    "cloth_id": cloth_id,
                    "output_cloth_id": output_cloth_id,
                    "frame_id": frame_id,
                    "filename": out_name,
                }

            self._send_json(
                200,
                {"ok": True, "cloth_id": output_cloth_id, "frame_id": frame_id, "filename": out_name},
            )
            return

        if parsed.path == "/api/run":
            length = int(self.headers.get("content-length") or "0")
            body = self.rfile.read(length) if length > 0 else b"{}"
            try:
                payload = json.loads(body.decode("utf-8"))
            except Exception:
                self._send_json(400, {"error": "bad_json"})
                return

            with STATE.lock:
                if not STATE.last_upload:
                    self._send_json(400, {"error": "no_upload"})
                    return
                if STATE.current_job_id:
                    self._send_json(409, {"error": "job_running", "job_id": STATE.current_job_id})
                    return

                job_id = uuid.uuid4().hex
                STATE.current_job_id = job_id
                STATE.jobs[job_id] = {
                    "status": "running",
                    "exit_code": None,
                    "created_at": time.time(),
                    "cloth_id": STATE.last_upload.get("output_cloth_id") or STATE.last_upload.get("cloth_id"),
                }

            t = threading.Thread(
                target=_run_pipeline_async,
                args=(job_id, self.server.wonder3d_env, self.server.vton360_env, payload),
                daemon=True,
            )
            t.start()
            self._send_json(200, {"ok": True, "job_id": job_id})
            return

        self._send_json(404, {"error": "not_found"})


class Server(ThreadingHTTPServer):
    def __init__(self, server_address, RequestHandlerClass, wonder3d_env: str, vton360_env: str):
        super().__init__(server_address, RequestHandlerClass)
        self.wonder3d_env = wonder3d_env
        self.vton360_env = vton360_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--wonder3d-env", required=True)
    parser.add_argument("--vton360-env", required=True)
    args = parser.parse_args()

    _ensure_dirs()
    httpd = Server((args.host, args.port), Handler, args.wonder3d_env, args.vton360_env)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
