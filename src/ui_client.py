import argparse
import socket
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>VTON 客户端</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Noto Sans", "PingFang SC", "Microsoft YaHei"; margin: 18px; }
    .row { display: flex; gap: 16px; flex-wrap: wrap; align-items: center; }
    .card { border: 1px solid #ddd; border-radius: 10px; padding: 12px; max-width: 960px; }
    label { display: block; font-size: 13px; color: #333; }
    input[type="text"] { width: 420px; padding: 6px 8px; }
    input[type="range"] { width: 320px; }
    select { padding: 6px 8px; }
    button { padding: 8px 12px; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New"; }
    #img { max-width: 900px; max-height: 650px; border: 1px solid #eee; }
    #log { white-space: pre-wrap; font-size: 12px; max-height: 220px; overflow: auto; background: #111; color: #ddd; padding: 10px; border-radius: 10px; }
  </style>
</head>
<body>
  <h2>VTON 可视化客户端</h2>

  <div class="card">
    <div class="row">
      <div>
        <label>服务端地址（建议用 SSH 端口转发后填本地地址）</label>
        <input id="serverUrl" type="text" value="http://127.0.0.1:8766" />
      </div>
      <div>
        <label>上传衣服图片</label>
        <input id="file" type="file" accept="image/*" />
      </div>
      <div>
        <label>&nbsp;</label>
        <button id="uploadBtn">上传并清空历史</button>
      </div>
      <div>
        <label>&nbsp;</label>
        <button id="runBtn" disabled>试穿</button>
      </div>
    </div>
    <div class="row" style="margin-top: 10px;">
      <div><span class="mono" id="uploadInfo"></span></div>
    </div>
    <div class="row" style="margin-top: 10px;" id="manualPickSection">
      <div>
        <label>点击选择前领口最低点（仅 manual_point 生效）</label>
        <div style="position: relative; display: inline-block;">
          <img id="uploadPreview" style="max-width: 520px; max-height: 360px; border: 1px solid #eee;" />
          <div id="pickDot" style="position:absolute; width:12px; height:12px; border:3px solid #ff0; border-radius:50%; transform: translate(-50%,-50%); display:none;"></div>
        </div>
        <div class="mono" id="pickInfo"></div>
      </div>
    </div>
  </div>

  <div class="card" style="margin-top: 14px;">
    <div class="row">
      <div>
        <label>领口模块</label>
        <select id="collarModule">
          <option value="none">none</option>
          <option value="neckline_edge" selected>neckline_edge</option>
          <option value="neckline_cut">neckline_cut</option>
          <option value="cut_top_bump">cut_top_bump</option>
          <option value="manual_point">manual_point</option>
        </select>
      </div>
      <div>
        <label>接缝模块</label>
        <select id="seamModule">
          <option value="none">none</option>
          <option value="side_views" selected>side_views</option>
          <option value="feather_stats">feather_stats</option>
        </select>
      </div>
      <div id="seamParams">
        <label>seam_band_width: <span id="seamBandWidthVal" class="mono"></span></label>
        <input id="seamBandWidth" type="range" min="8" max="64" step="1" value="24"/>
      </div>
    </div>

    <div class="row" style="margin-top: 10px;" id="necklineEdgeParams1">
      <div>
        <label>neckline_edge_ymax_scale: <span id="ymaxVal" class="mono"></span></label>
        <input id="ymax" type="range" min="0.30" max="0.90" step="0.01" value="0.60"/>
      </div>
      <div>
        <label>depth_bonus: <span id="bonusVal" class="mono"></span></label>
        <input id="bonus" type="range" min="0.00" max="1.20" step="0.01" value="0.45"/>
      </div>
      <div>
        <label>depth_penalty: <span id="penaltyVal" class="mono"></span></label>
        <input id="penalty" type="range" min="0.00" max="0.40" step="0.01" value="0.02"/>
      </div>
    </div>

    <div class="row" style="margin-top: 10px;" id="necklineEdgeParams2">
      <div>
        <label>slope_strength: <span id="slopeStrengthVal" class="mono"></span></label>
        <input id="slopeStrength" type="range" min="0.00" max="2.00" step="0.01" value="0.80"/>
      </div>
      <div>
        <label>slope_power: <span id="slopePowerVal" class="mono"></span></label>
        <input id="slopePower" type="range" min="0.60" max="3.00" step="0.01" value="1.20"/>
      </div>
    </div>
  </div>

  <div class="card" style="margin-top: 14px;">
    <div class="row">
      <button id="toggleFB" disabled>前后切换</button>
      <button id="left" disabled>左旋转</button>
      <button id="right" disabled>右旋转</button>
      <span class="mono" id="viewInfo"></span>
    </div>
    <div class="row" style="margin-top: 10px;">
      <img id="img" />
    </div>
  </div>

  <div class="card" style="margin-top: 14px;">
    <div class="row"><strong>日志（尾部）</strong></div>
    <div id="log"></div>
  </div>

<script>
  const el = (id) => document.getElementById(id);
  const setTxt = (id, v) => el(id).textContent = v;
  const asNum = (id) => Number(el(id).value);

  function bindRange(rangeId, labelId) {
    const r = el(rangeId);
    const update = () => setTxt(labelId, r.value);
    r.addEventListener("input", update);
    update();
  }

  bindRange("seamBandWidth", "seamBandWidthVal");
  bindRange("ymax", "ymaxVal");
  bindRange("bonus", "bonusVal");
  bindRange("penalty", "penaltyVal");
  bindRange("slopeStrength", "slopeStrengthVal");
  bindRange("slopePower", "slopePowerVal");

  let jobId = "";
  let clothId = "";
  let which = "front";
  let idx = 0;
  let filesFront = [];
  let filesBack = [];
  let pickX = -1.0;
  let pickY = -1.0;

  function updateVisibility() {
    const collar = el("collarModule").value;
    const seam = el("seamModule").value;

    el("manualPickSection").style.display = (collar === "manual_point") ? "flex" : "none";
    el("necklineEdgeParams1").style.display = (collar === "neckline_edge") ? "flex" : "none";
    el("necklineEdgeParams2").style.display = (collar === "neckline_edge") ? "flex" : "none";
    el("seamParams").style.display = (seam === "none") ? "none" : "block";

    if (collar !== "manual_point") {
      pickX = -1.0;
      pickY = -1.0;
      el("pickDot").style.display = "none";
      setTxt("pickInfo", "");
    }
  }

  function server() {
    return el("serverUrl").value.replace(/\/+$/, "");
  }

  async function upload() {
    const f = el("file").files[0];
    if (!f) return alert("请选择图片");
    const fd = new FormData();
    fd.append("file", f);
    const resp = await fetch(server() + "/api/upload", { method: "POST", body: fd });
    const data = await resp.json();
    if (!data.ok) return alert("上传失败: " + JSON.stringify(data));
    clothId = data.cloth_id;
    setTxt("uploadInfo", `已上传: ${data.filename}  (cloth_id=${data.cloth_id})`);
    el("uploadPreview").src = server() + "/api/uploaded_image?t=" + Date.now();
    el("pickDot").style.display = "none";
    pickX = -1.0;
    pickY = -1.0;
    setTxt("pickInfo", "");
    el("runBtn").disabled = false;
    el("toggleFB").disabled = true;
    el("left").disabled = true;
    el("right").disabled = true;
    el("img").src = "";
    setTxt("viewInfo", "");
    setTxt("log", "");
    updateVisibility();
  }

  function payload() {
    return {
      collar_module: el("collarModule").value,
      seam_module: el("seamModule").value,
      seam_band_width: asNum("seamBandWidth"),
      neckline_edge_ymax_scale: asNum("ymax"),
      neckline_edge_depth_bonus: asNum("bonus"),
      neckline_edge_depth_penalty: asNum("penalty"),
      neckline_edge_slope_strength: asNum("slopeStrength"),
      neckline_edge_slope_power: asNum("slopePower"),
      neckline_manual_x: pickX,
      neckline_manual_y: pickY,
    };
  }

  async function run() {
    const resp = await fetch(server() + "/api/run", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload()) });
    const data = await resp.json();
    if (!data.ok) return alert("启动失败: " + JSON.stringify(data));
    jobId = data.job_id;
    setTxt("log", "已启动任务: " + jobId + "\n等待完成...");
    poll();
  }

  async function poll() {
    if (!jobId) return;
    const resp = await fetch(server() + "/api/status?job_id=" + encodeURIComponent(jobId));
    const data = await resp.json();
    setTxt("log", data.log_tail || "");
    if (data.status === "done") {
      await refreshResults();
      return;
    }
    setTimeout(poll, 2000);
  }

  async function refreshResults() {
    const resp = await fetch(server() + "/api/results");
    const data = await resp.json();
    filesFront = data.front || [];
    filesBack = data.back || [];
    clothId = data.cloth_id || clothId;
    which = "front";
    idx = 0;
    el("toggleFB").disabled = false;
    el("left").disabled = false;
    el("right").disabled = false;
    show();
  }

  function currentList() {
    return which === "front" ? filesFront : filesBack;
  }

  function show() {
    const list = currentList();
    if (!clothId || list.length === 0) {
      setTxt("viewInfo", "无结果");
      return;
    }
    idx = ((idx % list.length) + list.length) % list.length;
    const name = list[idx];
    setTxt("viewInfo", `${which}  ${idx + 1}/${list.length}  ${name}`);
    const url = server() + `/api/image?which=${encodeURIComponent(which)}&cloth_id=${encodeURIComponent(clothId)}&name=${encodeURIComponent(name)}&t=${Date.now()}`;
    el("img").src = url;
  }

  function rotate(delta) {
    const list = currentList();
    if (list.length === 0) return;
    idx += delta;
    show();
  }

  function toggleFB() {
    which = which === "front" ? "back" : "front";
    idx = 0;
    show();
  }

  el("uploadBtn").addEventListener("click", () => upload().catch(e => alert(String(e))));
  el("runBtn").addEventListener("click", () => run().catch(e => alert(String(e))));
  el("left").addEventListener("click", () => rotate(-1));
  el("right").addEventListener("click", () => rotate(1));
  el("toggleFB").addEventListener("click", () => toggleFB());
  el("collarModule").addEventListener("change", () => updateVisibility());
  el("seamModule").addEventListener("change", () => updateVisibility());

  el("uploadPreview").addEventListener("click", (ev) => {
    if (el("collarModule").value !== "manual_point") return;
    const img = el("uploadPreview");
    if (!img.naturalWidth || !img.naturalHeight) return;
    const rect = img.getBoundingClientRect();
    const x = (ev.clientX - rect.left) / rect.width;
    const y = (ev.clientY - rect.top) / rect.height;
    pickX = Math.max(0, Math.min(1, x));
    pickY = Math.max(0, Math.min(1, y));
    const dot = el("pickDot");
    dot.style.left = (pickX * rect.width) + "px";
    dot.style.top = (pickY * rect.height) + "px";
    dot.style.display = "block";
    setTxt("pickInfo", `pick: x=${pickX.toFixed(3)}  y=${pickY.toFixed(3)}`);
  });

  updateVisibility();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        data = HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _pick_port(host: str, port: int) -> int:
    s = socket.socket()
    try:
        s.bind((host, port))
        return port
    except OSError:
        s.bind((host, 0))
        return s.getsockname()[1]
    finally:
        s.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    port = _pick_port(args.host, int(args.port))
    httpd = ThreadingHTTPServer((args.host, port), Handler)

    url = f"http://{args.host}:{port}"
    if not args.no_open:
        threading.Timer(0.2, lambda: webbrowser.open(url)).start()
    httpd.serve_forever()


if __name__ == "__main__":
    main()
