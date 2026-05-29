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
    :root {
      --bg: #0b1020;
      --panel: rgba(255, 255, 255, 0.07);
      --panel2: rgba(255, 255, 255, 0.09);
      --border: rgba(255, 255, 255, 0.14);
      --text: rgba(255, 255, 255, 0.92);
      --muted: rgba(255, 255, 255, 0.66);
      --muted2: rgba(255, 255, 255, 0.46);
      --primary: #6d8bff;
      --ok: #36d399;
      --warn: #f6c177;
      --danger: #ff5d7a;
      --shadow: 0 18px 60px rgba(0, 0, 0, 0.35);
      --radius: 14px;
      --radiusSm: 10px;
      --gap: 14px;
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Noto Sans", "PingFang SC", "Microsoft YaHei";
      color: var(--text);
      background:
        radial-gradient(1200px 700px at 20% 10%, rgba(109, 139, 255, 0.20), transparent 60%),
        radial-gradient(900px 600px at 75% 20%, rgba(54, 211, 153, 0.12), transparent 55%),
        radial-gradient(1000px 800px at 35% 90%, rgba(255, 93, 122, 0.10), transparent 60%),
        var(--bg);
    }

    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New"; }
    .small { font-size: 12px; }

    .topbar {
      position: sticky;
      top: 0;
      z-index: 10;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      padding: 18px 18px 14px 18px;
      background: linear-gradient(to bottom, rgba(11, 16, 32, 0.92), rgba(11, 16, 32, 0.65));
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      backdrop-filter: blur(10px);
    }
    .topbar h1 { margin: 0; font-size: 18px; font-weight: 680; letter-spacing: 0.2px; }
    .sub { margin-top: 4px; color: var(--muted); font-size: 12px; }

    .container { max-width: 1100px; margin: 0 auto; padding: 16px 18px 26px 18px; }

    .card {
      background: linear-gradient(180deg, var(--panel2), var(--panel));
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 14px;
      box-shadow: var(--shadow);
    }
    .card + .card { margin-top: 14px; }

    .cardTitle {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      font-size: 13px;
      font-weight: 650;
      color: rgba(255, 255, 255, 0.86);
      margin-bottom: 12px;
    }

    .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: var(--gap); }
    @media (max-width: 920px) { .grid2 { grid-template-columns: 1fr; } }

    .field label { display: block; font-size: 12px; color: rgba(255, 255, 255, 0.78); margin-bottom: 8px; }
    .help { color: var(--muted2); font-size: 12px; line-height: 1.35; margin-top: 8px; }
    .info { color: var(--muted); font-size: 12px; margin-top: 10px; }

    input[type="text"], select, input[type="file"] {
      width: 100%;
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid rgba(255, 255, 255, 0.14);
      background: rgba(0, 0, 0, 0.20);
      color: var(--text);
      outline: none;
    }
    input[type="text"]:focus, select:focus {
      border-color: rgba(109, 139, 255, 0.55);
      box-shadow: 0 0 0 4px rgba(109, 139, 255, 0.16);
    }
    select {
      appearance: none;
      background-image: linear-gradient(45deg, transparent 50%, rgba(255, 255, 255, 0.65) 50%),
                        linear-gradient(135deg, rgba(255, 255, 255, 0.65) 50%, transparent 50%);
      background-position: calc(100% - 18px) calc(50% - 3px), calc(100% - 13px) calc(50% - 3px);
      background-size: 6px 6px, 6px 6px;
      background-repeat: no-repeat;
      padding-right: 32px;
    }

    input[type="range"] { width: 100%; accent-color: var(--primary); }

    .actions { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 10px; }
    .btn {
      border: 1px solid rgba(255, 255, 255, 0.16);
      background: rgba(255, 255, 255, 0.08);
      color: var(--text);
      padding: 10px 12px;
      border-radius: 12px;
      cursor: pointer;
      font-weight: 620;
      letter-spacing: 0.1px;
      transition: transform 0.05s ease, background 0.15s ease, border-color 0.15s ease;
      user-select: none;
    }
    .btn:hover { background: rgba(255, 255, 255, 0.12); border-color: rgba(255, 255, 255, 0.20); }
    .btn:active { transform: translateY(1px); }
    .btn.primary { background: rgba(109, 139, 255, 0.22); border-color: rgba(109, 139, 255, 0.42); }
    .btn.primary:hover { background: rgba(109, 139, 255, 0.30); }
    .btn[disabled] { opacity: 0.55; cursor: not-allowed; transform: none; }
    .btnGroup { display: inline-flex; gap: 10px; }

    .status {
      padding: 8px 10px;
      border-radius: 999px;
      font-size: 12px;
      border: 1px solid rgba(255, 255, 255, 0.16);
      background: rgba(255, 255, 255, 0.08);
      color: rgba(255, 255, 255, 0.82);
      min-width: 84px;
      text-align: center;
    }
    .status[data-kind="idle"] { border-color: rgba(255, 255, 255, 0.14); }
    .status[data-kind="uploaded"] { border-color: rgba(54, 211, 153, 0.40); background: rgba(54, 211, 153, 0.12); }
    .status[data-kind="running"] { border-color: rgba(246, 193, 119, 0.55); background: rgba(246, 193, 119, 0.14); }
    .status[data-kind="done"] { border-color: rgba(54, 211, 153, 0.55); background: rgba(54, 211, 153, 0.16); }
    .status[data-kind="error"] { border-color: rgba(255, 93, 122, 0.55); background: rgba(255, 93, 122, 0.14); }

    .statusWrap { display: flex; align-items: center; gap: 10px; }
    .progressWrap { width: 180px; display: none; }
    .progressTrack { height: 10px; background: rgba(255, 255, 255, 0.10); border: 1px solid rgba(255, 255, 255, 0.10); border-radius: 999px; overflow: hidden; }
    .progressFill { height: 100%; width: 0%; background: linear-gradient(90deg, rgba(109, 139, 255, 0.85), rgba(54, 211, 153, 0.85)); border-radius: 999px; transition: width 0.25s ease; }
    .progressText { margin-top: 6px; font-size: 11px; color: rgba(255, 255, 255, 0.70); text-align: right; }

    .details { margin-top: 12px; border: 1px solid rgba(255, 255, 255, 0.10); border-radius: var(--radiusSm); background: rgba(0, 0, 0, 0.18); }
    .details summary { cursor: pointer; padding: 10px 12px; color: rgba(255, 255, 255, 0.86); font-size: 12px; font-weight: 650; }
    .detailsBody { padding: 12px; border-top: 1px solid rgba(255, 255, 255, 0.10); }

    .paramGrid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; }
    @media (max-width: 920px) { .paramGrid { grid-template-columns: 1fr; } }

    .check {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid rgba(255, 255, 255, 0.10);
      background: rgba(0, 0, 0, 0.14);
      user-select: none;
    }
    .check input[type="checkbox"] { width: 16px; height: 16px; accent-color: var(--primary); }

    .pickCard { margin-top: 12px; border: 1px solid rgba(255, 255, 255, 0.10); border-radius: var(--radiusSm); background: rgba(0, 0, 0, 0.16); overflow: hidden; display: none; }
    .pickHeader { display: flex; align-items: baseline; justify-content: space-between; gap: 10px; padding: 10px 12px; border-bottom: 1px solid rgba(255, 255, 255, 0.10); }
    .pickTitle { font-size: 12px; font-weight: 700; color: rgba(255, 255, 255, 0.86); }
    .pickBody { padding: 12px; }
    .pickStage { position: relative; display: inline-block; max-width: 100%; }
    #uploadPreview { display: block; max-width: min(760px, 100%); max-height: 520px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.12); background: rgba(255, 255, 255, 0.04); }
    #pickDot { position: absolute; width: 12px; height: 12px; border: 3px solid rgba(255, 255, 255, 0.95); box-shadow: 0 0 0 3px rgba(109, 139, 255, 0.35); border-radius: 50%; transform: translate(-50%, -50%); display: none; }

    .toolbar { display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; }
    .viewer { margin-top: 12px; background: rgba(0, 0, 0, 0.18); border: 1px solid rgba(255, 255, 255, 0.10); border-radius: var(--radiusSm); padding: 10px; }
    #img { display: block; max-width: 100%; max-height: 680px; margin: 0 auto; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.10); background: rgba(255, 255, 255, 0.03); }

    .thumbs { margin-top: 10px; display: flex; gap: 8px; overflow-x: auto; padding: 2px 2px 6px 2px; }
    .thumbs::-webkit-scrollbar { height: 10px; }
    .thumbs::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.14); border-radius: 999px; }
    .thumb { width: 92px; height: 68px; flex: 0 0 auto; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.10); overflow: hidden; background: rgba(255, 255, 255, 0.03); cursor: pointer; }
    .thumb img { width: 100%; height: 100%; object-fit: cover; display: block; }
    .thumb[data-active="1"] { border-color: rgba(109, 139, 255, 0.70); box-shadow: 0 0 0 3px rgba(109, 139, 255, 0.18); }

    .log {
      white-space: pre-wrap;
      font-size: 12px;
      line-height: 1.35;
      max-height: 260px;
      overflow: auto;
      background: rgba(0, 0, 0, 0.46);
      border: 1px solid rgba(255, 255, 255, 0.10);
      color: rgba(255, 255, 255, 0.86);
      padding: 12px;
      border-radius: var(--radiusSm);
    }
  </style>
</head>
<body>
  <header class="topbar">
    <div>
      <h1>VTON 可视化客户端</h1>
      <div class="sub">上传衣物图 → 选择模块/参数 → 运行 → 浏览多视角结果</div>
    </div>
    <div class="statusWrap">
      <div class="status" id="statusBadge" data-kind="idle">未上传</div>
      <div class="progressWrap" id="progressWrap">
        <div class="progressTrack"><div class="progressFill" id="progressFill"></div></div>
        <div class="progressText" id="progressText"></div>
      </div>
    </div>
  </header>

  <main class="container">
    <section class="card">
      <div class="cardTitle">连接与上传</div>
      <div class="grid2">
        <div class="field">
          <label>服务端地址</label>
          <input id="serverUrl" type="text" value="http://127.0.0.1:8766" />
          <div class="help">建议用 SSH 端口转发后填本地地址</div>
        </div>
        <div class="field">
          <label>上传衣服图片</label>
          <input id="file" type="file" accept="image/*" />
          <div class="actions">
            <button class="btn" id="uploadBtn">上传并清空历史</button>
            <button class="btn primary" id="runBtn" disabled>试穿</button>
          </div>
          <div class="mono info" id="uploadInfo"></div>
        </div>
      </div>
    </section>

    <section class="card">
      <div class="cardTitle">领口模块</div>
      <div class="grid2">
        <div class="field">
          <label>领口模块</label>
          <select id="collarModule">
            <option value="none">none</option>
            <option value="manual_point" selected>manual_point</option>
          </select>
        </div>
        <div class="field">
          <label>前视图 alpha 修正</label>
          <div class="check">
            <input id="frontAlphaFix" type="checkbox" checked />
            <div>启用（推荐）</div>
          </div>
          <div class="help">关闭后将直接使用 Wonder3D 的 masked_colors 输出 alpha</div>
        </div>
        <div class="field"></div>
      </div>

      <div id="manualPickSection" class="pickCard">
        <div class="pickHeader">
          <div>
            <div class="pickTitle">手动点选前领口最低点</div>
            <div class="help">点选坐标会在后端用于生成裁剪曲线；未点选时默认不裁剪</div>
          </div>
          <div class="mono small" id="pickInfo"></div>
        </div>
        <div class="pickBody">
          <div class="pickStage">
            <img id="uploadPreview" />
            <div id="pickDot"></div>
          </div>
        </div>
      </div>

      <details id="collarAdvanced" class="details" open>
        <summary>领口高级参数</summary>
        <div class="detailsBody">
          <div id="manualPointParams" class="paramGrid">
            <div class="field">
              <label>manual_shape（圆 ↔ 尖） <span id="manualShapeVal" class="mono"></span></label>
              <input id="manualShape" type="range" min="0.50" max="2.50" step="0.05" value="1.00"/>
              <div class="help">往左更圆：两肩到中心下降更快、靠近最低点更平；往右更尖：更接近直线 V</div>
            </div>
            <div class="field"></div>
            <div class="field"></div>
          </div>
        </div>
      </details>
    </section>

    <section class="card">
      <div class="cardTitle">接缝模块</div>
      <div class="grid2">
        <div class="field">
          <label>接缝模块</label>
          <select id="seamModule">
            <option value="none">none</option>
            <option value="side_views" selected>side_views</option>
            <option value="feather_stats">feather_stats</option>
          </select>
        </div>
        <div class="field"></div>
      </div>

      <details id="seamAdvanced" class="details" open>
        <summary>接缝高级参数</summary>
        <div class="detailsBody">
          <div class="paramGrid">
            <div id="seamParams" class="field">
              <label>seam_band_width <span id="seamBandWidthVal" class="mono"></span></label>
              <input id="seamBandWidth" type="range" min="8" max="64" step="1" value="24"/>
            </div>
            <div class="field"></div>
            <div class="field"></div>
          </div>
        </div>
      </details>
    </section>

    <section class="card">
      <div class="cardTitle">结果预览</div>
      <div class="toolbar">
        <div class="btnGroup">
          <select id="personId">
            <option value="100007">女（100007）</option>
            <option value="100067" selected>男（100067）</option>
          </select>
          <select id="poseId"></select>
        </div>
        <button class="btn" id="toggleFB" disabled>前后切换</button>
        <div class="btnGroup">
          <button class="btn" id="left" disabled>左旋转</button>
          <button class="btn" id="right" disabled>右旋转</button>
        </div>
        <span class="mono small" id="viewInfo"></span>
      </div>
      <div class="viewer">
        <img id="img" />
      </div>
      <div class="thumbs" id="thumbs"></div>
    </section>

    <section class="card">
      <div class="cardTitle">日志（尾部）</div>
      <div id="log" class="log"></div>
    </section>
  </main>

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
  bindRange("manualShape", "manualShapeVal");

  let jobId = "";
  let clothId = "";
  let which = "front";
  let idx = 0;
  let filesFront = [];
  let filesBack = [];
  let pickX = -1.0;
  let pickY = -1.0;

  function setStatus(text, kind) {
    setTxt("statusBadge", text);
    el("statusBadge").setAttribute("data-kind", kind || "idle");
    el("progressWrap").style.display = (kind === "running") ? "block" : "none";
    if (kind !== "running") {
      el("progressFill").style.width = "0%";
      setTxt("progressText", "");
    }
  }

  function setProgress(progress, stage) {
    const p = Math.max(0, Math.min(1, Number(progress ?? 0)));
    el("progressFill").style.width = Math.round(p * 100) + "%";
    const stageTxt = String(stage || "").trim();
    setTxt("progressText", stageTxt ? `${Math.round(p * 100)}% · ${stageTxt}` : `${Math.round(p * 100)}%`);
  }

  function updateVisibility() {
    const collar = el("collarModule").value;
    const seam = el("seamModule").value;
    el("manualPickSection").style.display = (collar === "manual_point") ? "block" : "none";
    el("manualPointParams").style.display = (collar === "manual_point") ? "grid" : "none";
    el("seamParams").style.display = (seam === "none") ? "none" : "";

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

  function updatePoseOptions() {
    const person = el("personId").value;
    const poseSel = el("poseId");
    poseSel.innerHTML = "";
    const poses = (person === "100007") ? ["1005", "1710"] : ["0320", "1220"];
    for (const p of poses) {
      const opt = document.createElement("option");
      opt.value = p;
      opt.textContent = `姿势 ${p}`;
      poseSel.appendChild(opt);
    }
  }

  async function upload() {
    const f = el("file").files[0];
    if (!f) return alert("请选择图片");
    setStatus("上传中", "running");
    const fd = new FormData();
    fd.append("file", f);
    const resp = await fetch(server() + "/api/upload", { method: "POST", body: fd });
    const data = await resp.json();
    if (!data.ok) {
      setStatus("上传失败", "error");
      return alert("上传失败: " + JSON.stringify(data));
    }
    clothId = data.cloth_id;
    setTxt("uploadInfo", `已上传: ${data.filename}  (cloth_id=${data.cloth_id})`);
    const prev = el("uploadPreview");
    prev.onerror = () => { prev.onerror = null; prev.src = server() + "/api/uploaded_image?t=" + Date.now(); };
    prev.src = server() + "/api/pick_preview?t=" + Date.now();
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
    el("thumbs").innerHTML = "";
    setStatus("已上传", "uploaded");
    updateVisibility();
  }

  function payload() {
    return {
      person_id: el("personId").value,
      pose_id: el("poseId").value,
      front_alpha_fix: el("frontAlphaFix").checked,
      collar_module: el("collarModule").value,
      seam_module: el("seamModule").value,
      seam_band_width: asNum("seamBandWidth"),
      neckline_manual_x: pickX,
      neckline_manual_y: pickY,
      neckline_manual_shape: asNum("manualShape"),
    };
  }

  async function run() {
    setStatus("运行中", "running");
    const resp = await fetch(server() + "/api/run", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload()) });
    const data = await resp.json();
    if (!data.ok) {
      setStatus("启动失败", "error");
      return alert("启动失败: " + JSON.stringify(data));
    }
    jobId = data.job_id;
    setTxt("log", "已启动任务: " + jobId + "\n等待完成...");
    poll();
  }

  async function poll() {
    if (!jobId) return;
    const resp = await fetch(server() + "/api/status?job_id=" + encodeURIComponent(jobId));
    const data = await resp.json();
    setTxt("log", data.log_tail || "");
    if (data.status !== "done") {
      setProgress(data.progress, data.stage);
      if (data.stage) setStatus("运行中", "running");
    }
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
    setStatus("已完成", "done");
    show();
  }

  function currentList() {
    return which === "front" ? filesFront : filesBack;
  }

  function renderThumbs() {
    const list = currentList();
    const box = el("thumbs");
    box.innerHTML = "";
    if (!clothId || list.length === 0) return;
    const limit = Math.min(list.length, 24);
    for (let i = 0; i < limit; i++) {
      const name = list[i];
      const thumb = document.createElement("div");
      thumb.className = "thumb";
      thumb.setAttribute("data-active", i === idx ? "1" : "0");
      const img = document.createElement("img");
      img.loading = "lazy";
      img.alt = name;
      img.src = server() + `/api/image?which=${encodeURIComponent(which)}&cloth_id=${encodeURIComponent(clothId)}&name=${encodeURIComponent(name)}&t=${Date.now()}`;
      thumb.appendChild(img);
      thumb.addEventListener("click", () => { idx = i; show(); });
      box.appendChild(thumb);
    }
  }

  function show() {
    const list = currentList();
    if (!clothId || list.length === 0) {
      setTxt("viewInfo", "无结果");
      el("thumbs").innerHTML = "";
      return;
    }
    idx = ((idx % list.length) + list.length) % list.length;
    const name = list[idx];
    setTxt("viewInfo", `${which}  ${idx + 1}/${list.length}  ${name}`);
    const url = server() + `/api/image?which=${encodeURIComponent(which)}&cloth_id=${encodeURIComponent(clothId)}&name=${encodeURIComponent(name)}&t=${Date.now()}`;
    el("img").src = url;
    renderThumbs();
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

  el("uploadBtn").addEventListener("click", () => upload().catch(e => { setStatus("上传失败", "error"); alert(String(e)); }));
  el("runBtn").addEventListener("click", () => run().catch(e => { setStatus("运行失败", "error"); alert(String(e)); }));
  el("left").addEventListener("click", () => rotate(-1));
  el("right").addEventListener("click", () => rotate(1));
  el("toggleFB").addEventListener("click", () => toggleFB());
  el("collarModule").addEventListener("change", () => updateVisibility());
  el("seamModule").addEventListener("change", () => updateVisibility());
  el("personId").addEventListener("change", () => updatePoseOptions());

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

  setStatus("未上传", "idle");
  updatePoseOptions();
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
