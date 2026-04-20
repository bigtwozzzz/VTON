import argparse
import json
import shutil
import subprocess
from pathlib import Path


def _require_executable(name: str) -> str:
    exe = shutil.which(name)
    if exe is None:
        raise RuntimeError(f"Missing executable in PATH: {name}")
    return exe


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd is not None else None, check=True)


def _conda_run(env_name: str, argv: list[str]) -> list[str]:
    conda = _require_executable("conda")
    env = env_name.strip()
    if "/" in env or env.startswith("."):
        # 兼容“prefix 环境”（conda env list 只显示路径、不显示名字的情况）
        return [conda, "run", "-p", env, *argv]
    # 兼容“named 环境”
    return [conda, "run", "-n", env, *argv]


def _bash_lc(command: str) -> list[str]:
    return ["bash", "-lc", command]


def _parse_cloth_id(path: Path) -> tuple[str, str]:
    stem = path.stem
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Cannot parse cloth id/frame id from filename: {path.name}")
    # VTON360 的 mvhumannet_2D_edit 数据约定：cloth_id 和 frame_id 分别是文件名前两个 '_' 分段
    return parts[0], parts[1]


def _find_latest_file(root: Path, relative_suffix: str) -> Path:
    matches = list(root.glob(f"**/{relative_suffix}"))
    if not matches:
        raise FileNotFoundError(f"Cannot find {relative_suffix} under {root}")
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def _find_scene_file(save_dir: Path, scene_name: str, relative_suffix: str) -> Path:
    matches = list(save_dir.glob(f"**/{scene_name}/{relative_suffix}"))
    if not matches:
        raise FileNotFoundError(f"Cannot find {scene_name}/{relative_suffix} under {save_dir}")
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def _iter_cloth_inputs(cloth_input_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    files = [p for p in cloth_input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files = [p for p in files if "input_rgba" not in p.stem and not p.name.endswith("_rgba.png")]
    files.sort()
    return files


def _reset_vton360_cloth(vton360_cloth_dir: Path, test_cloth_ids_path: Path) -> None:
    vton360_cloth_dir.mkdir(parents=True, exist_ok=True)
    for p in vton360_cloth_dir.iterdir():
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            p.unlink()
    test_cloth_ids_path.parent.mkdir(parents=True, exist_ok=True)
    # VTON360 的数据读取依赖 test_cloth_ids.txt；每次跑新衣服时清空并重建，避免混入旧 id
    test_cloth_ids_path.write_text("", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cloth-image", default="")
    parser.add_argument("--cloth-input-dir", default="/root/autodl-tmp/VTON/test_data/cloth_inputs")
    parser.add_argument("--wonder3d-env", default="wonder3d")
    parser.add_argument("--vton360-env", default="vton360")
    parser.add_argument("--vton360-data-root", default="/root/autodl-tmp/VTON/VTON360-main/src/data/mvhumannet_2D_edit")
    parser.add_argument("--invert-vton-cloth-names", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reset-vton360-cloth", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--collar-module", choices=["none", "cut_top_bump", "neckline_cut", "neckline_edge", "manual_point"], default="none")
    parser.add_argument("--seam-module", choices=["none", "feather_stats", "side_views"], default="none")
    parser.add_argument("--seam-band-width", type=int, default=24)
    parser.add_argument("--neckline-edge-ymax-scale", type=float, default=0.40)
    parser.add_argument("--neckline-edge-depth-bonus", type=float, default=0.28)
    parser.add_argument("--neckline-edge-depth-penalty", type=float, default=0.04)
    parser.add_argument("--neckline-edge-slope-strength", type=float, default=0.12)
    parser.add_argument("--neckline-edge-slope-power", type=float, default=1.6)
    parser.add_argument("--neckline-manual-x", type=float, default=-1.0)
    parser.add_argument("--neckline-manual-y", type=float, default=-1.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project_root = Path("/root/autodl-tmp/VTON")
    cloth_images: list[Path]
    if args.cloth_image:
        cloth_images = [Path(args.cloth_image).resolve()]
    else:
        cloth_input_dir = Path(args.cloth_input_dir).resolve()
        if not cloth_input_dir.exists():
            raise FileNotFoundError(str(cloth_input_dir))
        cloth_images = _iter_cloth_inputs(cloth_input_dir)

    missing = [str(p) for p in cloth_images if not p.exists()]
    if missing:
        raise FileNotFoundError("\n".join(missing))
    if not cloth_images:
        raise RuntimeError("No cloth input images found.")

    wonder3d_root = project_root / "Wonder3D" / "Wonder3D-main"
    vton360_root = project_root / "VTON360-main" / "src" / "multiview_consist_edit"
    vton360_data_root = Path(args.vton360_data_root).resolve()
    vton360_cloth_dir = vton360_data_root / "cloth"
    vton360_test_cloth_ids = vton360_data_root / "test_cloth_ids.txt"

    outputs_root = project_root / "test_data" / "outputs"
    # Wonder3D 的输出根目录（由命令行参数 save_dir 指定到这里）
    # 最终文件会落在：
    #   {save_dir}/cropsize-<crop>-cfg<cfg>/<scene_name>/
    # 其中 scene_name = 输入 RGBA 文件名去掉 .png 后缀，例如：
    #   102669_3135_input_rgba
    wonder3d_save_dir = outputs_root / "wonder3d"
    # VTON360 推理输出目录（脚本会跑两次：output_front=True/False 分别写到两个目录）
    outputs_vton_front = outputs_root / "vton360_front"
    outputs_vton_back = outputs_root / "vton360_back"

    work_dir = project_root / "test_data" / "work"
    # Wonder3D 读取输入图片是从 validation_dataset.root_dir + filepaths 组合出来的；
    # 所以把“临时 RGBA 输入”统一写在这个目录里，避免污染 Wonder3D 项目目录。
    inputs_dir = work_dir / "wonder3d_inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    outputs_root.mkdir(parents=True, exist_ok=True)
    wonder3d_save_dir.mkdir(parents=True, exist_ok=True)
    outputs_vton_front.mkdir(parents=True, exist_ok=True)
    outputs_vton_back.mkdir(parents=True, exist_ok=True)

    if args.reset_vton360_cloth:
        _reset_vton360_cloth(vton360_cloth_dir, vton360_test_cloth_ids)
    else:
        vton360_cloth_dir.mkdir(parents=True, exist_ok=True)
        vton360_test_cloth_ids.parent.mkdir(parents=True, exist_ok=True)

    # Step 0) 根据 /root/autodl-tmp/VTON/test_data/cloth_inputs 目录里的图片，
    # 自动生成 VTON360 需要的 test_cloth_ids.txt（每行：<cloth_id> <frame_id>）。
    # 约定：从文件名解析 cloth_id / frame_id（取前两个 '_' 分隔片段），例如：
    #   102669_3135_back.jpg  -> cloth_id=102669, frame_id=3135
    cloth_items: list[tuple[str, str, Path, str]] = []
    seen_keys: set[tuple[str, str]] = set()
    for img_path in cloth_images:
        cloth_id, frame_id = _parse_cloth_id(img_path)
        key = (cloth_id, frame_id)
        if key in seen_keys:
            raise RuntimeError(f"Duplicate cloth id/frame id detected: {cloth_id} {frame_id}")
        seen_keys.add(key)
        rgba_name = f"{cloth_id}_{frame_id}_input_rgba.png"
        cloth_items.append((cloth_id, frame_id, img_path, rgba_name))

    vton360_test_cloth_ids.write_text(
        "\n".join([f"{cid} {fid}" for cid, fid, _, _ in cloth_items]) + "\n",
        encoding="utf-8",
    )

    # Step 1) 生成 Wonder3D 需要的 RGBA（alpha 通道非常关键，Wonder3D 会用 alpha 做 crop/抠主体）
    # 临时文件路径：
    #   /root/autodl-tmp/VTON/test_data/work/wonder3d_inputs/{cloth_id}_{frame_id}_input_rgba.png
    # 注意：这里会优先尝试用 rembg 抠图生成 alpha；如果 rembg 不可用，则退化为仅转 RGBA。
    rgba_jobs = [(str(img_path), str(inputs_dir / rgba_name)) for _, _, img_path, rgba_name in cloth_items]
    rgba_py = "\n".join(
        [
            "from PIL import Image",
            "jobs = " + repr(rgba_jobs),
            "for inp_path, out_path in jobs:",
            "    img = Image.open(inp_path).convert('RGBA')",
            "    try:",
            "        from rembg import remove",
            "        img = remove(img, alpha_matting=True)",
            "    except Exception:",
            "        pass",
            "    img.save(out_path)",
            "    print(out_path)",
        ]
    )
    cmd_rgba = _conda_run(args.wonder3d_env, ["python", "-c", rgba_py])

    # Step 2) Wonder3D 推理（读取 inputs_dir 下的 rgba_name，输出写入 wonder3d_save_dir）
    # Wonder3D 的输出文件命名规则来自脚本 test_mvdiffusion_seq.py：
    # - 颜色结果：rgb_000_{view}.png（view 为 front/back/right/...）
    # - 抠图结果：masked_colors/rgb_000_{view}.png（remove 背景后的 RGBA）
    rgba_names = [rgba_name for _, _, _, rgba_name in cloth_items]
    filepaths_arg = "validation_dataset.filepaths=[" + ",".join([repr(n) for n in rgba_names]) + "]"
    wonder3d_args = [
        "accelerate",
        "launch",
        "--config_file",
        "1gpu.yaml",
        "test_mvdiffusion_seq.py",
        "--config",
        "configs/mvdiffusion-joint-ortho-6views.yaml",
        f"validation_dataset.root_dir={str(inputs_dir)}",
        filepaths_arg,
        f"save_dir={str(wonder3d_save_dir)}",
    ]
    cmd_wonder3d = _conda_run(args.wonder3d_env, wonder3d_args)

    if args.dry_run:
        print("RGBA:", " ".join(cmd_rgba))
        print("Wonder3D:", " ".join(cmd_wonder3d))
        return

    _run(cmd_rgba)
    _run(cmd_wonder3d, cwd=wonder3d_root)

    # Step 3) 为每个输入衣服，从 Wonder3D 输出中取 front/back 视角图片。
    # 默认优先使用 masked_colors 下的文件（抠掉背景后的 RGBA），路径形如：
    #   /root/autodl-tmp/VTON/test_data/outputs/wonder3d/
    #     cropsize-192-cfg1.0/102669_3135_input_rgba/masked_colors/rgb_000_back.png
    #
    # 注意：Wonder3D 的 scene_name 是输入 RGBA 文件名去掉 .png 后缀：
    #   rgba_name = 102669_3135_input_rgba.png  -> scene_name = 102669_3135_input_rgba
    #
    # 经验问题：部分浅色短袖在 Wonder3D 的 masked_colors/rgb_000_front.png 里会把袖子抠没（alpha 变小），
    # 合成白底后看起来像“袖子变白”。因此这里对 front 视图做一个修正：
    # - 读取 Wonder3D 的 rgb_000_front.png（不带 mask）
    # - 使用输入的 input_rgba.png 的 alpha 作为 front 视图的 alpha（front 视角与输入一致）
    fixed_root = work_dir / "wonder3d_fixed"
    fixed_root.mkdir(parents=True, exist_ok=True)
    fixed_jobs: list[tuple[str, str, str]] = []
    for cloth_id, frame_id, _, rgba_name in cloth_items:
        scene_name = Path(rgba_name).stem
        try:
            raw_front = _find_scene_file(wonder3d_save_dir, scene_name, "rgb_000_front.png")
        except FileNotFoundError:
            continue
        in_rgba = inputs_dir / rgba_name
        out_dir = fixed_root / scene_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_front = out_dir / "front_rgba.png"
        fixed_jobs.append((str(raw_front), str(in_rgba), str(out_front)))

    if fixed_jobs:
        fix_front_py = "\n".join(
            [
                "from PIL import Image",
                "jobs = " + repr(fixed_jobs),
                "for raw_front, in_rgba, out_path in jobs:",
                "    rgb = Image.open(raw_front).convert('RGB')",
                "    a = Image.open(in_rgba).convert('RGBA').split()[-1].resize(rgb.size, Image.BILINEAR)",
                "    Image.merge('RGBA', (*rgb.split(), a)).save(out_path)",
                "    print(out_path)",
            ]
        )
        _run(_conda_run(args.wonder3d_env, ["python", "-c", fix_front_py]))
    # Step 3+4) 生成 VTON360 的 cloth 输入（front/back 两张图）。
    # 这里支持把“衣领处理模块 / 接缝处理模块”插到 Wonder3D 与 VTON360 之间，通过命令行参数控制开启哪一个。
    vton_items: list[dict[str, str]] = []
    for cloth_id, frame_id, src_input_img, rgba_name in cloth_items:
        scene_name = Path(rgba_name).stem
        back_png = _find_scene_file(wonder3d_save_dir, scene_name, "masked_colors/rgb_000_back.png")
        front_png: Path | None
        try:
            front_png = _find_scene_file(wonder3d_save_dir, scene_name, "masked_colors/rgb_000_front.png")
        except FileNotFoundError:
            front_png = None
        left_png: Path | None
        right_png: Path | None
        try:
            left_png = _find_scene_file(wonder3d_save_dir, scene_name, "masked_colors/rgb_000_left.png")
        except FileNotFoundError:
            left_png = None
        try:
            right_png = _find_scene_file(wonder3d_save_dir, scene_name, "masked_colors/rgb_000_right.png")
        except FileNotFoundError:
            right_png = None

        front_dst = vton360_cloth_dir / f"{cloth_id}_{frame_id}_front.jpg"
        back_dst = vton360_cloth_dir / f"{cloth_id}_{frame_id}_back.jpg"

        # semantic：Wonder3D 的 front/back 视角（与 VTON360 的文件名语义无关）
        fixed_front = fixed_root / scene_name / "front_rgba.png"
        if fixed_front.exists():
            src_sem_front = str(fixed_front)
        else:
            src_sem_front = str(front_png) if front_png is not None else str(src_input_img)
        src_sem_back = str(back_png)
        src_sem_left = str(left_png) if left_png is not None else ""
        src_sem_right = str(right_png) if right_png is not None else ""

        vton_items.append(
            {
                "src_sem_front": src_sem_front,
                "src_sem_back": src_sem_back,
                "src_sem_left": src_sem_left,
                "src_sem_right": src_sem_right,
                "dst_front": str(front_dst),
                "dst_back": str(back_dst),
            }
        )

    items_json_path = work_dir / "vton_items.json"
    items_json_path.write_text(json.dumps(vton_items, ensure_ascii=False, indent=2), encoding="utf-8")

    modules_script = project_root / "src" / "cloth_modules.py"
    cmd_modules = _conda_run(
        args.vton360_env,
        [
            "python",
            str(modules_script),
            "--items-json",
            str(items_json_path),
            "--collar-module",
            args.collar_module,
            "--seam-module",
            args.seam_module,
            "--seam-band-width",
            str(args.seam_band_width),
            "--invert-vton-cloth-names" if args.invert_vton_cloth_names else "--no-invert-vton-cloth-names",
            "--neckline-edge-ymax-scale",
            str(args.neckline_edge_ymax_scale),
            "--neckline-edge-depth-bonus",
            str(args.neckline_edge_depth_bonus),
            "--neckline-edge-depth-penalty",
            str(args.neckline_edge_depth_penalty),
            "--neckline-edge-slope-strength",
            str(args.neckline_edge_slope_strength),
            "--neckline-edge-slope-power",
            str(args.neckline_edge_slope_power),
            "--neckline-manual-x",
            str(args.neckline_manual_x),
            "--neckline-manual-y",
            str(args.neckline_manual_y),
            "--debug-dir",
            str(work_dir),
        ],
    )
    # 处理模块单独放在 src/cloth_modules.py：
    # - 做领口/接缝等 2D 修复（可选）
    # - 把处理后的 cloth 正背面写入 VTON360 的 cloth/ 目录
    _run(cmd_modules)

    # Step 5) 运行 VTON360 推理。
    # 因为 VTON360 推理脚本固定读取 config/infer_tryon_multi.yaml，
    # 这里通过“保存原内容 → 临时改写 → 跑两次 → 还原”的方式解决路径配置问题。
    vton_cfg = vton360_root / "config" / "infer_tryon_multi.yaml"
    infer_script = vton360_root / "infer_tryon_multi.py"

    vton_runner = "\n".join(
        [
            "import subprocess, sys",
            "from pathlib import Path",
            "from omegaconf import OmegaConf",
            f"cfg_path = Path({repr(str(vton_cfg))})",
            f"workdir = Path({repr(str(vton360_root))})",
            f"dataset_root = {repr(str(vton360_data_root))}",
            f"out_front = {repr(str(outputs_vton_front))}",
            f"out_back = {repr(str(outputs_vton_back))}",
            "orig = cfg_path.read_text(encoding='utf-8')",
            "try:",
            "    cfg = OmegaConf.load(cfg_path)",
            "    cfg.infer_data.dataroot = dataset_root",
            "    cfg.infer_data.output_front = True",
            "    cfg.out_dir = out_front",
            "    OmegaConf.save(cfg, cfg_path)",
            f"    subprocess.run([sys.executable, {repr(infer_script.name)}], cwd=str(workdir), check=True)",
            "    cfg = OmegaConf.load(cfg_path)",
            "    cfg.infer_data.output_front = False",
            "    cfg.out_dir = out_back",
            "    OmegaConf.save(cfg, cfg_path)",
            f"    subprocess.run([sys.executable, {repr(infer_script.name)}], cwd=str(workdir), check=True)",
            "finally:",
            "    cfg_path.write_text(orig, encoding='utf-8')",
        ]
    )

    _run(_conda_run(args.vton360_env, ["python", "-c", vton_runner]))

    print(str(outputs_root))


if __name__ == "__main__":
    main()
