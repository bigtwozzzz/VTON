# Wonder3D 使用说明（生成多视图 → 生成衣服背面 → 配合 VTON360）

本文面向你当前的目录结构：

- Wonder3D：/root/autodl-tmp/VTON/Wonder3D/Wonder3D-main
- VTON360：/root/autodl-tmp/VTON/VTON360-main

目标是：给定一张衣服正面图（VTON360 的 `*_front.jpg`），用 Wonder3D 生成多视图，从中取 `back` 视角，转换成 VTON360 需要的 `*_back.jpg`，从而让 VTON360 在推理阶段不再依赖真实背面拍摄图。

---

## 0. 你需要额外准备/下载的资源

Wonder3D 推理阶段主要依赖两类资源：

1) 模型权重（两种方式二选一）
- 在线方式：直接从 Hugging Face 自动下载（前提是服务器能访问 Hugging Face）。
- 离线方式：下载作者提供的 checkpoints，放到 `Wonder3D-main/ckpts/`，并修改配置 `configs/mvdiffusion-joint-ortho-6views.yaml` 的 `pretrained_model_name_or_path` 为 `./ckpts`。

2) SAM 权重（用于分割/抠图相关流程）
- 文件名：`sam_vit_h_4b8939.pth`
- 放置位置：`/root/autodl-tmp/VTON/Wonder3D/Wonder3D-main/sam_pt/sam_vit_h_4b8939.pth`

备注：
- Wonder3D 推理脚本会用 `rembg` 做背景移除，并在输出目录里额外保存一份“抠图后的结果”（更适合当作 VTON360 的衣服参考图）。

---

## 1. 环境准备（强烈建议与 VTON360 分开）

Wonder3D 的 `requirements.txt` 固定了 `diffusers[torch]==0.19.3`、`torch==1.13.1` 等版本；而 VTON360 里你已看到 `diffusers==0.25.0`。两者放在同一个 Python 环境里很容易冲突。

建议你创建独立 conda 环境：

```bash
cd /root/autodl-tmp/VTON/Wonder3D/Wonder3D-main

conda create -n wonder3d python=3.10 -y
conda activate wonder3d

pip install -r requirements.txt
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

如果你已经有 VTON360 的环境（例如 `vton360`），建议保持分开：
- `conda activate wonder3d`：只跑 Wonder3D 生成多视图/背面
- `conda activate vton360`：只跑 VTON360 的试穿推理

---

## 2. （可选）离线权重准备

如果服务器无法稳定访问 Hugging Face，按 Wonder3D 仓库说明离线准备：

1) 在 `Wonder3D-main` 下创建目录并放入 checkpoints：

```bash
cd /root/autodl-tmp/VTON/Wonder3D/Wonder3D-main
mkdir -p ckpts
```

将下载到的 `unet/ scheduler/ vae/ ...` 放到：

```
Wonder3D-main/
  ckpts/
    unet/
    scheduler/
    vae/
    ...
```

2) 修改配置文件：

文件：`/root/autodl-tmp/VTON/Wonder3D/Wonder3D-main/configs/mvdiffusion-joint-ortho-6views.yaml`

把第一行改为：

```yaml
pretrained_model_name_or_path: './ckpts'
```

---

## 3. SAM 权重准备

下载 `sam_vit_h_4b8939.pth` 后放到：

```bash
mkdir -p /root/autodl-tmp/VTON/Wonder3D/Wonder3D-main/sam_pt
# 把 sam_vit_h_4b8939.pth 放到 sam_pt/ 下
```

最终路径应为：

`/root/autodl-tmp/VTON/Wonder3D/Wonder3D-main/sam_pt/sam_vit_h_4b8939.pth`

---

## 4. 生成多视图（RGB + normals）

### 4.1 准备输入图片

建议输入为“单件衣服”的抠图/白底图，且衣服位于画面中心；效果通常明显优于“模特上身”或背景复杂的图。

你可以先把衣服正面图准备到一个目录，比如：

`/root/autodl-tmp/VTON/Wonder3D/cloth_inputs/`

并放入类似文件名：

`200543_2495_front.jpg`

### 4.2 运行推理脚本

在 Wonder3D 环境中执行：

```bash
conda activate wonder3d
cd /root/autodl-tmp/VTON/Wonder3D/Wonder3D-main

python - <<'PY'
from PIL import Image
from rembg import remove
inp = Image.open("/root/autodl-tmp/VTON/Wonder3D/Wonder3D-main/cloth_inputs/102669_3135_back.jpg").convert("RGBA")
out = remove(inp, alpha_matting=True)
out.save("/root/autodl-tmp/VTON/Wonder3D/Wonder3D-main/cloth_inputs/102669_3135_back_rgba.png")
print("saved")
PY

accelerate launch --config_file 1gpu.yaml test_mvdiffusion_seq.py \
  --config configs/mvdiffusion-joint-ortho-6views.yaml \
  validation_dataset.root_dir=/root/autodl-tmp/VTON/Wonder3D/Wonder3D-main/cloth_inputs \
  "validation_dataset.filepaths=['102669_3135_back_rgba.png']" \
  save_dir=/root/autodl-tmp/VTON/Wonder3D/Wonder3D-main/outputs
```

常用可调参数（直接改 `configs/mvdiffusion-joint-ortho-6views.yaml` 更方便）：
- `validation_dataset.crop_size`：默认 192。衣服占画面比例不合适时可以调大/调小。
- `validation_guidance_scales`：默认 `[1.0, 3.0]`。建议先用 `1.0`（更保真），再试 `3.0`（更“补全/脑补”）。

---

## 5. 输出目录结构与“背面”文件在哪里

以配置中的 `save_dir=/root/autodl-tmp/VTON/Wonder3D/outputs`、`crop_size=192`、`cfg=1.0` 为例，输出在：

`/root/autodl-tmp/VTON/Wonder3D/outputs/cropsize-192-cfg1.0/200543_2495_front.jpg/`

这个目录下你会看到（文件名由脚本固定生成）：

- `rgb_000_front.png`
- `rgb_000_back.png`
- `rgb_000_right.png` 等其它视角
- `normals_000_front.png`、`normals_000_back.png` 等
- `masked_colors/`：存放 `rembg` 抠图后的 RGB（更推荐作为衣服图）
  - `masked_colors/rgb_000_back.png`
  - `masked_colors/rgb_000_front.png`

Wonder3D 的 6 视角中，`back` 就是你要拿来当“背面衣服参考图”的主要候选。

---

## 6. 把 Wonder3D 的 back 视角接入 VTON360

VTON360 的 MVHumanNet 数据集读取衣服参考图时，文件名是固定规则：

`cloth/<cloth_id>_<cloth_frame_id>_front.jpg` 与 `cloth/<cloth_id>_<cloth_frame_id>_back.jpg`

你当前 demo 数据位于：

`/root/autodl-tmp/VTON/VTON360-main/src/demo_data/mvhumannet_2D_edit/cloth/`

### 6.1 生成/覆盖 `*_back.jpg`

假设你要处理的衣服 ID 是 `200543 2495`（在 demo 的 `test_cloth_ids.txt` 里就有这一条），则目标文件是：

- `.../cloth/200543_2495_front.jpg`（你已有的正面图）
- `.../cloth/200543_2495_back.jpg`（你要用 Wonder3D 产出的背面覆盖/生成）

推荐用 Wonder3D 的“抠图背面”：

`/root/autodl-tmp/VTON/Wonder3D/outputs/cropsize-192-cfg1.0/200543_2495_front.jpg/masked_colors/rgb_000_back.png`

然后把它转换成 JPG 并保存为 VTON360 的 back 文件名。示例（在任意 python 环境都可执行）：

```python
from PIL import Image
import os

src = "/root/autodl-tmp/VTON/Wonder3D/outputs/cropsize-192-cfg1.0/200543_2495_front.jpg/masked_colors/rgb_000_back.png"
dst = "/root/autodl-tmp/VTON/VTON360-main/src/demo_data/mvhumannet_2D_edit/cloth/200543_2495_back.jpg"

os.makedirs(os.path.dirname(dst), exist_ok=True)
img = Image.open(src).convert("RGB")
img.save(dst, quality=95)
print("saved:", dst)
```

如果你发现 `masked_colors` 抠图把衣服边缘抠坏了，可以改用未抠图版本：

`.../rgb_000_back.png`

或者自己用更稳定的抠图方案（比如你已有的衣服 mask）。

### 6.2 运行 VTON360 推理

切换到 VTON360 的环境（与你 wonder3d 环境分开），并保持 `infer_tryon_multi.yaml` 指向你的数据根目录，例如：

`/root/autodl-tmp/VTON/VTON360-main/src/demo_data/mvhumannet_2D_edit`

确保 `test_cloth_ids.txt` 里包含你要用的 `<cloth_id> <cloth_frame_id>`，然后运行你平时的 VTON360 推理命令即可（你之前已经能跑 demo 的话，这一步不需要额外改 VTON360 代码）。

---

## 7. 常见问题/排错要点

1) 结果不像背面/结构不合理
- 先换 `validation_guidance_scales`：对衣服这类输入，`cfg=1.0` 往往更“保形”，`cfg=3.0` 更“补全但可能乱编”。
- 确保输入衣服是居中、干净背景（最好是抠图后白底/透明）。
- 多跑几个 seed，选更像背面的那一张（Wonder3D 的配置里 `seed: 42` 可改）。

2) 生成很慢
- Wonder3D README 提到“2–3 分钟”通常指包含后续重建；你这里只跑 `test_mvdiffusion_seq.py` 的多视图生成，一般会快很多，但仍取决于 GPU 与步数。
- 减小 `validation_guidance_scales` 数量（只保留 `[1.0]`），减少重复生成。

3) 许可证
- Wonder3D 仓库为 AGPL-3.0，涉及“包含其代码/模型的下游服务”时需要特别注意合规（尤其是云服务/闭源部署场景）。

