# VTON

---

## 项目简介

本仓库用于毕业设计中的虚拟试穿推理链路，目标是把单张服装图（正面）扩展为多视角条件，并与 VTON360 推理对接，实现“Wonder3D →（中间模块）→ VTON360”的端到端试穿结果生成。

默认目录结构：
- Wonder3D：`./Wonder3D/Wonder3D-main`
- VTON360：`./VTON360-main`

## 模型下载与放置目录

### Wonder3D（多视图生成）

1) Wonder3D checkpoints（离线权重）
- 国内下载（阿里云盘）：https://www.alipan.com/s/T4rLUNAVq6V
- 放置目录：`./Wonder3D/Wonder3D-main/ckpts/`
- 目录结构示例：
  - `ckpts/unet/`
  - `ckpts/scheduler/`
  - `ckpts/vae/`

2) SAM 权重（用于分割/抠图流程）
- 下载：`https://huggingface.co/kunkaran/sam_vit_h_4b8939.pth/tree/main`
- 放置路径：`./Wonder3D/Wonder3D-main/sam_pt/sam_vit_h_4b8939.pth`

### VTON360（多视角一致性试穿推理）

1) 核心模型权重（百度网盘）
- Thuman 权重：https://pan.baidu.com/s/1SJH3QI30UKihOaU9owta5Q（提取码：32h3）
- MVHumannet 权重：https://pan.baidu.com/s/1Onu7BIFzOppRSzO97ZmlmQ（提取码：mahx）
- 放置目录：解压到 `VTON360-main/src/multiview_consist_edit/checkpoints/`

2) 基础模型权重（Hugging Face）
- Stable Diffusion 主模型（示例）：https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
- CLIP：https://huggingface.co/openai/clip-vit-base-patch32
- VAE：https://huggingface.co/diffusers/sd-vae-ft-mse
- Human Parsing：https://huggingface.co/spaces/yisol/IDM-VTON/tree/main/ckpt/humanparsing

建议放置位置：
- CLIP/VAE/Stable Diffusion 主模型：建议统一放到持久化目录（如 `./VTON360-main/pretrained_models`），并确保配置文件里的路径指向实际目录
- Parsing：`VTON360-main/src/multiview_consist_edit/parse_tool/ckpt/`

## 使用方法

项目运行方法见：`项目使用手册.txt`
环境配置见两个项目的requirements.txt，如果有冲突的部分，参照log/VTON360_env.txt与log/Wonder3D_env.txt进行调整。