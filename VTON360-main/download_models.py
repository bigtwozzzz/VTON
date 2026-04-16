import os
import shutil
import subprocess
import sys

# 设置环境变量以使用国内镜像加速 (可选，但在国内非常推荐)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 尝试导入 huggingface_hub，如果不存在则自动安装
try:
    from huggingface_hub import snapshot_download, hf_hub_download
except ImportError:
    print("正在安装 huggingface_hub...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
    from huggingface_hub import snapshot_download, hf_hub_download

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 1. 基础模型下载目录
PRETRAINED_MODELS_DIR = os.path.join(PROJECT_ROOT, "pretrained_models")
os.makedirs(PRETRAINED_MODELS_DIR, exist_ok=True)

# 2. Parsing 模型下载目录 (必须严格对应代码中的路径)
PARSING_CKPT_DIR = os.path.join(PROJECT_ROOT, "src", "multiview_consist_edit", "parse_tool", "ckpt", "humanparsing")
os.makedirs(PARSING_CKPT_DIR, exist_ok=True)

print(f"项目根目录: {PROJECT_ROOT}")
print(f"基础模型目录: {PRETRAINED_MODELS_DIR}")
print(f"Parsing 模型目录: {PARSING_CKPT_DIR}")
print("-" * 50)

def download_clip():
    print("\n[1/3] 正在下载 CLIP 模型 (openai/clip-vit-base-patch32)...")
    try:
        snapshot_download(
            repo_id="openai/clip-vit-base-patch32",
            local_dir=os.path.join(PRETRAINED_MODELS_DIR, "clip-vit-base-patch32"),
            local_dir_use_symlinks=False,
            # resume_download=True, # Deprecated
            etag_timeout=60,
        )
        print("✅ CLIP 下载完成")
    except Exception as e:
        print(f"❌ CLIP 下载失败: {e}")

def download_vae():
    print("\n[2/3] 正在下载 VAE 模型 (diffusers/sd-vae-ft-mse)...")
    try:
        snapshot_download(
            repo_id="diffusers/sd-vae-ft-mse",
            local_dir=os.path.join(PRETRAINED_MODELS_DIR, "sd-vae-ft-mse"),
            local_dir_use_symlinks=False,
            # resume_download=True, # Deprecated
            etag_timeout=60,
        )
        print("✅ VAE 下载完成")
    except Exception as e:
        print(f"❌ VAE 下载失败: {e}")

def download_parsing():
    print("\n[3/3] 正在下载 Human Parsing 模型 (yisol/IDM-VTON)...")
    files_to_download = [
        "humanparsing/parsing_atr.onnx",
        "humanparsing/parsing_lip.onnx"
    ]
    
    try:
        for file_path in files_to_download:
            print(f"  - 下载 {os.path.basename(file_path)}...")
            hf_hub_download(
                repo_id="yisol/IDM-VTON",
                filename=file_path,
                local_dir=os.path.join(PROJECT_ROOT, "src", "multiview_consist_edit", "parse_tool", "ckpt"),
                local_dir_use_symlinks=False,
                resume_download=True,
                repo_type="space"
            )
        print("✅ Human Parsing 下载完成")
    except Exception as e:
        print(f"❌ Human Parsing 下载失败: {e}")

if __name__ == "__main__":
    print("🚀 开始下载模型权重...")
    print("提示: 已自动启用 HF-Mirror 镜像加速下载。")
    
    # download_clip()
    # download_vae()
    download_parsing()
    
    print("\n🎉 所有下载任务结束！")
    print("请记得在 config/infer_tryon_multi.yaml 中更新 CLIP 和 VAE 的路径：")
    print(f"  clip_model_path: '{os.path.join(PRETRAINED_MODELS_DIR, 'clip-vit-base-patch32').replace(os.sep, '/')}'")
    print(f"  vae_path: '{os.path.join(PRETRAINED_MODELS_DIR, 'sd-vae-ft-mse').replace(os.sep, '/')}'")
