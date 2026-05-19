import PIL
from PIL import Image
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random
import copy
import time
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionInstructPix2PixPipeline, DDIMScheduler
from torchvision.utils import make_grid as make_image_grid
from torchvision.utils import save_image
from models.condition_encoder import FrozenOpenCLIPImageEmbedderV2
from omegaconf import OmegaConf
from pipelines.pipeline_tryon_multi import TryOnPipeline
from models.hack_poseguider import Hack_PoseGuider as PoseGuider

from models.ReferenceNet import ReferenceNet
from models.ReferenceEncoder import ReferenceEncoder

from data.Thuman2_multi import Thuman2_Dataset, collate_fn
# from data.Thuman2_multi_ps2 import Thuman2_Dataset, collate_fn
from data.MVHumanNet_multi import MVHumanNet_Dataset
from models.hack_unet2d import Hack_UNet2DConditionModel as UNet2DConditionModel

config = OmegaConf.load('config/infer_tryon_multi.yaml')

# seed 
seed = config.seed
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# dataset
infer_data_config = config.infer_data
if 'mvhumannet' in infer_data_config['dataroot']:
    infer_dataset = MVHumanNet_Dataset(**infer_data_config)
    print('using mvhumannet')
else:
    infer_dataset = Thuman2_Dataset(**infer_data_config)
    print('using Thuman2_Dataset')

batch_size = config.batch_size
# multi_length = 16

test_dataloader = torch.utils.data.DataLoader(
    infer_dataset,
    shuffle=False,
    collate_fn=collate_fn,
    batch_size=config.batch_size,
    num_workers=config.dataloader_num_workers,
)

unet = UNet2DConditionModel.from_pretrained(
    config.unet_path, subfolder="unet",torch_dtype=torch.float16
).to("cuda")
# unet = UNet2DConditionModel.from_pretrained(
# config.unet_path, subfolder=None,torch_dtype=torch.float16
# ).to("cuda")

vae= AutoencoderKL.from_pretrained(
    config.vae_path, subfolder="vae",torch_dtype=torch.float16
).to("cuda")

referencenet = ReferenceNet.from_pretrained(
    config.pretrained_referencenet_path, subfolder="referencenet",torch_dtype=torch.float16
).to("cuda")
# referencenet = ReferenceNet.load_referencenet(pretrained_model_path=config.pretrained_referencenet_path).to("cuda", dtype=torch.float16)

pose_guider = PoseGuider.from_pretrained(pretrained_model_path=config.pretrained_poseguider_path).to("cuda", dtype=torch.float16)
pose_guider.eval()
scheduler = DDIMScheduler.from_pretrained(config.model_path, subfolder='scheduler')

pipe = TryOnPipeline(pose_guider=pose_guider, referencenet=referencenet, vae=vae, unet=unet, scheduler=scheduler)
pipe.enable_xformers_memory_efficient_attention()
# pipe._execution_device = torch.device("cuda")
# pipe.to("cuda")

clip_image_encoder = ReferenceEncoder(model_path=config.clip_model_path).to(device='cuda',dtype=torch.float16)

pipe.scheduler = DDIMScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    )
generator = torch.Generator("cuda").manual_seed(seed)

# infer
out_dir = config.out_dir
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

num_inference_steps = config.num_inference_steps
guidance_scale = config.guidance_scale
weight_dtype = torch.float16

# # check vae reconstruction
# image_idx = 0
# for i, batch in enumerate(test_dataloader):
#     video = batch['pixel_values'].to(device='cuda', dtype=torch.float16)
#     out = video[0].cpu() /2 +0.5
#     out = out.detach().permute(1,2,0).numpy()
#     out = (out * 255).astype(np.uint8)
#     out = Image.fromarray(out)
#     out.save('%d_test_ori.png' % i)

#     latents = vae.encode(video)
#     latents = latents.latent_dist.sample()

#     reconstruct_video = vae.decode(latents).sample

#     reconstruct_video = reconstruct_video.clamp(-1, 1)
#     out = reconstruct_video[0].cpu() /2 +0.5
#     out = out.detach().permute(1,2,0).numpy()
#     out = (out * 255).astype(np.uint8)
#     out = Image.fromarray(out)
#     out.save('%d_test2.png' % i)


image_idx = 0
for i, batch in enumerate(test_dataloader):

    pixel_values = batch["pixel_values"]
    pixel_values_pose = batch["pixel_values_pose"].to(device='cuda')
    pixel_values_agnostic = batch["pixel_values_agnostic"].to(device='cuda')
    clip_ref_front = batch["clip_ref_front"].to(device='cuda')
    clip_ref_back = batch["clip_ref_back"].to(device='cuda')
    pixel_values_ref_front = batch["pixel_values_ref_front"].to(device='cuda')
    pixel_values_ref_back = batch["pixel_values_ref_back"].to(device='cuda')
    camera_pose = batch["camera_parm"]
    img_name = batch["img_name"]
    cloth_name = batch["cloth_name"]
    multi_length = pixel_values.shape[1]
    # dino_fea = dino_fea.unsqueeze(1)
    # print(dino_fea.shape) # [bs,1,768]
    print(img_name)

    bs = int(pixel_values_pose.shape[0])
    # For inference on MVHumanNet, multi_length is typically 16 (predefined camera list).
    # The effective batch in the pipeline is (bs * multi_length). Large values easily OOM.
    # Keep the config batch size for dataloader throughput, but run the diffusion model per-sample
    # when the effective batch becomes too large.
    max_effective_bf = 16
    run_per_sample = (bs * int(multi_length)) > max_effective_bf

    def _run_one_sample(bi: int):
        global image_idx
        clip_f = clip_ref_front[bi:bi + 1].to(weight_dtype)
        clip_b = clip_ref_back[bi:bi + 1].to(weight_dtype)
        front_dino_fea = clip_image_encoder(clip_f)
        back_dino_fea = clip_image_encoder(clip_b)

        # keep determinism independent of dataloader batching
        local_gen = torch.Generator("cuda").manual_seed(int(seed) + 1000 * bi)

        edited_images = pipe(
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            front_image=pixel_values_ref_front[bi:bi + 1].to(weight_dtype),
            back_image=pixel_values_ref_back[bi:bi + 1].to(weight_dtype),
            pose_image=pixel_values_pose[bi:bi + 1].to(weight_dtype),
            camera_pose=camera_pose[bi:bi + 1],
            agnostic_image=pixel_values_agnostic[bi:bi + 1].to(weight_dtype),
            generator=local_gen,
            front_dino_fea=front_dino_fea,
            back_dino_fea=back_dino_fea,
        ).images

        cloth_id = cloth_name[bi].split('/')[-1].split('_')[0]
        sub_cloth_root = os.path.join(out_dir, cloth_id)
        if not os.path.exists(sub_cloth_root):
            os.makedirs(sub_cloth_root)

        start = bi * int(multi_length)
        for fi in range(int(multi_length)):
            total_idx = start + fi
            name = img_name[total_idx].replace('/', '_')
            edited_image = edited_images[fi]
            edited_image = torch.tensor(np.array(edited_image)).permute(2, 0, 1) / 255.0
            grid = make_image_grid(
                [
                    (pixel_values[bi][fi].cpu() / 2 + 0.5),
                    edited_image.cpu(),
                    (pixel_values_pose[bi][fi].cpu() / 2 + 0.5),
                    (pixel_values_agnostic[bi][fi].cpu() / 2 + 0.5),
                    (pixel_values_ref_front[bi].cpu() / 2 + 0.5),
                    (pixel_values_ref_back[bi].cpu() / 2 + 0.5),
                ],
                nrow=2,
            )
            print(name, cloth_id)
            save_image(edited_image, os.path.join(sub_cloth_root, name))
            save_image(grid, os.path.join(sub_cloth_root, 'cond_' + name))
            image_idx += 1

    if run_per_sample:
        for bi in range(bs):
            _run_one_sample(bi)
    else:
        front_dino_fea = clip_image_encoder(clip_ref_front.to(weight_dtype))
        back_dino_fea = clip_image_encoder(clip_ref_back.to(weight_dtype))

        edited_images = pipe(
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            front_image=pixel_values_ref_front.to(weight_dtype),
            back_image=pixel_values_ref_back.to(weight_dtype),
            pose_image=pixel_values_pose.to(weight_dtype),
            camera_pose=camera_pose,
            agnostic_image=pixel_values_agnostic.to(weight_dtype),
            generator=generator,
            front_dino_fea=front_dino_fea,
            back_dino_fea=back_dino_fea,
        ).images

        for batch_idx in range(bs):
            cloth_id = cloth_name[batch_idx].split('/')[-1].split('_')[0]
            sub_cloth_root = os.path.join(out_dir, cloth_id)
            if not os.path.exists(sub_cloth_root):
                os.makedirs(sub_cloth_root)

            for fi in range(multi_length):
                total_idx = batch_idx * multi_length + fi
                name = img_name[total_idx].replace('/', '_')
                edited_image = edited_images[total_idx]
                edited_image = torch.tensor(np.array(edited_image)).permute(2, 0, 1) / 255.0
                grid = make_image_grid(
                    [
                        (pixel_values[batch_idx][fi].cpu() / 2 + 0.5),
                        edited_image.cpu(),
                        (pixel_values_pose[batch_idx][fi].cpu() / 2 + 0.5),
                        (pixel_values_agnostic[batch_idx][fi].cpu() / 2 + 0.5),
                        (pixel_values_ref_front[batch_idx].cpu() / 2 + 0.5),
                        (pixel_values_ref_back[batch_idx].cpu() / 2 + 0.5),
                    ],
                    nrow=2,
                )
                print(name, cloth_id)
                save_image(edited_image, os.path.join(sub_cloth_root, name))
                save_image(grid, os.path.join(sub_cloth_root, 'cond_' + name))
                image_idx += 1
