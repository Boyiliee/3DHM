import joblib
import torch
from torchvision import transforms
import numpy as np
import os
from PIL import Image, ImageFilter
import joblib
from diffusers import StableDiffusionInpaintPipeline, DDPMScheduler, DDIMScheduler
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from ip_adapter import IPAdapter
from safetensors.torch import load_file

import argparse
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"
root = os.path.dirname(os.path.abspath(__file__))
uv_mask = Image.open(os.path.join(root, "data/smpl_uv_mask.png"))
enhance_mask = Image.open(os.path.join(root, "data/enhanced_mask.png"))

def load_ckpt(ckpt_path):
    pipe_stage1 = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16,
    )
    pipe_stage1.to(device)
    pipe_stage1.safety_checker=None

    pipe_stage1.text_encoder = add_lora_to_text_encoder(pipe_stage1.text_encoder)
    pipe_stage1.unet = add_lora_to_unet(pipe_stage1.unet)
    pipe_stage1.unet.eval()
    pipe_stage1.text_encoder.eval()

    lora_path = ckpt_path
    # Load weights you saved (UNet LoRA)
    unet_lora_sd = load_file(os.path.join(lora_path, "unet_lora_ema.safetensors"))
    set_peft_model_state_dict(pipe_stage1.unet, unet_lora_sd)

    # If you saved TE-LoRA:
    te_lora_sd = load_file(os.path.join(lora_path, "text_lora.safetensors"))
    set_peft_model_state_dict(pipe_stage1.text_encoder, te_lora_sd)
    return pipe_stage1

def get_mask_image(pkl_file, blur=False, enhance_mask=None):
    mask_image  = joblib.load(pkl_file)
    mask_image = mask_image.repeat(3,1,1).cpu()
    # Move the artifact from rasterized mask
    mask_image = transforms.GaussianBlur(kernel_size=(21, 21), sigma=(1.0, 1.0))(mask_image) > 0
    # Soft mask
    mask_image = mask_image.float()
    mask_image = 1.0 - mask_image
    if blur:
        mask_image = transforms.GaussianBlur(kernel_size=(33, 33), sigma=(10.0, 10.0))(mask_image) > 0
    if enhance_mask is not None:
        mask_image = (enhance_mask + mask_image).clamp(0, 1)
    mask_image = mask_image.float().numpy()
    mask_image *= np.array(uv_mask)/255
    mask_image = Image.fromarray((mask_image * 255).astype(np.uint8).transpose(1, 2, 0))
    return mask_image

def add_lora_to_unet(unet, r=64, alpha=64, dropout=0.05):
    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        bias="none",
        task_type="UNET",
    )
    unet = get_peft_model(unet, cfg)
    return unet

def add_lora_to_text_encoder(text_encoder, r=8, alpha=8, dropout=0.05):
    cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        bias="none",
        task_type="TEXT_ENCODER",
    )
    return get_peft_model(text_encoder, cfg)

def advanced_preprocess_texture(texture_pil, mask_pil):
    """"""
    texture = np.array(texture_pil)
    mask = np.array(mask_pil.convert('L'))
    mask = 255 - mask
    mask = (mask > 127).astype(np.uint8) * 255

    # 1. extend
    mask_inv = 255 - mask
    texture_dilated = cv2.dilate(texture, np.ones((5,5), np.uint8), iterations=3)

    # 2. inpainting
    if np.sum(mask_inv > 0) > 0:
        texture_inpainted = cv2.inpaint(texture_dilated, mask_inv, 5, cv2.INPAINT_NS)
    else:
        texture_inpainted = texture.copy()
    
    # 3. blender two level
    blurred_mask_1 = cv2.GaussianBlur(mask, (11, 11), 0)
    blurred_mask_2 = cv2.GaussianBlur(mask, (21, 21), 0)
    
    alpha_1 = blurred_mask_1[..., None] / 255.0
    alpha_2 = blurred_mask_2[..., None] / 255.0
    
    texture_blend = texture * alpha_1 + texture_inpainted * (1 - alpha_1)
    texture_blend = texture_blend * alpha_2 + texture_inpainted * (1 - alpha_2)
    
    return Image.fromarray(texture_blend.astype(np.uint8))

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_encoder_path = "./IP-Adapter/models/image_encoder/"
    ip_ckpt = "./IP-Adapter/models/ip-adapter_sd15.bin"

    # WITH UNET + TEXT ENCODER
    ckpt_path = os.path.join(root, "models/stage1")

    pipe_stage1 = load_ckpt(ckpt_path).to(device)
    pipe_stage1.safety_checker=None

    ip_model = IPAdapter(pipe_stage1, image_encoder_path, ip_ckpt, device)
    
    
    ref_folder = args.reference_imgs
    for subject_name in os.listdir(ref_folder):
        subject_dir = os.path.join(ref_folder, subject_name)
        if not os.path.isdir(subject_dir):
            print("subject_dir is not a directory, skipping:", subject_dir)
            continue

        texture_path = os.path.join(subject_dir, "texture.png")
        mask_path = os.path.join(subject_dir, "mask.pkl")
        print(f"Texture path: {texture_path}")
        if os.path.exists(texture_path) and os.path.exists(mask_path):
            print(f"Processing: {texture_path}")
            
            init_image = Image.open(texture_path).convert("RGB")
            mask_image = get_mask_image(mask_path, blur=True, enhance_mask=enhance_mask)
            cv_inpainted_image = advanced_preprocess_texture(init_image, mask_image)

            # inpainting
            result_image = ip_model.generate(
                pil_image=cv_inpainted_image, 
                num_samples=1,
                prompt='3D realistic human, UV texturemap', 
                num_inference_steps=50, 
                seed=42, 
                guidance_scale=3.5, 
                image=cv_inpainted_image, 
                scale=1, 
                mask_image=mask_image,
                strength=0.8, 
                height=512, 
                width=512
            )[0]
            
            # Replace original texture.png
            result_image.save(texture_path)
            print(f"Saved inpainted texture to: {texture_path}")
        else:
            print(f"Skipping {subject_dir}, texture.png or mask.pkl not found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage-1 Inpainting for 3DHM")
    parser.add_argument('reference_imgs', type=str, help='Path to the reference images folder.')
    args = parser.parse_args()
    
    main(args)
