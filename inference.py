from share import *
import config
import os
import cv2
import einops
import numpy as np
import torch
import random
from PIL import Image
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from celdm.celdm import create_model, load_state_dict, CelControlLDM
from celdm.ddim_hacked import DDIMSampler
from torchvision.transforms.functional import to_tensor

# 初始化模型
model_name = 'controledit_v11p_sd15_lineart'
model = create_model(f'./models/{model_name}.yaml').cpu()
model.load_state_dict(load_state_dict('./models/v1-5-pruned.ckpt', location='cuda'), strict=False)
model.load_state_dict(load_state_dict(f'./models/controledit_sd1.5_v1.ckpt', location='cuda'), strict=False)
model = model.cuda()
ddim_sampler = DDIMSampler(model)

def read_mask(mask_path: str, dilation_iterations: int = 0, dest_size=(64, 64), img_size=(512, 512)):
    org_mask = Image.open(mask_path).convert("L")
    mask = org_mask.resize(dest_size, Image.NEAREST)
    mask = 1-np.array(mask) / 255

    masks_array = []
    for i in reversed(range(dilation_iterations)):
        k_size = 3 + 2 * i
        masks_array.append(binary_dilation(mask, structure=np.ones((k_size, k_size))))
    masks_array.append(mask)
    masks_array = np.array(masks_array).astype(np.float32)
    masks_array = masks_array[:, np.newaxis, :]
    masks_array = torch.from_numpy(masks_array)

    org_mask = org_mask.resize(img_size, Image.LANCZOS)
    org_mask = np.array(org_mask).astype(np.float32) / 255.0
    org_mask = org_mask[None, None]
    org_mask[org_mask < 0.5] = 0
    org_mask[org_mask >= 0.5] = 1
    org_mask = torch.from_numpy(org_mask)

    return masks_array, org_mask

def read_image(img_path: str, dest_size=(512, 512)):
    image = Image.open(img_path).convert("RGB")
    image = image.resize(dest_size, Image.LANCZOS)
    image = np.array(image)
    image = image.astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    image = image * 2.0 - 1.0
    return image

def process(input_image, mask, init_image, img_mask, prompt, a_prompt, n_prompt, num_samples, 
           image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    with torch.no_grad():
        input_image = Image.open(input_image)
        input_image = np.array(input_image)
        input_image = HWC3(input_image)
        
        img_mask = Image.open(img_mask)
        img_mask = np.array(img_mask)
        img_mask = cv2.cvtColor(np.array(img_mask), cv2.COLOR_BGR2RGB)
        
        init_image = read_image(init_image)

        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        control = torch.from_numpy(input_image.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        img_mask = torch.from_numpy(img_mask).float().cuda() / 255.0
        img_mask = einops.rearrange(img_mask.unsqueeze(0), 'b h w c -> b c h w').clone()

        mask_c = Image.open(mask).convert("L")
        mask_c = np.array(mask_c).astype(np.float32)/255.0
        mask_c = torch.from_numpy(mask_c)
        mask_c = mask_c.unsqueeze(0).unsqueeze(0)
        
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        mask_c = mask_c.to(device)
        img_mask = img_mask.to(device)
        control = control.to(device)
        
        control = torch.cat((mask_c, img_mask, control), dim=1)

        mask_size = (W//8, H//8)
        img_size = (W, H)
        mask, org_mask = read_mask(
            mask_path=mask,
            dilation_iterations=0,
            dest_size=mask_size,
            img_size=img_size
        )

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {
            "c_concat": [control],
            "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)],
            "org_mask": org_mask,
            "init_image": init_image
        }
        un_cond = {
            "c_concat": None if guess_mode else [control],
            "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]
        }
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)

        samples, intermediates = ddim_sampler.sample(
            ddim_steps, num_samples, shape, cond, mask=mask, verbose=False,
            eta=eta, unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond
        )

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
        
        return results

def process_file(input_image_path, mask_path, init_image_path, image_mask_path, output_folder,
                prompt="", a_prompt="best quality", 
                n_prompt="lowres, bad anatomy, bad hands, cropped, worst quality",
                num_samples=1, image_resolution=512, detect_resolution=512, ddim_steps=20,
                guess_mode=False, strength=1.0, scale=9.0, seed=-1, eta=1.0):
    
    results = process(
        input_image_path, mask_path, init_image_path, image_mask_path,
        prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution,
        ddim_steps, guess_mode, strength, scale, seed, eta
    )
    return results

def main():

    input_sketch_folder = "./examples/sketch"
    input_real_image_folder = "./examples/real"
    image_mask_folder = "./examples/imagemask"
    output_folder = "./output"
    
   
    mask_path = "./examples/mask/1.jpg"  
    
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取所有输入文件
    sketch_files = sorted(os.listdir(input_sketch_folder))
    real_image_files = sorted(os.listdir(input_real_image_folder))
    image_mask_files = sorted(os.listdir(image_mask_folder))
    
    # 验证文件数量匹配
    total_files = len(sketch_files)
    if not (len(real_image_files) == len(image_mask_files) == total_files):
        print("Error: Number of files in input folders don't match!")
        return
    
    print(f"Starting processing of {total_files} images...")
    
    # 处理每个图像
    for idx, (sketch_file, real_image_file, image_mask_file) in enumerate(
        zip(sketch_files, real_image_files, image_mask_files), 1):
        try:
            print(f"\nProcessing image {idx}/{total_files}: {sketch_file}")
            
            # 构建完整的文件路径
            sketch_path = os.path.join(input_sketch_folder, sketch_file)
            real_image_path = os.path.join(input_real_image_folder, real_image_file)
            image_mask_path = os.path.join(image_mask_folder, image_mask_file)
            
            # 构建输出文件路径
            output_file_name = os.path.splitext(sketch_file)[0] + ".png"
            output_file_path = os.path.join(output_folder, output_file_name)
            
            # 检查输出文件是否已存在
            if os.path.exists(output_file_path):
                print(f"Skipping {sketch_file}, output file already exists.")
                continue
            
            # 处理图像
            results = process_file(
                sketch_path, mask_path, real_image_path, image_mask_path, output_folder
            )
            
            # 保存结果
            for i, result in enumerate(results):
                output_path = output_file_path
                if len(results) > 1:
                    base, ext = os.path.splitext(output_file_path)
                    output_path = f"{base}_{i}{ext}"
                
                pil_image = Image.fromarray(result)
                pil_image.save(output_path)
                print(f"Saved output to: {output_path}")
            
        except Exception as e:
            print(f"Error processing {sketch_file}: {str(e)}")
            continue
    
    print("\nProcessing completed!")

if __name__ == "__main__":
    main()