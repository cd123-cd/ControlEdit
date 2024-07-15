import torch
import numpy as np
from PIL import Image

#from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from diffusers import  AutoencoderKL
from diffusers.utils import load_image
from src.pipeline_controlnet_sd_xl import StableDiffusionXLControlNetPipeline
from src.controlnet import ControlNetModel
from src.image_processor import VaeImageProcessor


controlnet = ControlNetModel.from_pretrained(
    "/root/autodl-tmp/save/checkpoint-12000/controlnet",
    #variant="fp16",
    use_safetensors=True,
    torch_dtype=torch.float16,
)
vae = AutoencoderKL.from_pretrained("/root/autodl-tmp/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "/root/autodl-tmp/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    variant="fp16",
    use_safetensors=True,
    torch_dtype=torch.float16,
)
pipe.enable_model_cpu_offload()

# def get_depth_map(image):
#     image = feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
#     with torch.no_grad(), torch.autocast("cuda"):
#         depth_map = depth_estimator(image).predicted_depth

#     depth_map = torch.nn.functional.interpolate(
#         depth_map.unsqueeze(1),
#         size=(1024, 1024),
#         mode="bicubic",
#         align_corners=False,
#     )
    # depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    # depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    # depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    # image = torch.cat([depth_map] * 3, dim=1)
def apply_mask_to_image(image, mask, chose):
    mask=torch.tensor(mask,dtype=torch.float32)
    masked_image = np.array(image)

    if chose == 0:
        mask = 1 - np.array(mask)

    # Ensure mask has the same shape as the image
    if mask.shape[0] != masked_image.shape[0] or mask.shape[1] != masked_image.shape[1]:
        mask = np.resize(mask, (masked_image.shape[0], masked_image.shape[1]))

    # Apply the mask to each channel of the image
    for c in range(masked_image.shape[2]):
        masked_image[:, :, c] = masked_image[:, :, c] * mask

    masked_image = masked_image.astype(np.uint8)
    #masked_image=torch.tensor(masked_image,dtype=torch.float32)
    masked_image = Image.fromarray(masked_image)

    return masked_image


controlnet_conditioning_scale = 1.0 # recommended for good generalization

prompt = " "
image_sketch=Image.open("/root/autodl-tmp/diffusers/examples/controlnet/sketch.jpg").resize((1024, 1024)).convert("RGB")
image=Image.open("/root/autodl-tmp/diffusers/examples/controlnet/30.jpeg").resize((1024, 1024)).convert("RGB")
#image_sketch = load_image("/root/autodl-tmp/diffusers/examples/controlnet/sketch.jpg").resize((1024, 1024))
#image_sketch = image_sketch.permute(0, 2, 3, 1).cpu().numpy()[0]，
#image_sketch = Image.fromarray((image_sketch * 255.0).clip(0, 255).astype(np.uint8))
#image_mask = load_image("/root/autodl-tmp/diffusers/examples/controlnet/imagemask.png").resize((1024, 1024))
#image_mask=Image.open("/root/autodl-tmp/diffusers/examples/controlnet/imagemask.png").resize((1024, 1024)).convert("RGB")
mask=Image.open("/root/autodl-tmp/diffusers/examples/controlnet/mask.jpg").resize((1024, 1024))
#image_mask = image_mask.permute(0, 2, 3, 1).cpu().numpy()[0]
#image_mask = Image.fromarray((image_mask* 255.0).clip(0, 255).astype(np.uint8))
#mask = load_image("/root/autodl-tmp/diffusers/examples/controlnet/mask.jpg")
mask=mask.convert("L")
# 定义阈值，可以根据具体
# 情况调整
mask = 1-np.array(mask)//255.0   # 转换为二值掩码


image_mask=apply_mask_to_image(image, mask, 0)
mask = Image.fromarray(mask)

#mask = mask.permute(0, 2, 3, 1).cpu().numpy()[0]
#mask = Image.fromarray((mask * 255.0).clip(0, 255).astype(np.uint8))

# depth_image = get_depth_map(image)
print(type(image_sketch), type(image_mask), type(mask),image_sketch.size, image_mask.size, mask.size)
images = pipe(
    prompt, image_source=image,image_sketch=image_mask, image_mask=image_mask,mask=mask, num_inference_steps=30, controlnet_conditioning_scale=controlnet_conditioning_scale,
).images
images[0]

images[0].save(f"stormtrooper.png")