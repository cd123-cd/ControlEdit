"""SAMPLING ONLY."""
from torchvision import transforms
import torch
import numpy as np
from einops import einops
from tqdm import tqdm
import os
from PIL import Image
import torch.nn.functional as F
from cldm.ddim_hacked import DDIMSampler as BaseDDIMSampler

class DDIMSampler(BaseDDIMSampler):
    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, dynamic_threshold=None,
                      ucg_schedule=None):
        device = self.model.betas.device
        b = shape[0]
        org_mask = cond.pop("org_mask", None)
        init_image = cond.pop("init_image", None)
        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]
        time_range = (
            reversed(range(0, timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        )
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        if init_image is not None:
            assert (
                    x0 is None and x_T is None
            ), "Try to infer x0 and x_t from init_image, but they already provided"
            init_image = init_image.to(device)
            encoder_posterior = self.model.encode_first_stage(init_image)
            x0 = self.model.get_first_stage_encoding(encoder_posterior)
            last_ts = torch.full((1,), time_range[0], device=device, dtype=torch.long)
            x_T = torch.cat([self.model.q_sample(x0, last_ts) for _ in range(b)])
            img = x_T
        elif x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = {'x_inter': [img], 'pred_x0': [img]}

        save_dir = "./image_results"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                n_masks = mask.shape[0]
                masks_interval = len(time_range) // n_masks + 1
                curr_mask = mask[i // masks_interval].unsqueeze(0)
                curr_mask = curr_mask.to(device)
                img_orig = self.model.q_sample(x0, ts)
                img = img_orig * (1 - curr_mask) + curr_mask * img

            if ucg_schedule is not None:
                assert len(ucg_schedule) == len(time_range)
                unconditional_guidance_scale = ucg_schedule[i]

            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      dynamic_threshold=dynamic_threshold)
            img, pred_x0 = outs
            
            # 保存中间结果
            imgs = self.model.decode_first_stage(img)
            imgs = (einops.rearrange(imgs, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            imgs = np.squeeze(imgs)
            img_filename = os.path.join(save_dir, f"img_step_{step}_iter_{i}.png")
            Image.fromarray(imgs).save(img_filename)

            # 处理原始mask
            if org_mask is None:
                org_mask = org_mask.to(device)
                foreground_pixels = self.model.decode_first_stage(pred_x0)
                background_pixels = init_image

                pixel_blended = foreground_pixels * org_mask + background_pixels * (1 - org_mask)
                
                # 保存混合结果
                imgs = (einops.rearrange(pixel_blended, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
                imgs = np.squeeze(imgs)
                img_filename = os.path.join(save_dir, f"blended_step_{step}_iter_{i}.png")
                Image.fromarray(imgs).save(img_filename)

                # 保存前景
                fg_imgs = (einops.rearrange(foreground_pixels, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
                Image.fromarray(np.squeeze(fg_imgs)).save(os.path.join(save_dir, f"foreground_pixels_step_{step}_iter_{i}.png"))

                # 保存背景
                bg_imgs = (einops.rearrange(background_pixels, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
                Image.fromarray(np.squeeze(bg_imgs)).save(os.path.join(save_dir, f"background_pixels_step_{step}_iter_{i}.png"))

                # 重新编码混合结果
                img_x0 = self.model.get_first_stage_encoding(
                    self.model.encode_first_stage(pixel_blended)
                )
                img = self.model.q_sample(img_x0, ts)

            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates