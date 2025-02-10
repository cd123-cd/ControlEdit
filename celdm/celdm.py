import einops
import torch
import torch.nn as nn
from cldm.cldm import ControlNet, ControlLDM, ControlledUnetModel
import os


from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

from ldm.modules.diffusionmodules.util import timestep_embedding

class CelControlledUnetModel(ControlledUnetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        return super().forward(x, timesteps, context, control, only_mid_control, **kwargs)

class CelControlNet(ControlNet):
    def forward(self, x, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        
        guided_hint = self.input_hint_block(hint, emb, context)
        
        outs = []
        h = x.type(self.dtype)
        
        for i, (module, zero_conv) in enumerate(zip(self.input_blocks, self.zero_convs)):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))
        
        return outs

class CelControlLDM(ControlLDM):
    def __init__(self, control_stage_config, control_key, random_bezier_mask, mask_aim, 
                 only_mid_control, *args, **kwargs):
        super().__init__(control_stage_config, control_key, only_mid_control, *args, **kwargs)
        self.random_bezier_mask = random_bezier_mask
        self.mask_aim = mask_aim

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        control = batch[self.control_key]
        random_bezier_mask = batch[self.random_bezier_mask]
        mask_aim = batch[self.mask_aim]

        if bs is not None:
            control = control[:bs]
            random_bezier_mask = random_bezier_mask[:bs]
            mask_aim = mask_aim[:bs]

        random_bezier_mask = random_bezier_mask.to(self.device)
        mask_aim = mask_aim.to(self.device)
        control = control.to(self.device)
        
        random_bezier_mask = random_bezier_mask.unsqueeze(1)
        mask_aim = einops.rearrange(mask_aim, 'b h w c -> b c h w')
        control = einops.rearrange(control, 'b h w c -> b c h w')
        
        random_bezier_mask = random_bezier_mask.to(memory_format=torch.contiguous_format).float()
        mask_aim = mask_aim.to(memory_format=torch.contiguous_format).float()
        control = control.to(memory_format=torch.contiguous_format).float()
        
        control = torch.cat((random_bezier_mask, mask_aim, control), dim=1)
        return x, dict(c_crossattn=[c], c_concat=[control])

    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, 
                  return_keys=None, **kwargs):
        log = super().log_images(batch, N, n_row, sample, ddim_steps, ddim_eta, 
                               return_keys, **kwargs)
        
        if "control" in log:
            c_cat = log["control"]
            index = torch.tensor([0, 1, 2]).to(c_cat.device)
            log["control"] = torch.index_select(c_cat.clone(), dim=1, index=index)
            
        return log

def get_state_dict(d):
    return d.get('state_dict', d)


def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict


def create_model(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).cpu()
    print(f'Loaded model config from [{config_path}]')
    return model
