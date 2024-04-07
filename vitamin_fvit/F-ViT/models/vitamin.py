import open_clip
from functools import partial
import os
import torch
from torch import nn
from mmdet.models.builder import BACKBONES
from mmcv.runner import BaseModule
from torch.nn import functional as F
from mmcv.utils.logging import print_log
from mmcv.cnn import build_norm_layer
import math


@BACKBONES.register_module()
class CLIPViTamin(BaseModule):
    def __init__(self, model_name, pretrained, out_indices=[3, 5, 7, 11], norm_cfg=None):
        super().__init__()
        self.vit_layers = out_indices
        self.model_name = model_name
        self.pretrained = pretrained
        clip_model = open_clip.create_model(model_name, pretrained=pretrained,)
        
        # TODO: export to config files
        self.embed_dim = embed_dim = 768 
        self.width = width = 1024
        self.window_size = 336
        self.interpolate1 = nn.Sequential(
            nn.Conv2d(160, 160, kernel_size=3, stride=1, padding=1),
            build_norm_layer(norm_cfg, 160)[1] if norm_cfg else nn.Identity(),
            nn.GELU(),
            nn.Conv2d(160, 160, kernel_size=3, stride=1, padding=1),
        )
        self.interpolate2 = nn.Sequential(
            nn.Conv2d(320, 320, kernel_size=3, stride=1, padding=1),
        )
        self.interpolate3 = nn.Identity()
        self.interpolate4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.visual = clip_model.visual

    def init_weights(self):
        clip_model = open_clip.create_model(self.model_name,
                                            pretrained=self.pretrained,
                                            device="cpu")
        print_log(self.visual.load_state_dict(clip_model.visual.state_dict(), strict=True))
        for param in self.visual.parameters():  # only freeze the CLIP model
            param.requires_grad = False

    def train(self, mode=True):
        # print(f"Set train mode for CLIP ViTamin: {mode}", flush=True)
        self.training = mode
        self.visual.train(False)
        self.interpolate1.train(mode)
        self.interpolate2.train(mode)
        self.interpolate3.train(mode)
        self.interpolate4.train(mode)

        return self

    @torch.no_grad()
    def forward_per_window(self, x):
        def map1dto2d(x_1d):
            if len(x_1d.shape) == 4:
                assert x_1d.shape[-1] == x_1d.shape[-2]
                return x_1d
            else:
                assert len(x_1d.shape) == 3
                b, l, c = x_1d.shape
                h = w = int(math.sqrt(l))
                assert h * w == l
                return x_1d.permute(0, 2, 1).reshape(b, c, h, w)
        outs= []   
        x = self.visual.trunk.patch_embed.backbone.stem(x)
        x = self.visual.trunk.patch_embed.backbone.stages[0](x)
        outs.append(map1dto2d(x)) # os4
        x = self.visual.trunk.patch_embed.backbone.stages[1](x)
        outs.append(map1dto2d(x)) # os8
        x = self.visual.trunk.patch_embed.backbone.pool(x)
        x = self.visual.trunk.patch_embed.proj(x)
        x = x.flatten(2).transpose(1, 2)
        if self.visual.trunk.is_pos_embed:
            x = self.visual.trunk._pos_embed(x)
        x = self.visual.trunk.patch_drop(x)
        x = self.visual.trunk.norm_pre(x)
        x = self.visual.trunk.blocks[:len(self.visual.trunk.blocks)//2](x)
        outs.append(map1dto2d(x)) # os16
        x = self.visual.trunk.blocks[len(self.visual.trunk.blocks)//2:](x)
        outs.append(map1dto2d(x)) # os16 for os 32
        x = self.visual.trunk.norm(x)
        outs.append(map1dto2d(x)) # os16, final feat
        return outs

    def forward(self, x):
        batch_size, c_img, h_img, w_img = x.shape
        h_crop = w_crop = self.window_size
        h_stride = w_stride = (self.window_size // 2)
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        outs = []
        count_mats = []
        for patch_size, channels in zip([4, 8, 16, 16, 16], [160, 320, 1024, 1024, 1024]):
            outs.append(torch.zeros((batch_size, channels, h_img//patch_size, w_img//patch_size), dtype=x.dtype, device=x.device))
            count_mats.append(torch.zeros((batch_size, channels, h_img//patch_size, w_img//patch_size), dtype=x.dtype, device=x.device))

        with torch.no_grad():
            for h_idx in range(h_grids):
                for w_idx in range(w_grids):
                    y1 = h_idx * h_stride
                    x1 = w_idx * w_stride
                    y2 = min(y1 + h_crop, h_img)
                    x2 = min(x1 + w_crop, w_img)
                    y1 = max(y2 - h_crop, 0)
                    x1 = max(x2 - w_crop, 0)
                    crop_x = x[:, :, y1:y2, x1:x2]
                    tmp_out = self.forward_per_window(crop_x)
                    for i, patch_size in enumerate([4, 8, 16, 16, 16]):
                        #print(outs[i].shape, i, y1//patch_size, y2//patch_size, x1//patch_size, x2//patch_size, tmp_out[i].shape)
                        outs[i][:, :, y1//patch_size:y2//patch_size, x1//patch_size:x2//patch_size] += tmp_out[i]
                        count_mats[i][:, :, y1//patch_size:y2//patch_size, x1//patch_size:x2//patch_size] += 1
            assert all((count_mat == 0).sum() == 0 for count_mat in count_mats)
            for i in range(5):
                outs[i] /= count_mats[i]

            if not self.training:
                x = outs[-1] # B C H W
                x = x.permute(0, 2, 3, 1) # B H W C
                x = self.visual.trunk.fc_norm(x)
                x = self.visual.trunk.head_drop(x)
                x = self.visual.trunk.head(x)
                x = self.visual.head(x)
                x = F.normalize(x, dim=-1)  # normalize along last dimension
                feature_map = x.view(batch_size, h_img//16, w_img//16, -1).permute(0, 3, 1, 2)
            else:
                feature_map = None
        outs = outs[:-1]
        assert len(outs) == 4
        for idx, out in enumerate(outs):
            interpolate = getattr(self, f"interpolate{idx + 1}")
            outs[idx] = interpolate(out.detach())

        outs.append(feature_map)

        return tuple(outs)
    
if __name__ == "__main__":
    # cd /data/jieneng/vitamin_opensource/fvit
    # PYTHONPATH="./" python3 F-ViT/models/evaclip_vit.py
    clip_model = open_clip.create_model("ViTamin-L-336", pretrained=False,) # bug-free