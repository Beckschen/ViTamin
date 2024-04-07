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
class EvaCLIPViT(BaseModule):
    def __init__(self, model_name, pretrained, out_indices=[3, 5, 7, 11], norm_cfg=None):
        super().__init__()
        self.vit_layers = out_indices
        self.model_name = model_name
        self.pretrained = pretrained  # the pretrained .pt file
        clip_model = open_clip.create_model(model_name,
                                            pretrained="eva",
                                            cache_dir=pretrained)
        self.embed_dim = embed_dim = clip_model.embed_dim  # output dim
        self.width = width = clip_model.visual.embed_dim
        self.patch_size = patch_size = clip_model.visual.patch_embed.patch_size[0]
        self.interpolate1 = nn.Sequential(
            nn.ConvTranspose2d(width, width, kernel_size=2, stride=2),
            build_norm_layer(norm_cfg, width)[1] if norm_cfg else nn.Identity(),
            nn.GELU(),
            nn.ConvTranspose2d(width, width, kernel_size=2, stride=2),
        )
        self.interpolate2 = nn.Sequential(
            nn.ConvTranspose2d(width, width, kernel_size=2, stride=2),
        )
        self.interpolate3 = nn.Identity()
        self.interpolate4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.visual = clip_model.visual
        # self.interpolate3 = nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1)
        # self.interpolate4 = nn.Conv2d(width, width, kernel_size=3, stride=2, padding=1)

    def init_weights(self):
        clip_model = open_clip.create_model(self.model_name,
                                            pretrained="eva",
                                            cache_dir=self.pretrained,
                                            device="cpu")
        print_log(self.visual.load_state_dict(clip_model.visual.state_dict(), strict=True))
        for param in self.visual.parameters():  # only freeze the CLIP model
            param.requires_grad = False

    def train(self, mode=True):
        print(f"Set train mode for EVA: {mode}", flush=True)
        self.training = mode
        self.visual.train(False)
        self.interpolate1.train(mode)
        self.interpolate2.train(mode)
        self.interpolate3.train(mode)
        self.interpolate4.train(mode)

        return self

    def forward(self, x):
        visual = self.visual
        bs, _, h, w = x.shape
        h = h // visual.patch_embed.patch_size[0]
        w = w // visual.patch_embed.patch_size[1]

        with torch.no_grad():
            x = visual.patch_embed(x)
            batch_size, seq_len, _ = x.size()

            cls_tokens = visual.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)
            if visual.pos_embed is not None:
                x = x + visual.rescale_positional_embedding(out_size=(h, w))
            x = visual.pos_drop(x)

            # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
            if os.getenv('RoPE') == '1':
                if visual.training and not isinstance(visual.patch_dropout, nn.Identity):
                    x, patch_indices_keep = visual.patch_dropout(x)
                    visual.rope.forward = partial(visual.rope.forward, patch_indices_keep=patch_indices_keep)
                else:
                    visual.rope.forward = partial(visual.rope.forward, patch_indices_keep=None)
                    x = visual.patch_dropout(x)
            else:
                x = visual.patch_dropout(x)

            rel_pos_bias = visual.rel_pos_bias() if visual.rel_pos_bias is not None else None

            outs = []
            for i, blk in enumerate(visual.blocks[:-1]):
                x = blk(x, rel_pos_bias=rel_pos_bias)
                if i in self.vit_layers:
                    outs.append(self._expand_x(x, h, w))
            x = visual.blocks[-1].forward_without_attn(x)
            if (len(visual.blocks) - 1) in self.vit_layers:
                outs.append(self._expand_x(x, h, w))
            if not self.training:
                x = x[:, 1:]
                x = visual.norm(x)
                x = visual.head(x)
                assert visual.fc_norm is None
                x = F.normalize(x, dim=-1)  # normalize along last dimension
                feature_map = x.view(bs, h, w, -1).permute(0, 3, 1, 2)
            else:
                feature_map = None

        assert len(outs) == 4
        for idx, out in enumerate(outs):
            interpolate = getattr(self, f"interpolate{idx + 1}")
            outs[idx] = interpolate(out.detach())

        outs.append(feature_map)

        return tuple(outs)

    def _expand_x(self, x, h, w):
        # x: bs q c
        x = x[:, 1:].permute(0, 2, 1).contiguous()
        x = x.view(-1, self.width, h, w)

        return x



@BACKBONES.register_module()
class CLIPViT(BaseModule):
    def __init__(self, model_name, pretrained, out_indices=[3, 5, 7, 11], norm_cfg=None):
        super().__init__()
        self.vit_layers = out_indices
        self.model_name = model_name
        self.pretrained = pretrained
        
        clip_model = open_clip.create_model(model_name,
                                            pretrained=pretrained,)
        self.embed_dim = embed_dim = 768  # output dim
        self.width = width = 1024
        self.patch_size = patch_size = 14
        self.window_size = 336
        self.interpolate1 = nn.Sequential(
            nn.ConvTranspose2d(width, width, kernel_size=2, stride=2),
            build_norm_layer(norm_cfg, width)[1] if norm_cfg else nn.Identity(),
            nn.GELU(),
            nn.ConvTranspose2d(width, width, kernel_size=2, stride=2),
        )
        self.interpolate2 = nn.Sequential(
            nn.ConvTranspose2d(width, width, kernel_size=2, stride=2),
        )
        self.interpolate3 = nn.Identity()
        self.interpolate4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.visual = clip_model.visual
        # self.interpolate3 = nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1)
        # self.interpolate4 = nn.Conv2d(width, width, kernel_size=3, stride=2, padding=1)

    def init_weights(self):
        clip_model = open_clip.create_model(self.model_name,
                                            pretrained=self.pretrained,
                                            device="cpu")
        print_log(self.visual.load_state_dict(clip_model.visual.state_dict(), strict=True))
        for param in self.visual.parameters():  # only freeze the CLIP model
            param.requires_grad = False

    def train(self, mode=True):
        print(f"Set train mode for CLIP ViT: {mode}", flush=True)
        self.training = mode
        self.visual.train(False)
        self.interpolate1.train(mode)
        self.interpolate2.train(mode)
        self.interpolate3.train(mode)
        self.interpolate4.train(mode)

        return self

    @torch.no_grad()
    def forward_per_window(self, x):
        def map1d_to_2d(x):
            x = x.permute(1, 0, 2)
            # note that we need to exclude cls token
            B, L, C = x.shape
            height = width = int(math.sqrt(L))
            x = x[:, -height*width:, :] # remove the potential cls token
            return x.permute(0, 2, 1).reshape(B, C, height, width)
        
        x = self.visual.conv1(x) # B x C x H x W
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        def _expand_token(token, batch_size: int):
            return token.view(1, 1, -1).expand(batch_size, -1, -1)
        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.visual.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]
        x = x + self.visual.positional_embedding.to(x.dtype)

        x = self.visual.patch_dropout(x)
        x = self.visual.ln_pre(x)

        outs = []
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i, r in enumerate(self.visual.transformer.resblocks):
            x = r(x)
            if i in self.vit_layers:
                outs.append(map1d_to_2d(x))
        
        outs.append(map1d_to_2d(x))
        return outs

    def forward(self, x):
        batch_size, c_img, h_img, w_img = x.shape
        h_crop = w_crop = self.window_size
        h_stride = w_stride = (self.window_size // 2)
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        outs = []
        count_mats = []
        for i in range(5):
            outs.append(torch.zeros((batch_size, self.width, h_img//self.patch_size, w_img//self.patch_size), dtype=x.dtype, device=x.device))
            count_mats.append(torch.zeros((batch_size, self.width, h_img//self.patch_size, w_img//self.patch_size), dtype=x.dtype, device=x.device))

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
                    for i in range(5):
                        outs[i][:, :, y1//14:y2//14, x1//14:x2//14] += tmp_out[i]
                        count_mats[i][:, :, y1//14:y2//14, x1//14:x2//14] += 1
            assert all((count_mat == 0).sum() == 0 for count_mat in count_mats)
            for i in range(5):
                outs[i] /= count_mats[i]

            if not self.training:
                x = outs[-1] # B C H W
                x = x.permute(0, 2, 3, 1) # B H W C
                x = self.visual.ln_post(x)
                if self.visual.proj is not None:
                    x = x @ self.visual.proj
                x = F.normalize(x, dim=-1)  # normalize along last dimension
                feature_map = x.view(batch_size, h_img//self.patch_size, w_img//self.patch_size, -1).permute(0, 3, 1, 2)
            else:
                feature_map = None
        outs = outs[:-1]
        assert len(outs) == 4
        for idx, out in enumerate(outs):
            interpolate = getattr(self, f"interpolate{idx + 1}")
            outs[idx] = interpolate(out.detach())

        outs.append(feature_map)

        return tuple(outs)
    
@BACKBONES.register_module()
class CLIPConvNeXt(BaseModule):
    def __init__(self, model_name, pretrained, out_indices=[3, 5, 7, 11], norm_cfg=None):
        super().__init__()
        self.vit_layers = out_indices
        self.model_name = model_name
        self.pretrained = pretrained
        clip_model = open_clip.create_model(model_name,
                                            pretrained=pretrained,)
        self.window_size = 320
        self.embed_dim = embed_dim = 768  # output dim
        self.width = width = 1536
        self.interpolate1 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            build_norm_layer(norm_cfg, 192)[1] if norm_cfg else nn.Identity(),
            nn.GELU(),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
        )
        self.interpolate2 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
        )
        self.interpolate3 = nn.Identity()
        self.interpolate4 = nn.Identity()
        self.visual = clip_model.visual
        # self.interpolate3 = nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1)
        # self.interpolate4 = nn.Conv2d(width, width, kernel_size=3, stride=2, padding=1)

    def init_weights(self):
        clip_model = open_clip.create_model(self.model_name,
                                            pretrained=self.pretrained,
                                            device="cpu")
        print_log(self.visual.load_state_dict(clip_model.visual.state_dict(), strict=True))
        for param in self.visual.parameters():  # only freeze the CLIP model
            param.requires_grad = False

    def train(self, mode=True):
        print(f"Set train mode for CLIP ConvNeXt: {mode}", flush=True)
        self.training = mode
        self.visual.train(False)
        self.interpolate1.train(mode)
        self.interpolate2.train(mode)
        self.interpolate3.train(mode)
        self.interpolate4.train(mode)

        return self


    def forward_global(self, x):
        outs = []
        with torch.no_grad():

            x = self.visual.trunk.stem(x)
            for i in range(4):
                x = self.visual.trunk.stages[i](x)
                outs.append(x)
            

            if not self.training:
                x = self.visual.trunk.norm_pre(x)
                batch, channel, height, width = x.shape
                x = x.permute(0, 2, 3, 1) # B H W C
                x = x.reshape(batch*height*width, channel, 1, 1) # fake 2D input
                x = self.visual.trunk.head(x)
                x = self.visual.head(x)
                x = x.view(batch, height, width, -1)
                x = F.normalize(x, dim=-1)  # normalize along last dimension
                feature_map = x.permute(0, 3, 1, 2)

            else:
                feature_map = None
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