"""
Copyright (2023) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 
"""

import torch
import torch.nn.functional as F
import math
from detectron2.utils import comm

import open_clip

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

@BACKBONE_REGISTRY.register()
class CLIP(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()
        model_name = cfg.MODEL.FC_CLIP.CLIP_MODEL_NAME
        pretrained= cfg.MODEL.FC_CLIP.CLIP_PRETRAINED_WEIGHTS
        # download on local rank 0 first
        if comm.get_local_rank() == 0:
            open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        comm.synchronize()

        self.clip_model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.text_tokenizer = open_clip.get_tokenizer(model_name)

        self.model_name = model_name = model_name.lower()
        if 'convnext_' in model_name:
            self.model_type = 'convnext'
            if '_base' in model_name:
                self.output_channels = [128, 128, 256, 512, 1024]
            elif '_large' in model_name:
                self.output_channels = [192, 192, 384, 768, 1536]
            elif '_xxlarge' in model_name:
                self.output_channels = [384, 384, 768, 1536, 3072]
        else:
            self.model_type = 'vitamin'
            self.output_channels = cfg.MODEL.FC_CLIP.CLIP_OUTPUT_CHANNELS


        if 'convnext_' in self.model_name:
            self.dim_latent = self.clip_model.text_projection.shape[-1]
        else:
            self.dim_latent = self.output_channels[4] #cfg.MODEL.FC_CLIP.EMBED_DIM

        self._out_feature_strides = {
            "stem": 2,
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
            "clip_embedding": -1
        }
        self._out_feature_channels = {
            "stem": self.output_channels[0],
            "res2": self.output_channels[1],
            "res3": self.output_channels[2],
            "res4": self.output_channels[3],
            "res5": self.output_channels[4],
            "clip_embedding": self.dim_latent
        }

        self.window_size = cfg.MODEL.FC_CLIP.SLIDING_WINDOW_SIZE # 336
        self.sliding_stride = cfg.MODEL.FC_CLIP.SLIDING_STRIDE
        self.eval()
        self.freeze_everything()


    def freeze_everything(self):
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.clip_model.transformer.get_cast_dtype()

        x = self.clip_model.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.clip_model.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.transformer(x, attn_mask=self.clip_model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.clip_model.text_projection
        return F.normalize(x, dim=-1) if normalize else x

    def encode_customtext(self, text, normalize: bool = False):
        features = self.clip_model.text(text)
        return F.normalize(features, dim=-1) if normalize else features

    def tokenize_text(self, text):
        return self.text_tokenizer(text)

    def extract_features(self, x):
        return {
            'convnext': self.extract_features_convnext, 'vitamin': self.extract_features_vitamin
        }[self.model_type](x)
    
    def visual_prediction_forward(self, x):
        return {
            'convnext': self.visual_prediction_forward_convnext, 'vitamin': self.visual_prediction_forward_vitamin
        }[self.model_type](x)

    def extract_features_convnext(self, x):
        out = {}
        x = self.clip_model.visual.trunk.stem(x)
        out['stem'] = x.contiguous() # os4
        for i in range(4):
            x = self.clip_model.visual.trunk.stages[i](x)
            out[f'res{i+2}'] = x.contiguous() # res 2 (os4), 3 (os8), 4 (os16), 5 (os32)
        
        x = self.clip_model.visual.trunk.norm_pre(x)
        out['clip_vis_dense'] = x.contiguous()
        return out

    @torch.no_grad()
    def extract_features_vitamin_perwindow(self, x):
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

        out = {}
        x = self.clip_model.visual.trunk.patch_embed.backbone.stem(x)
        out['stem'] = map1dto2d(x) # os2
        x = self.clip_model.visual.trunk.patch_embed.backbone.stages[0](x)
        out["res2"] = map1dto2d(x) # os4
        x = self.clip_model.visual.trunk.patch_embed.backbone.stages[1](x)
        out["res3"] = map1dto2d(x) # os8
        x = self.clip_model.visual.trunk.patch_embed.backbone.pool(x)
        x = self.clip_model.visual.trunk.patch_embed.proj(x)
        x = x.flatten(2).transpose(1, 2)
        if hasattr(self.clip_model.visual.trunk, 'is_pos_embed') and hasattr(self.clip_model.visual.trunk, 'pos_embed') and self.clip_model.visual.trunk.is_pos_embed:
            x = self.clip_model.visual.trunk._pos_embed(x)
        x = self.clip_model.visual.trunk.patch_drop(x)
        x = self.clip_model.visual.trunk.norm_pre(x)
        x = self.clip_model.visual.trunk.blocks(x)
        out["res4"] = map1dto2d(x) # os16
        x = self.clip_model.visual.trunk.norm(x)
        out['clip_vis_dense'] = map1dto2d(x)
        return out
    
    @torch.no_grad()
    def extract_features_vitamin(self, x):
        # we always ensure that h_img % window_size == 0
        batch_size, c_img, h_img, w_img = x.shape
        h_crop = w_crop = self.window_size
        if self.training:
            h_stride = w_stride = self.window_size # during training
        else:
            # h_stride = w_stride = self.window_size // 3 # during test
            h_stride = w_stride = self.sliding_stride
            # h_stride = w_stride = self.window_size // 2 # during test 04072024
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        out = {}
        out["stem"] = torch.zeros((batch_size, self.output_channels[0], h_img // 2, w_img // 2), dtype=x.dtype, device=x.device)
        out["res2"] = torch.zeros((batch_size, self.output_channels[1], h_img // 4, w_img // 4), dtype=x.dtype, device=x.device)
        out["res3"] = torch.zeros((batch_size, self.output_channels[2], h_img // 8, w_img // 8), dtype=x.dtype, device=x.device)
        out["res4"] = torch.zeros((batch_size, self.output_channels[3], h_img // 16, w_img // 16), dtype=x.dtype, device=x.device)
        out["clip_vis_dense"] = torch.zeros((batch_size, self.dim_latent, h_img // 16, w_img // 16), dtype=x.dtype, device=x.device)

        count_mats = {k: torch.zeros_like(v) for k, v in out.items()}

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_x = x[:, :, y1:y2, x1:x2]
                tmp_out = self.extract_features_vitamin_perwindow(crop_x)
                out["stem"][:, :, y1//2:y2//2, x1//2:x2//2] += tmp_out["stem"]
                count_mats["stem"][..., y1//2:y2//2, x1//2:x2//2] += 1
                out["res2"][:, :, y1//4:y2//4, x1//4:x2//4] += tmp_out["res2"]
                count_mats["res2"][..., y1//4:y2//4, x1//4:x2//4] += 1
                out["res3"][:, :, y1//8:y2//8, x1//8:x2//8] += tmp_out["res3"]
                count_mats["res3"][..., y1//8:y2//8, x1//8:x2//8] += 1
                out["res4"][:, :, y1//16:y2//16, x1//16:x2//16] += tmp_out["res4"]
                count_mats["res4"][..., y1//16:y2//16, x1//16:x2//16] += 1
                out["clip_vis_dense"][:, :, y1//16:y2//16, x1//16:x2//16] += tmp_out["clip_vis_dense"]
                count_mats["clip_vis_dense"][..., y1//16:y2//16, x1//16:x2//16] += 1
        assert all((count_mats[k] == 0).sum() == 0 for k in count_mats)
        for k in out:
            out[k] /= count_mats[k]

        out["res5"] = out["res4"][..., ::2, ::2]
        return out

    def visual_prediction_forward_convnext(self, x,):
        batch, num_query, channel = x.shape
        x = x.reshape(batch*num_query, channel, 1, 1) # fake 2D input
        x = self.clip_model.visual.trunk.head(x)
        x = self.clip_model.visual.head(x)
        return x.view(batch, num_query, x.shape[-1]) # B x num_queries x 640


    def visual_prediction_forward_vitamin(self, x,):
        batch, num_query, channel = x.shape
        x = self.clip_model.visual.trunk.fc_norm(x)
        x = self.clip_model.visual.trunk.head_drop(x)
        x = self.clip_model.visual.trunk.head(x)
        x = self.clip_model.visual.head(x)
        return x.view(batch, num_query, x.shape[-1]) # B x num_queries x 640


    def get_text_classifier(self, text_list, device):
        self.eval()
        with torch.no_grad():
            # reference for templates: https://github.com/mlfoundations/open_clip/blob/91f6cce16b7bee90b3b5d38ca305b5b3b67cc200/src/training/imagenet_zeroshot_data.py
            text_tokens = self.tokenize_text(text_list)
            text_tokens = text_tokens.to(device)
            # we return un-normalized text feature.
            if self.model_type == 'vitamin':
                text_features = self.encode_customtext(text_tokens, normalize=False)
            else:
                text_features = self.encode_text(text_tokens, normalize=False)
            return text_features

    def forward(self, x):
        self.eval()
        with torch.no_grad():
            return self.extract_features(x)
    
    # @property
    # def dim_latent(self):
    #     if 'convnext_' in self.model_name:
    #         return self.clip_model.text_projection.shape[-1]
    #     else:
    #         return None
    
    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in ["stem", "res2", "res3", "res4", "res5", "clip_embedding"]
        }

    @property
    def size_divisibility(self):
        return -1