_base_ = 'fvit_vitb16_upsample_fpn_bs64_4x_ovlvis_eva_original.py'
model = dict(
    backbone=dict(
        pretrained='checkpoints/eva_vitb16_lvis_clipself_patches.pt'),
)
