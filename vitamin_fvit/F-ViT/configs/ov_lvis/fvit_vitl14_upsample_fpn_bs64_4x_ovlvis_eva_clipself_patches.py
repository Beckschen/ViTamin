_base_ = 'fvit_vitl14_upsample_fpn_bs64_4x_ovlvis_eva_original.py'
model = dict(
    backbone=dict(
        pretrained='checkpoints/eva_vitl14_lvis_clipself_patches.pt'),
)
