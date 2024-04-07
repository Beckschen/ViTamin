_base_ = 'fvit_vitb16_upsample_fpn_bs64_3e_ovcoco_eva_original.py'
model = dict(
    backbone=dict(
        pretrained='checkpoints/eva_vitb16_coco_clipself_patches.pt',
    ),
)
