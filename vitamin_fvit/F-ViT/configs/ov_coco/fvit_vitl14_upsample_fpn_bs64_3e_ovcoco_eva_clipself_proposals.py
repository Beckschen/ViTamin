_base_ = 'fvit_vitl14_upsample_fpn_bs64_3e_ovcoco_eva_original.py'
model = dict(
    backbone=dict(
        pretrained='checkpoints/eva_vitl14_coco_clipself_proposals.pt',
    ),
)
