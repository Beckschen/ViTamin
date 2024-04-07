_base_ = "../ov_lvis/fvit_vitl14_upsample_fpn_bs64_4x_ovlvis_eva_clipself_patches.py"
num_classes = 80
model = dict(
    roi_head=dict(
        bbox_head=dict(
            type='FViTTransferBBoxHead',
            num_classes=num_classes,
            fixed_temperature=50.0,
            vlm_temperature=120.0,
            alpha=0.2,
            class_embed=
            'datasets/embeddings/coco_transfer_background_evaclip_vitl14x336.pt',
            seen_classes='datasets/mscoco_all_classes.json',
            all_classes='datasets/mscoco_all_classes.json'),
        mask_roi_extractor=None,
        mask_head=None)
)
# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'

data = dict(
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/'),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/')
)
evaluation = dict(interval=1, metric='bbox')
