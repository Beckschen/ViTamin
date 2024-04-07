_base_ = "../ov_lvis/fvit_vitl14_upsample_fpn_bs64_4x_ovlvis_eva_clipself_patches.py"
num_classes = 20
model = dict(
    roi_head=dict(
        bbox_head=dict(
            type='FViTTransferBBoxHead',
            num_classes=num_classes,
            fixed_temperature=50.0,
            vlm_temperature=120.0,
            alpha=0.3,
            class_embed=
            'datasets/embeddings/voc_transfer_background_evaclip_vitl14x336.pt',
            seen_classes='datasets/voc_classes.json',
            all_classes='datasets/voc_classes.json'),
        mask_roi_extractor=None,
        mask_head=None)
)
# dataset settings
dataset_type = 'VOCDataset'
data_root = 'data/VOCdevkit/'

data = dict(
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/'),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/')
)
evaluation = dict(interval=1, metric='mAP', iou_thr=[0.5, 0.75])
