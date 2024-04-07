find_unused_parameters = True
num_classes = 65
class_weight = [
    1.0, 1.0, 1.0, 1.0, 0, 0, 1.0, 1.0, 1.0, 1.0, 1.0, 0, 0, 1.0, 1.0, 0, 0,
    1.0, 1.0, 1.0, 1.0, 0, 1.0, 0, 1.0, 1.0, 1.0, 0, 1.0, 0, 1.0, 1.0, 0, 1.0,
    0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0, 1.0, 0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 0, 1.0, 1.0, 1.0, 0, 1.0, 1.0, 1.0, 1.0, 0, 1.0, 0.6
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='FViT',
    backbone=dict(
        type='EvaCLIPViT',
        model_name='EVA02-CLIP-B-16',
        pretrained='checkpoints/EVA02_CLIP_B_psz16_s8B.pt',
        norm_cfg=norm_cfg,
        out_indices=[3, 5, 7, 11]),
    neck=dict(
        type='FPN',
        in_channels=[768, 768, 768, 768],
        out_channels=256,
        num_outs=5,
        norm_cfg=norm_cfg),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0),
        num_convs=2),
    roi_head=dict(
        type='FViTRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='FViTBBoxHead',
            in_channels=256,
            fc_out_channels=512,
            roi_feat_size=7,
            num_classes=num_classes,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CustomCrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0,
                class_weight=class_weight),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0),
            norm_cfg=norm_cfg,
            fixed_temperature=0,
            learned_temperature=50.0,
            vlm_temperature=75.0,
            alpha=0.1,
            beta=0.8,
            class_embed=
            'datasets/embeddings/coco_with_background_evaclip_vitb_16.pt',
            seen_classes='datasets/mscoco_seen_classes.json',
            all_classes='datasets/mscoco_65_classes.json',
            num_shared_convs=4,
            num_shared_fcs=2,
            num_cls_fcs=1,
            num_reg_fcs=1),
        vlm_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlign',
                output_size=1,
                sampling_ratio=0,
                use_torchvision=True),
            out_channels=512,
            featmap_strides=[16])),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.01,
            nms=dict(type='nms', iou_threshold=0.4),
            max_per_img=100)))
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=True, base_batch_size=64)
dataset_type = 'CocoDatasetOV'
image_size = (640, 640)
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(
        type='Resize',
        img_scale=image_size,
        ratio_range=(0.1, 2.0),
        multiscale_mode='range',
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=image_size,
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(0.01, 0.01)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=image_size),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug',
        img_scale=image_size,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', pad_to_square=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        ann_file=
        'data/coco/zero-shot/instances_train2017_seen_2_65_cat.json',
        img_prefix='data/coco/train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file='data/coco/zero-shot/instances_val2017_all_2.json',
        img_prefix='data/coco/val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='data/coco/zero-shot/instances_val2017_all_2.json',
        img_prefix='data/coco/val2017/',
        pipeline=test_pipeline)
)
evaluation = dict(interval=1, metric=['bbox'])
optimizer = dict(type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.1)
optimizer_config = dict(grad_clip=dict(max_norm=1.0, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=250,
    warmup_ratio=0.001,
    step=[100])
runner = dict(type='EpochBasedRunner', max_epochs=3)
fp16 = dict(loss_scale=512.0)
auto_resume = False
