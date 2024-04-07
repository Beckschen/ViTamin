find_unused_parameters = True
norm_cfg = dict(type='SyncBN', requires_grad=True)
head_norm_cfg = dict(type='MMSyncBN', requires_grad=True)
num_classes = 1203
model = dict(
    type='FViT',
    backbone=dict(
        type='CLIPViTamin',
        norm_cfg=norm_cfg,
        out_indices=[6, 10, 14, 23],
        model_name='ViTamin-L-336',
        pretrained='./vitamin_l_336_16.pt'), # change to the downloaded pretrained weight path
    neck=dict(
        type='FPN',
        in_channels=[160, 320, 1024, 1024],
        out_channels=256,
        num_outs=5,
        norm_cfg=norm_cfg),
    rpn_head=dict(
        type='CustomRPNHead',
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
        num_convs=2,
        norm_cfg=head_norm_cfg),
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
            fc_out_channels=768,
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
                bg_weight=0.9,
                class_weight='datasets/lvis_v1_train_cat_norare_info.json'),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0),
            norm_cfg=norm_cfg,
            learned_temperature=50.0,
            vlm_temperature=50.0,
            alpha=0.1,
            beta=0.4,       # a larger beta can significantly increase APr, e.g., setting beta=0.6 or 0.8
            class_embed= 
            'datasets/embeddings/lvis_with_background_vitamin_l_336_16.pt', # change to your own path
            seen_classes='datasets/lvis_v1_seen_classes.json',
            all_classes='datasets/lvis_v1_all_classes.json',
            num_shared_convs=4,
            num_shared_fcs=2,
            num_cls_fcs=1,
            num_reg_fcs=1),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=num_classes,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
            class_agnostic=True,
            norm_cfg=norm_cfg),
        vlm_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlign',
                output_size=1,
                sampling_ratio=0,
                use_torchvision=True),
            out_channels=768,
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
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.0001,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=300,
            mask_thr_binary=0.5)))
image_size = (896, 896)
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
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
        mean=[122.7709383, 116.7460125, 104.09373615],
        std=[68.5005327, 66.6321579, 70.32316305],
        to_rgb=True),
    dict(type='Pad', size=image_size),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
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
                mean=[122.7709383, 116.7460125, 104.09373615],
                std=[68.5005327, 66.6321579, 70.32316305],
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
        type='LVISV1DatasetOV',
        ann_file='data/lvis_v1/annotations/lvis_v1_train_seen_1203_cat.json',
        img_prefix='data/lvis_v1/',
        pipeline=train_pipeline),
    val=dict(
        type='LVISV1DatasetOV',
        ann_file='data/lvis_v1/annotations/lvis_v1_val.json',
        img_prefix='data/lvis_v1/',
        pipeline=test_pipeline),
    test=dict(
        type='LVISV1DatasetOV',
        ann_file='data/lvis_v1/annotations/lvis_v1_val.json',
        img_prefix='data/lvis_v1/',
        pipeline=test_pipeline))
evaluation = dict(interval=12, metric=['segm'])
optimizer = dict(type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.1)
optimizer_config = dict(grad_clip=dict(max_norm=1.0, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=250,
    warmup_ratio=0.001,
    step=[32, 44])
runner = dict(type='EpochBasedRunner', max_epochs=48)
checkpoint_config = dict(interval=1, max_keep_ckpts=1, save_last=True)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=True, base_batch_size=64)
fp16 = dict(loss_scale=512.0)
auto_resume = False
