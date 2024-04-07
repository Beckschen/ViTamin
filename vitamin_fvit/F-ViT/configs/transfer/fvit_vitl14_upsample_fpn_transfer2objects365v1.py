_base_ = "../ov_lvis/fvit_vitl14_upsample_fpn_bs64_4x_ovlvis_eva_clipself_patches.py"
num_classes = 365
model = dict(
    roi_head=dict(
        bbox_head=dict(
            type='FViTTransferBBoxHead',
            num_classes=num_classes,
            fixed_temperature=50.0,
            vlm_temperature=120.0,
            alpha=0.3,
            class_embed=
            'datasets/embeddings/objects365v1_transfer_background_evaclip_vitl14x336.pt',
            seen_classes='datasets/objects365v1_fix_classes.json',
            all_classes='datasets/objects365v1_fix_classes.json'),
        mask_roi_extractor=None,
        mask_head=None)
)
# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/Objects365v1/'
classes = ('person', 'sneakers', 'chair', 'hat', 'lamp', 'bottle', 'cabinet/shelf', 'cup', 'car', 'glasses',
           'picture/frame', 'desk', 'handbag', 'street lights', 'book', 'plate', 'helmet', 'leather shoes', 'pillow',
           'glove', 'potted plant', 'bracelet', 'flower', 'tv', 'storage box', 'vase', 'bench', 'wine glass', 'boots',
           'bowl', 'dining table', 'umbrella', 'boat', 'flag', 'speaker', 'trash bin/can', 'stool', 'backpack',
           'couch', 'belt', 'carpet', 'basket', 'towel/napkin', 'slippers', 'barrel/bucket', 'coffee table', 'suv',
           'toy', 'tie', 'bed', 'traffic light', 'pen/pencil', 'microphone', 'sandals', 'canned', 'necklace', 'mirror',
           'faucet', 'bicycle', 'bread', 'high heels', 'ring', 'van', 'watch', 'sink', 'horse', 'fish', 'apple',
           'camera', 'candle', 'teddy bear', 'cake', 'motorcycle', 'wild bird', 'laptop', 'knife', 'traffic sign',
           'cell phone', 'paddle', 'truck', 'cow', 'power outlet', 'clock', 'drum', 'fork', 'bus', 'hanger',
           'nightstand', 'pot/pan', 'sheep', 'guitar', 'traffic cone', 'tea pot', 'keyboard', 'tripod', 'hockey',
           'fan', 'dog', 'spoon', 'blackboard/whiteboard', 'balloon', 'air conditioner', 'cymbal', 'mouse',
           'telephone', 'pickup truck', 'orange', 'banana', 'airplane', 'luggage', 'skis', 'soccer', 'trolley',
           'oven', 'remote', 'baseball glove', 'paper towel', 'refrigerator', 'train', 'tomato', 'machinery vehicle',
           'tent', 'shampoo/shower gel', 'head phone', 'lantern', 'donut', 'cleaning products', 'sailboat',
           'tangerine', 'pizza', 'kite', 'computer box', 'elephant', 'toiletries', 'gas stove', 'broccoli', 'toilet',
           'stroller', 'shovel', 'baseball bat', 'microwave', 'skateboard', 'surfboard', 'surveillance camera',
           'gun', 'life saver', 'cat', 'lemon', 'liquid soap', 'zebra', 'duck', 'sports car', 'giraffe', 'pumpkin',
           'piano', 'stop sign', 'radiator', 'converter', 'tissue ', 'carrot', 'washing machine', 'vent', 'cookies',
           'cutting/chopping board', 'tennis racket', 'candy', 'skating and skiing shoes', 'scissors', 'folder',
           'baseball', 'strawberry', 'bow tie', 'pigeon', 'pepper', 'coffee machine', 'bathtub', 'snowboard',
           'suitcase', 'grapes', 'ladder', 'pear', 'american football', 'basketball', 'potato', 'paint brush',
           'printer', 'billiards', 'fire hydrant', 'goose', 'projector', 'sausage', 'fire extinguisher',
           'extension cord', 'facial mask', 'tennis ball', 'chopsticks', 'electronic stove and gas stove', 'pie',
           'frisbee', 'kettle', 'hamburger', 'golf club', 'cucumber', 'clutch', 'blender', 'tong', 'slide', 'hot dog',
           'toothbrush', 'facial cleanser', 'mango', 'deer', 'egg', 'violin', 'marker', 'ship', 'chicken', 'onion',
           'ice cream', 'tape', 'wheelchair', 'plum', 'bar soap', 'scale', 'watermelon', 'cabbage', 'router/modem',
           'golf ball', 'pine apple', 'crane', 'fire truck', 'peach', 'cello', 'notepaper', 'tricycle', 'toaster',
           'helicopter', 'green beans', 'brush', 'carriage', 'cigar', 'earphone', 'penguin', 'hurdle', 'swing',
           'radio', 'CD', 'parking meter', 'swan', 'garlic', 'french fries', 'horn', 'avocado', 'saxophone',
           'trumpet', 'sandwich', 'cue', 'kiwi fruit', 'bear', 'fishing rod', 'cherry', 'tablet', 'green vegetables',
           'nuts', 'corn', 'key', 'screwdriver', 'globe', 'broom', 'pliers', 'volleyball', 'hammer', 'eggplant',
           'trophy', 'dates', 'board eraser', 'rice', 'tape measure/ruler', 'dumbbell', 'hamimelon', 'stapler',
           'camel', 'lettuce', 'goldfish', 'meat balls', 'medal', 'toothpaste', 'antelope', 'shrimp', 'rickshaw',
           'trombone', 'pomegranate', 'coconut', 'jellyfish', 'mushroom', 'calculator', 'treadmill', 'butterfly',
           'egg tart', 'cheese', 'pig', 'pomelo', 'race car', 'rice cooker', 'tuba', 'crosswalk sign', 'papaya',
           'hair drier', 'green onion', 'chips', 'dolphin', 'sushi', 'urinal', 'donkey', 'electric drill',
           'spring rolls', 'tortoise/turtle', 'parrot', 'flute', 'measuring cup', 'shark', 'steak', 'poker card',
           'binoculars', 'llama', 'radish', 'noodles', 'yak', 'mop', 'crab', 'microscope', 'barbell', 'bread/bun',
           'baozi', 'lion', 'red cabbage', 'polar bear', 'lighter', 'seal', 'mangosteen', 'comb', 'eraser', 'pitaya',
           'scallop', 'pencil case', 'saw', 'table tennis paddle', 'okra', 'starfish', 'eagle', 'monkey', 'durian',
           'game board', 'rabbit', 'french horn', 'ambulance', 'asparagus', 'hoverboard', 'pasta', 'target',
           'hotair balloon', 'chainsaw', 'lobster', 'iron', 'flashlight')


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
image_size = (896, 896)
# file_client_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         'data/Objects365v1/val': 'openmmlab:s3://openmmlab/datasets/detection/Objects365/val',
#     }))
file_client_args = dict(backend='disk')
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
    val=dict(
        pipeline=test_pipeline,
        classes=classes,
        type=dataset_type,
        ann_file=data_root + 'objects365_reorder_val.json',
        img_prefix=data_root + "val/"),
    test=dict(
        pipeline=test_pipeline,
        classes=classes,
        type=dataset_type,
        ann_file=data_root + 'objects365_reorder_val.json',
        img_prefix=data_root + "val/")
)
evaluation = dict(interval=1, metric='bbox')
