# F-ViT: Build Open-Vocabulary Object Detectors Upon Frozen CLIP ViTs
## Requirements
The detection framework is built upon [MMDetection2.x](https://github.com/open-mmlab/mmdetection/tree/v2.28.1). To install MMDetection2.x, run

```bash
cd ~/your/project/directory
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
git checkout v1.7.0
MMCV_WITH_OPS=1 pip install -e . -v
cd ..
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git checkout v2.28.1
pip install -e . -v
```
For other installation methods, please refer to the official website of 
[MMCV](https://github.com/open-mmlab/mmcv.git) and [MMDetection](https://github.com/open-mmlab/mmdetection.git).

## Data Preparation
The main experiments are conducted on [COCO](https://cocodataset.org/#home) 
and [LVIS](https://www.lvisdataset.org/) datasets. We also perform transfer evaluation on 
[Objects365v1](https://www.objects365.org/overview.html). 
Please prepare datasets and organize them like the 
following:


```text
CLIPSelf/F-ViT
├── data         # use soft link to save storage on the disk
    ├── coco
        ├── annotations
            ├── instances_val2017.json       # for transfer evaluation
        ├── train2017
        ├── val2017
        ├── zero-shot         # obtain the files from the drive 
            ├── instances_val2017_all_2.json
            ├── instances_train2017_seen_2_65_cat.json
    ├── lvis_v1
        ├── annotations
            ├── lvis_v1_train_seen_1203_cat.json  # obtain the files from the drive 
            ├── lvis_v1_val.json 
        ├── train2017    # the same with coco
        ├── val2017      # the same with coco
    ├── Objects365v1
        ├── objects365_reorder_val.json         # obtain the files from the drive 
        ├── val
    
```
For open-vocabulary detection, we provide some preprocessed json files in 
[Drive](https://drive.google.com/drive/folders/19Ez8zL1cycP1FeQPpSsqCVsgsRPREQRg?usp=sharing).
Put `instances_val2017_all_2.json` and `instances_train2017_seen_2_65_cat.json` under `data/coco/zero-shot/`, 
`lvis_v1_train_seen_1203_cat.json` under `data/lvis_v1/annotations/`, and `objects365_reorder_val.json` under 
`data/Objects365v1/`.


## CLIPSelf Checkpoints
Obtain the checkpoints from 
[Drive](https://drive.google.com/drive/folders/1APWIE7M5zcymbjh5OONqXdBOxFy3Ghwm?usp=sharing). 
And they can be organized as follows:

```text
CLIPSelf/FViT/  
├── checkpoints  # use soft link to save storage on the disk
    ├── eva_vitb16_coco_clipself_patches.pt     # 1
    ├── eva_vitb16_coco_clipself_proposals.pt   # 2
    ├── eva_vitb16_coco_regionclip.pt           # 3
    ├── eva_vitl14_coco_clipself_patches.pt     # 4
    ├── eva_vitl14_coco_clipself_proposals.pt   # 5
    ├── eva_vitl14_coco_regionclip.pt           # 6
    ├── eva_vitb16_lvis_clipself_patches.pt     # 7
    ├── eva_vitl14_lvis_clipself_patches.pt     # 8
```

## Detectors 

The detectors on OV-COCO are summarized as follows:

|  #  | Backbone | CLIP Refinement | Proposals | AP50novel |                                           Config                                           | Checkpoint |
|:---:|:--------:|:---------------:|:---------:|:----:|:------------------------------------------------------------------------------------------:|:----------:|
|  1  | ViT-B/16 |    CLIPSelf     |     -     | 33.6 |   [cfg](configs/ov_coco/fvit_vitb16_upsample_fpn_bs64_3e_ovcoco_eva_clipself_patches.py)   | [model](https://drive.google.com/file/d/1uoWWYN8HlNghBY8B9GH50z1W1OysU5Kw/view?usp=sharing)  |
|  2  | ViT-B/16 |    CLIPSelf     |     +     | 37.6 |  [cfg](configs/ov_coco/fvit_vitb16_upsample_fpn_bs64_3e_ovcoco_eva_clipself_proposals.py)  | [model](https://drive.google.com/file/d/1NyolDlN5MZSlEdXB3QOgI23NHf68IjdE/view?usp=sharing)  |
|  3  | ViT-B/16 |   RegionCLIP    |     +     | 34.4 |      [cfg](configs/ov_coco/fvit_vitb16_upsample_fpn_bs64_3e_ovcoco_eva_regionclip.py)      | [model](https://drive.google.com/file/d/1KB2ko6oz1WmY_XSDJ-iJTNdOmj4Comdk/view?usp=sharing)  |
|  4  | ViT-L/14 |    CLIPSelf     |     -     | 38.4 |   [cfg](configs/ov_coco/fvit_vitl14_upsample_fpn_bs64_3e_ovcoco_eva_clipself_patches.py)   | [model](https://drive.google.com/file/d/1wn2dDlhq-3LI1MNBVzVnv7xd7NGo2nBX/view?usp=sharing)  |
|  5  | ViT-L/14 |    CLIPSelf     |     +     | 44.3 |  [cfg](configs/ov_coco/fvit_vitl14_upsample_fpn_bs64_3e_ovcoco_eva_clipself_proposals.py)  | [model](https://drive.google.com/file/d/17U46gEt57eIc3wZ7SlpG_MLbpE0-MM0Z/view?usp=sharing)  |
|  6  | ViT-L/14 |   RegionCLIP    |     +     | 38.7 |      [cfg](configs/ov_coco/fvit_vitl14_upsample_fpn_bs64_3e_ovcoco_eva_regionclip.py)      | [model](https://drive.google.com/file/d/1Fsg82-McQiHfIh8cxrG3C7eQnlAYHzX7/view?usp=sharing)  |


The detectors on OV-LVIS are summarized as follows:


|  #  | Backbone | CLIP Refinement | Proposals | mAPr |                                         Config                                         | Checkpoint |
|:---:|:--------:|:---------------:|:---------:|:----:|:--------------------------------------------------------------------------------------:|:----------:|
|  7  | ViT-B/16 |    CLIPSelf     |     -     | 25.3 | [cfg](configs/ov_lvis/fvit_vitb16_upsample_fpn_bs64_4x_ovlvis_eva_clipself_patches.py) | [model](https://drive.google.com/file/d/1e_skYDzBttUfMzpfaUJA8bIcnUFVROrE/view?usp=sharing)  |
|  8  | ViT-L/14 |    CLIPSelf     |     -     | 34.9 | [cfg](configs/ov_lvis/fvit_vitl14_upsample_fpn_bs64_4x_ovlvis_eva_clipself_patches.py) | [model](https://drive.google.com/file/d/1j-5P-RsJkOZtRoBLcz_1hJwGcY2M_GLl/view?usp=sharing)  |


### Test 
We provide the checkpoints of the object detectors in 
[Drive](https://drive.google.com/drive/folders/1MaBjbZZCfFd2HG3eCX98myYgWoWlPxrf?usp=sharing). 
And they can be organized as follows:

```text
CLIPSelf/FViT/  
├── checkpoints  # use soft link to save storage on the disk
    ├── fvit_eva_vitb16_ovcoco_clipself_patches.pth     # 1
    ├── fvit_eva_vitb16_ovcoco_clipself_proposals.pth   # 2
    ├── fvit_eva_vitb16_ovcoco_regionclip.pth           # 3
    ├── fvit_eva_vitb16_ovlvis_clipself_patches.pth     # 4
    ├── fvit_eva_vitl14_ovcoco_clipself_patches.pth     # 5
    ├── fvit_eva_vitl14_ovcoco_clipself_proposals.pth   # 6
    ├── fvit_eva_vitl14_ovcoco_regionclip.pth           # 7
    ├── fvit_eva_vitl14_ovlvis_clipself_patches.pth     # 8
```

An example of evaluation on OV-COCO
```bash
bash dist_test.sh configs/ov_coco/fvit_vitb16_upsample_fpn_bs64_3e_ovcoco_eva_clipself_proposals.py \
     checkpoints/fvit_eva_vitb16_ovcoco_clipself_proposals.pth  8  \
     --work-dir your/working/directory --eval bbox
```

An example of evaluation on OV-LVIS
```bash
bash dist_test.sh configs/ov_lvis/fvit_vitl14_upsample_fpn_bs64_4x_ovlvis_eva_clipself_patches.py \
     checkpoints/fvit_eva_vitl14_ovlvis_clipself_patches.pth   8  \
     --work-dir your/working/directory --eval segm
```


### Transfer
Transfer evaluation on COCO:
```bash
bash dist_test.sh configs/transfer/fvit_vitl14_upsample_fpn_transfer2coco.py \
     checkpoints/fvit_eva_vitl14_ovlvis_clipself_patches.pth  8  \
     --work-dir your/working/directory --eval bbox
```

Transfer evaluation on Objects365v1:
```bash
bash dist_test.sh configs/transfer/fvit_vitl14_upsample_fpn_transfer2objects365v1.py \
     checkpoints/fvit_eva_vitl14_ovlvis_clipself_patches.pth   8  \
     --work-dir your/working/directory --eval bbox
```


### Train
Prepare the CLIPSelf/RegionCLIP checkpoints as shown in the [previous section](#clipself-checkpoints).
An example of training on OV-COCO:

```bash
bash dist_train.sh  configs/ov_coco/fvit_vitb16_upsample_fpn_bs64_3e_ovcoco_eva_clipself_proposals.py \
                   8 --work-dir your/working/directory
```

An example of training on OV-LVIS:
```bash
bash dist_train.sh configs/ov_lvis/fvit_vitl14_upsample_fpn_bs64_4x_ovlvis_eva_clipself_patches.py \
                  8 --work-dir your/working/directory
```

To use multiple machines (e.g., 2x8=16 GPUs) to expedite the training on OV-LVIS, refer to the tutorial of 
[MMDetection](https://mmdetection.readthedocs.io/en/latest/user_guides/train.html). We have set 
`auto_scale_lr = dict(enable=True, base_batch_size=64)` in the config files, so the learning rate will be
modified automatically.
