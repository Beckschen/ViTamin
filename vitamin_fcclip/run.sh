#!/bin/bash

export DETECTRON2_DATASETS=/data/jieneng/data/d2_datasets
python3 train_net.py --dist-url "auto" --num-gpus 8 --config-file configs/coco/panoptic-segmentation/fcclip/fcclip_vitamin_l_eval_ade20k.yaml