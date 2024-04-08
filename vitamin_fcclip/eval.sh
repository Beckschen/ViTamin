
export DETECTRON2_DATASETS=/data/jieneng/data/d2_datasets
CONFIG=configs/coco/panoptic-segmentation/fcclip/fcclip_vitamin_l_eval_ade20k.yaml
SNAPSHOT=/data/jieneng/vitamin_weights/vitamin_ckpt/fcclip_eu/ov_coco/model_state_dict_0294999.pth

python train_net.py \
  --config-fil $CONFIG  \
  --eval-only MODEL.WEIGHTS $SNAPSHOT
