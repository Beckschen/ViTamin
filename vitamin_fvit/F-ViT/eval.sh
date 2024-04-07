# eval on 2080
SNAPSHOT=/data/jieneng/vitamin_weights/vitamin_ckpt/fvit_us/vitamin_l_16_336_res896_stride168/epoch_2.pth
WORKDIR=./

bash dist_test.sh configs/ov_coco/fvit_vitamin_l_336_upsample_fpn_bs64_3e_ovcoco.py \
     $SNAPSHOT  \
     8  \
     --work-dir $WORKDIR \
     --eval bbox