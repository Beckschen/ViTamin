python3 tools/dump_coco_openclip_feature.py \
--out_path datasets/embeddings/coco_with_background_movit_l_336_16.pt \
--model_name vit_l16_mbconv_glu_d31_ft336 \
--pretrained /data/jieneng/vitamin_weights/vitamin_ckpt/vitamin_pieces/vit_l16_mbconv_glu_d31_224_d1.4Bs12.8B_bs90k_lr2e3_184_ft336s512Mlr1e5wd0_440gpu/checkpoints/final_ckpt.pt