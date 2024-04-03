export CUDA_VISIBLE_DEVICES=0

MODEL_NAME='ViTamin-XL-384'
MODEL_WEIGHT='/data/jieneng/huggingface/upload_todo/ViTamin-XL-384px/pytorch_model.bin'
YOUR_EVAL_DATA_DIR='/data/jieneng/data/datacomp_eval'
OUTPUT_DIR='/data/jieneng/huggingface/upload_todo/eval/ViTamin-XL-384px'

cd datacomp && python3 evaluate.py \
    --use_model "$MODEL_NAME $MODEL_WEIGHT" \
    --data_dir $YOUR_EVAL_DATA_DIR \
    --train_output_dir  $OUTPUT_DIR
