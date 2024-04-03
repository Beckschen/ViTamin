export PYTHONPATH=".:$PYTHONPATH"

torchrun --nnodes=1 --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=9999 --node_rank=0 \
        -m training.main \
        --config='./configs/vitamin_s_s1.28B_bs8k.yaml'