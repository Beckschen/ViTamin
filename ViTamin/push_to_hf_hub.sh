python3 -m open_clip.push_to_hf_hub \
--model ViTamin-L \
--pretrained /data/jieneng/huggingface/my_vitamin/pytorch_model.pth \
--repo-id bbexx/text \
--image-mean  0.48145466 0.4578275 0.40821073 \
--image-std 0.26862954 0.26130258 0.27577711
