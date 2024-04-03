import torch
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor
import open_clip

device = "cuda" if torch.cuda.is_available() else "cpu"

def interface_openclip():
    model, _, preprocess = open_clip.create_model_and_transforms('ViTamin-L', pretrained='/data/jieneng/huggingface/my_vitamin/pytorch_model.bin')
    model = model.to(device).eval()

    image = Image.open('/data/jieneng/huggingface/vitaminicon.png').convert('RGB')
    image_processor = CLIPImageProcessor.from_pretrained('/data/jieneng/huggingface/my_vitamin')

    pixel_values = image_processor(images=image, return_tensors='pt').pixel_values
    pixel_values = pixel_values.to(device)
    tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K')
    text = tokenizer(["a photo of V", "a dog", "a cat"]).to(device)

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features, text_features, logit_scale = model(pixel_values, text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True).to(torch.float)
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    print("Label probs:", text_probs) 

def interface_huggingface():
    image = Image.open('./vitaminicon.png').convert('RGB')
    image_processor = CLIPImageProcessor.from_pretrained('/data/jieneng/huggingface/upload_todo/ViTamin-XL-384px')
    pixel_values = image_processor(images=image, return_tensors='pt').pixel_values
    pixel_values = pixel_values.to(device)
    tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K')
    text = tokenizer(["a photo of V", "a dog", "a cat"]).to(device)

    model = AutoModel.from_pretrained(
        'jienengchen/ViTamin-XL-384px',
        trust_remote_code=True,
        ).to(device).eval()
        
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features, text_features, logit_scale = model(pixel_values, text)
        text_probs = (100.0 * image_features @ text_features.to(torch.float).T).softmax(dim=-1)

    print("Label probs:", text_probs) 