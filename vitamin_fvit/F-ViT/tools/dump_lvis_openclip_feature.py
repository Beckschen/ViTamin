import argparse
import json
import torch


from open_clip import create_model, get_tokenizer
from clip_utils import build_text_embedding_openclip


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann', default='data/lvis_v1/annotations/lvis_v1_val.json')
    parser.add_argument('--out_path', default='datasets/embeddings/lvis_v1_with_background_clip_vitb_16.pt')
    parser.add_argument('--model_name', default="EVA02-CLIP-B-16")
    parser.add_argument('--pretrained', default="eva")
    parser.add_argument('--cache_dir', default="")
    args = parser.parse_args()

    print('Loading', args.ann)
    data = json.load(open(args.ann, 'r'))
    cat_names = [x['name'] for x in \
        sorted(data['categories'], key=lambda x: x['id'])]
    cat_names = cat_names + ['background']
    ori_cat_names = cat_names
    print('cat_names', cat_names)
    model = create_model(model_name=args.model_name, pretrained=args.pretrained)
    tokenizer = get_tokenizer(model_name=args.model_name)
    text_embeddings = build_text_embedding_openclip(cat_names, model, tokenizer)
    text_embeddings = text_embeddings.cpu()
    text_embeddings = text_embeddings.to(torch.float32)
    print('text_embeddings.shape', text_embeddings.shape)
    class_embed = {k:v for k, v in zip(ori_cat_names, text_embeddings)}
    torch.save(class_embed, args.out_path)
