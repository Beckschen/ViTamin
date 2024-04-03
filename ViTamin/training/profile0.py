import argparse
import sys
print(sys.path)
import torch
import open_clip
import pandas as pd
from fvcore.nn import FlopCountAnalysis, flop_count_str, ActivationCountAnalysis


parser = argparse.ArgumentParser(description='OpenCLIP Profiler')

# benchmark specific args
parser.add_argument('--model', metavar='NAME', default='',
                    help='model(s) to profile')
parser.add_argument('--results-file', default='', type=str, metavar='FILENAME',
                    help='Output csv file for results')
parser.add_argument('--size', default=224, type=int, help='img_size')
parser.add_argument('--text-size', default=77, type=int, help='img_size')
# python3 training/profile0.py --model 'vit_h16_mbconv_glu'

"""
                        model  image_size  image_width  text_width  embed_dim   gmacs   macts  mparams  image_gmacs  image_macts  image_mparams  text_gmacs  text_macts  text_mparams
1  vit_l16_mbconv_glu_d31_224         224          768         768        768   79.15   93.95   456.97        72.60        87.56         333.32        6.55        6.39        123.65
2                    ViT-L-14         224         1024         768        768   84.38   63.49   427.62        77.83        57.11         303.97        6.55        6.39        123.65
0          vit_h16_mbconv_glu         224          768        1024       1024  159.07  146.09   991.33       135.80       129.05         637.29       23.27       17.03        354.03
3                    ViT-H-14         224         1280        1024       1024  185.26  112.10   986.11       161.99        95.07         632.08       23.27       17.03        354.03

0  vit_b16_mbconv_glu_d12             224          768         512        512  22.44  45.75   139.13        19.53         41.5          75.71        2.91        4.26         63.43
0  vit_b16_mbconv_d12                 224          768         512        512  25.21  45.75   153.27        22.29         41.5          89.84        2.91        4.26         63.43
0  vit_b16_mbconv_d14                 224          768         512        512  27.98  48.46   167.45        25.07        44.21         104.02        2.91        4.26         63.43

0  vit_b16_mbconv_8c_glu              224          768         512        512  22.38  42.19   149.44        19.47        37.93          86.01        2.91        4.26         63.43
1  vit_b16_mbconv_4c_glu              224          768         512        512  31.44  61.01   154.88        28.53        56.75          91.45        2.91        4.26         63.43

    vit_450M_mbconv_glu_cl77_224
"""
def profile_fvcore(
        model,
        image_input_size=(3, 224, 224),
        # image_input_size=(3, 336, 336),
        text_input_size=(77,),
        batch_size=1,
        detailed=False,
        force_cpu=False
):
    if force_cpu:
        model = model.to('cpu')
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    example_image_input = torch.ones((batch_size,) + image_input_size, device=device, dtype=dtype)
    example_text_input = torch.ones((batch_size,) + text_input_size, device=device, dtype=torch.int64)
    fca = FlopCountAnalysis(model, (example_image_input, example_text_input))
    aca = ActivationCountAnalysis(model, (example_image_input, example_text_input))
    if detailed:
        fcs = flop_count_str(fca)
        print(fcs)
    return fca.total(), aca.total()


def profile_fvcore_text(
        model,
        text_input_size=(77,),
        batch_size=1,
        detailed=False,
        force_cpu=False
):
    if force_cpu:
        model = model.to('cpu')
    device = next(model.parameters()).device
    example_input = torch.ones((batch_size,) + text_input_size, device=device, dtype=torch.int64)
    fca = FlopCountAnalysis(model, example_input)
    aca = ActivationCountAnalysis(model, example_input)
    if detailed:
        fcs = flop_count_str(fca)
        print(fcs)
    return fca.total(), aca.total()


def profile_fvcore_image(
        model,
        image_input_size=(3, 224, 224),
        batch_size=1,
        detailed=False,
        force_cpu=False
):
    if force_cpu:
        model = model.to('cpu')
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    example_input = torch.ones((batch_size,) + image_input_size, device=device, dtype=dtype)
    fca = FlopCountAnalysis(model, example_input)
    aca = ActivationCountAnalysis(model, example_input)
    if detailed:
        fcs = flop_count_str(fca)
        print(fcs)
    return fca.total(), aca.total()

# python3 open_clip/src/training/profile.py --model 'vit_h16_mbconv_glu'
def count_params(model):
    return sum([m.numel() for m in model.parameters()])


def profile_model(model_name, args):
    model = open_clip.create_model(model_name, force_custom_text=True, pretrained_hf=False)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()


    if isinstance(model.visual.image_size, (tuple, list)):
        image_input_size = (3,) + tuple(model.visual.image_size[-2:])
    else:
        image_input_size = (3, model.visual.image_size, model.visual.image_size)
    if args.size != 224:
        image_input_size = (3, args.size, args.size)

    text_input_size = (args.text_size,)

    results = {}
    # results['model'] = model_name
    results['image_size'] = image_input_size[1]

    model_cfg = open_clip.get_model_config(model_name)
    if model_cfg:
        vision_cfg = open_clip.CLIPVisionCfg(**model_cfg['vision_cfg'])
        text_cfg = open_clip.CLIPTextCfg(**model_cfg['text_cfg'])
        results['image_width'] = int(vision_cfg.width)
        results['text_width'] = int(text_cfg.width)
        # results['embed_dim'] = int(model_cfg['embed_dim'])
    else:
        results['image_width'] = 0
        results['text_width'] = 0
        # results['embed_dim'] = 0


    image_macs, image_acts = profile_fvcore_image(
        model.visual, image_input_size=image_input_size, force_cpu=False)

    retries = 2
    while retries:
        retries -= 1
        # try:
        if True:
            macs, acts = profile_fvcore(
                model, image_input_size=image_input_size, text_input_size=text_input_size, force_cpu=not retries)

            image_macs, image_acts = profile_fvcore_image(
                model.visual, image_input_size=image_input_size, force_cpu=not retries)

            text_macs, text_acts = profile_fvcore_text(
                model.text, text_input_size=text_input_size, force_cpu=not retries)

            # results['gmacs'] = round(macs / 1e9, 2)
            # results['macts'] = round(acts / 1e6, 2)
            # results['mparams'] = round(count_params(model) / 1e6, 2)
            results['image_gmacs'] = round(image_macs / 1e9, 2)
            # results['image_macts'] = round(image_acts / 1e6, 2)
            results['image_mparams'] = round(count_params(model.visual) / 1e6, 2)
            results['text_gmacs'] = round(text_macs / 1e9, 2)
            # results['text_macts'] = round(text_acts / 1e6, 2)
            results['text_mparams'] = round(count_params(model.text) / 1e6, 2)
        # except RuntimeError as e:
        #     pass
    return results


def main():
    args = parser.parse_args()
    # FIXME accept a text file name to allow lists of models in txt/csv
    if args.model == 'all':
        parsed_model = open_clip.list_models()
    else:
        parsed_model = args.model.split(',')

    results = []
    for m in parsed_model:
        row = profile_model(m, args)
        results.append(row)

    df = pd.DataFrame(results, columns=results[0].keys())
    # df = df.sort_values('gmacs')
    print(df)
    if args.results_file:
        df.to_csv(args.results_file, index=False)


if __name__ == '__main__':
    main()
