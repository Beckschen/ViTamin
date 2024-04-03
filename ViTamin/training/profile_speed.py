import argparse

import logging
import time
import torch
import open_clip
import pandas as pd
from contextlib import suppress
from functools import partial
from fvcore.nn import FlopCountAnalysis, flop_count_str, ActivationCountAnalysis
from timm.utils import setup_default_logging
_logger = logging.getLogger('validate')

parser = argparse.ArgumentParser(description='OpenCLIP Profiler')

# benchmark specific args
parser.add_argument('--model', metavar='NAME', default='',
                    help='model(s) to profile')
parser.add_argument('--results-file', default='', type=str, metavar='FILENAME',
                    help='Output csv file for results')
parser.add_argument('--size', default=224, type=int, help='img_size')
parser.add_argument('--batch-size', default=256, type=int, help='batch size')
parser.add_argument('--amp-dtype', default='amp_bfloat16', type=str, help='amp-dtype')
parser.add_argument('--channels-first', action='store_true', default=False,help='')
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

"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def resolve_precision(precision: str):
    assert precision in ('amp', 'amp_bfloat16', 'float16', 'bfloat16', 'float32')
    amp_dtype = None  # amp disabled
    model_dtype = torch.float32
    data_dtype = torch.float32
    if precision == 'amp':
        amp_dtype = torch.float16
    elif precision == 'amp_bfloat16':
        amp_dtype = torch.bfloat16
    elif precision == 'float16':
        model_dtype = torch.float16
        data_dtype = torch.float16
    elif precision == 'bfloat16':
        model_dtype = torch.bfloat16
        data_dtype = torch.bfloat16
    return amp_dtype, model_dtype, data_dtype

def timestamp(sync=False):
    return time.perf_counter()

def cuda_timestamp(sync=False, device=None):
    if sync:
        torch.cuda.synchronize(device=device)
    return time.perf_counter()

class BenchmarkRunner:
    def __init__(
        self,
        model_name,
        image_input_size=(3, 224, 224),
        text_input_size=(77,),
        batch_size=256,
        device='cuda',
        precision='amp_bfloat16',
        channels_last=True,
        num_warm_iter=10,
        num_bench_iter=50,
    ):
        self.model_name = model_name
        self.device = device
        self.channels_last = channels_last
        self.amp_dtype, self.model_dtype, self.data_dtype = resolve_precision(precision)
        if self.amp_dtype is not None:
            self.amp_autocast = partial(torch.cuda.amp.autocast, dtype=self.amp_dtype)
        else:
            self.amp_autocast = suppress

        self.model  = open_clip.create_model(model_name, force_custom_text=True, pretrained_hf=False)
        self.model.to(
            device=self.device,
            dtype=self.model_dtype,
            memory_format=torch.channels_last if self.channels_last else None
        )
        # TODO: torchscript; torchcompile
        self.batch_size = batch_size
        self.image_input_size = image_input_size
        self.text_input_size = text_input_size
        self.example_image_input = None
        self.example_text_input = None

        if 'cuda' in self.device:
            self.time_fn = partial(cuda_timestamp, device=self.device)
        else:
            self.time_fn = timestamp

        self.num_warm_iter = num_warm_iter
        self.num_bench_iter = num_bench_iter
        self.log_freq = num_bench_iter // 5

        self.model.eval() # inference mode

    def _init_input(self):
        self.example_image_input = torch.ones((self.batch_size,) + self.image_input_size, device=self.device, dtype=self.data_dtype)
        self.example_text_input = torch.ones((self.batch_size,) + self.text_input_size, device=device, dtype=torch.int64)

        if self.channels_last:
            self.example_image_input = self.example_image_input.contiguous(memory_format=torch.channels_last)

    def run(self):
        # print(self.model)
        def _step():
            t_step_start = self.time_fn()
            with self.amp_autocast():
                image_features, text_features, logit_scale = self.model(self.example_image_input, self.example_text_input)
                # image_features /= image_features.norm(dim=-1, keepdim=True)
                # text_features /= text_features.norm(dim=-1, keepdim=True)
                # text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

            t_step_end = self.time_fn(True)
            return t_step_end - t_step_start

        _logger.info(
            f'Running inference benchmark on {self.model_name} for {self.num_bench_iter} steps w/ '
            f'input image size {self.image_input_size} text size {self.text_input_size} and batch size {self.batch_size}.')


        with torch.no_grad():
            self._init_input()

            for _ in range(self.num_warm_iter):
                _step()

            total_step = 0.
            num_samples = 0
            t_run_start = self.time_fn()
            for i in range(self.num_bench_iter):
                delta_fwd = _step()
                total_step += delta_fwd
                num_samples += self.batch_size
                num_steps = i + 1
                # if num_steps % self.log_freq == 0:
                #     _logger.info(
                #         f"Infer [{num_steps}/{self.num_bench_iter}]."
                #         f" {num_samples / total_step:0.2f} samples/sec."
                #         f" {1000 * total_step / num_steps:0.3f} ms/step.")
            t_run_end = self.time_fn(True)
            t_run_elapsed = t_run_end - t_run_start
        
        results = dict(
            model= self.model_name,
            samples_per_sec=round(num_samples / t_run_elapsed, 2),
            step_time=round(1000 * total_step / self.num_bench_iter, 3),
            batch_size=self.batch_size,
            img_size=self.image_input_size[-1],
        )

        _logger.info(
            f"Inference benchmark of {self.model_name} done. "
            f"{results['samples_per_sec']:.2f} samples/sec, {results['step_time']:.2f} ms/step")
        
        return results

def profile_throughput(
        model_name,
        args,
):
    image_input_size=(3, args.size, args.size)
    text_input_size=(77,)
    torch.cuda.empty_cache()
    bench = BenchmarkRunner(model_name=model_name, image_input_size=image_input_size, text_input_size=text_input_size, batch_size=args.batch_size, channels_last= not args.channels_first)
    results = bench.run()
    return results


def profile_fvcore(
        model,
        image_input_size=(3, 224, 224),
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

    text_input_size = (77,)

    results = {}
    results['model'] = model_name
    results['image_size'] = image_input_size[1]

    model_cfg = open_clip.get_model_config(model_name)
    if model_cfg:
        vision_cfg = open_clip.CLIPVisionCfg(**model_cfg['vision_cfg'])
        text_cfg = open_clip.CLIPTextCfg(**model_cfg['text_cfg'])
        results['image_width'] = int(vision_cfg.width)
        results['text_width'] = int(text_cfg.width)
        results['embed_dim'] = int(model_cfg['embed_dim'])
    else:
        results['image_width'] = 0
        results['text_width'] = 0
        results['embed_dim'] = 0

    retries = 2
    while retries:
        retries -= 1
        try:
            macs, acts = profile_fvcore(
                model, image_input_size=image_input_size, text_input_size=text_input_size, force_cpu=not retries)

            image_macs, image_acts = profile_fvcore_image(
                model.visual, image_input_size=image_input_size, force_cpu=not retries)

            text_macs, text_acts = profile_fvcore_text(
                model.text, text_input_size=text_input_size, force_cpu=not retries)


            results['gmacs'] = round(macs / 1e9, 2)
            results['macts'] = round(acts / 1e6, 2)
            results['mparams'] = round(count_params(model) / 1e6, 2)
            results['image_gmacs'] = round(image_macs / 1e9, 2)
            results['image_macts'] = round(image_acts / 1e6, 2)
            results['image_mparams'] = round(count_params(model.visual) / 1e6, 2)
            results['text_gmacs'] = round(text_macs / 1e9, 2)
            results['text_macts'] = round(text_acts / 1e6, 2)
            results['text_mparams'] = round(count_params(model.text) / 1e6, 2)

            results_throughput = profile_throughput(model_name, args)
            results['samples_per_sec'] = results_throughput['samples_per_sec']
            results['batch_size'] = results_throughput['batch_size']
            


        except RuntimeError as e:
            pass
    return results


def main():
    setup_default_logging()
    args = parser.parse_args()

    # FIXME accept a text file name to allow lists of models in txt/csv
    if args.model == 'all':
        parsed_model = open_clip.list_models()
    else:
        parsed_model = args.model.split(',')

    results = []
    for m in parsed_model:
        # row = profile_model(m, args)
        row = profile_throughput(m, args)
        results.append(row)

    df = pd.DataFrame(results, columns=results[0].keys())
    df = df.sort_values('samples_per_sec')
    print(df)
    if args.results_file:
        df.to_csv(args.results_file, index=False)


if __name__ == '__main__':
    main()
