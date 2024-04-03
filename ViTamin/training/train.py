import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype, CLIP, CustomTextCLIP
from .distributed import is_master
from .zero_shot import zero_shot_eval
from .precision import get_autocast

def bin_balanced_sampling(scores, images, texts, N, M):
    # A: scores
    # B: image/text
    # M: number of each bins

    # Step 1: Sort tensor A
    sorted_indices = torch.argsort(scores)
    sorted_scores = scores[sorted_indices]
    sorted_images = images[sorted_indices]
    sorted_texts = texts[sorted_indices]
    
    # Step 2: Divide into N bins
    bins = torch.linspace(sorted_scores[0], sorted_scores[-1], steps=N+1)
    binned_indices = torch.bucketize(sorted_scores, bins)
    
    # Step 3: Sample M values from each bin
    sampled_indices = []
    non_selected_indices = set(range(len(scores)))
    
    for i in range(1, N+1):
        bin_indices = (binned_indices == i).nonzero().squeeze(-1)
        
        if bin_indices.numel() > M:
            selected_indices = bin_indices[torch.randperm(bin_indices.numel())[:M]]
            
        else:
            selected_indices = bin_indices
        # print("selected_indices", i, N, bin_indices.numel(), M, selected_indices, (binned_indices == i).nonzero().size()) 
        
        sampled_indices.extend(selected_indices.tolist())

        non_selected_indices -= set(selected_indices.tolist())

    # Step 4: Sample from non-selected values for bins with fewer than M values
    for i in range(1, N+1):
        bin_indices = (binned_indices == i).nonzero().squeeze()
        # print(i, bin_indices.numel(), M)
        if bin_indices.numel() < M:
            required_samples = M - bin_indices.numel()
            additional_samples = torch.tensor(list(non_selected_indices))[torch.randperm(len(non_selected_indices))[:required_samples]]
            sampled_indices.extend(additional_samples.tolist())
            non_selected_indices -= set(additional_samples.tolist())
            # print(i, non_selected_indices)
    perm = torch.randperm(len(sampled_indices)) # Shuffle using the same permutation

    return sorted_images[sampled_indices][perm], sorted_texts[sampled_indices][perm]

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2]
    }

def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None, start_step=None,):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    
    step_s1 = time.time()
    torch.cuda.synchronize()
    for i, batch in enumerate(dataloader):
        # torch.cuda.synchronize()
        # cur_step_s1 = time.time()
        # torch.cuda.synchronize()
        # if i > 0:
        #     step_e1 = time.time()
        #     torch.cuda.synchronize()
        #     step_time = (step_e1-step_s1) / i
        #     if is_master(args):
        #         print("------ step time: ", step_time)
        #         print("------ data loading", step_e1 - cur_step_end)
            
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if start_step is not None and start_step != 0 and step < start_step: # added by jieneng
            if is_master(args):
                print("Skip step ", i_accum)
            continue

        if not args.skip_scheduler:
            scheduler(step)
        
        if len(batch) == 3:
            # mix of real-distribution
            images, texts, scores = batch
            if args.bin_balanced_sampling_nbins is not None:
                M = int(len(scores) / args.bin_balanced_sampling_expand / args.bin_balanced_sampling_nbins)
                images, texts = bin_balanced_sampling(scores, images, texts, args.bin_balanced_sampling_nbins, M)
        else:
            images, texts = batch

        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        batch_size = len(images)


        data_time_m.update(time.time() - end)
        optimizer.zero_grad()
        

        if args.accum_freq == 1:
            with autocast():

                # torch.cuda.synchronize()
                # foward_s1 = time.time()
                # torch.cuda.synchronize()
                # if is_master(args):
                #     print("------ data convert, zero grad", foward_s1 - cur_step_s1)

                model_out = model(images, texts)


                # torch.cuda.synchronize()
                # foward_e1 = time.time()
                # torch.cuda.synchronize()
                # if is_master(args):
                #     print("------ forward time", foward_e1-foward_s1)
                # torch.cuda.synchronize()
                # loss_s1 = time.time()
                # torch.cuda.synchronize()

                logit_scale = model_out["logit_scale"]
                if args.distill:
                    with torch.no_grad():
                        dist_model_out = dist_model(images, texts)
                    model_out.update({f'dist_{k}' : v for k, v in dist_model_out.items()})
                losses = loss(**model_out, output_dict=True)


                total_loss = sum(losses.values())
                losses["loss"] = total_loss
                

                # torch.cuda.synchronize()
                # loss_e1 = time.time()
                # torch.cuda.synchronize()
                # if is_master(args):
                #     print("------ loss time", loss_e1-loss_s1)

            # torch.cuda.synchronize()
            # backward_s1 = time.time()
            # torch.cuda.synchronize()

            backward(total_loss, scaler)

            # torch.cuda.synchronize()
            # backward_e1 = time.time()
            # torch.cuda.synchronize()
            # if is_master(args):
            #     print("------ backward time", backward_e1-backward_s1)

        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)
                    model_out.pop("logit_scale")
                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                
                with autocast():
                    model_out = model(images, texts)
                    logit_scale = model_out.pop("logit_scale")
                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] +  [model_out[key]] + accumulated[j + 1:])
                    losses = loss(**inputs, logit_scale=logit_scale, output_dict=True)
                    del inputs
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss
                backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()


        del images
        del texts
        torch.cuda.empty_cache()
        
        # torch.cuda.synchronize()
        # optim_e1 = time.time()
        # torch.cuda.synchronize()
        # if is_master(args):
        #     print("------ optim step time", optim_e1-backward_e1)
        
        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and i_accum % args.log_every_n_steps == 0:
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})" 
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }            
            log_data.update({name:val.val for name,val in losses_m.items()})

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()


        #  adopt after-time step in clipa-v1
        completed_step = step + 1
        if any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')):
            should_zero_eval = args.zeroshot_steps != 0 and completed_step % args.zeroshot_steps == 0
            should_val = args.val_steps != 0 and completed_step % args.val_steps == 0
            # print(f"should_zero_eval {should_zero_eval} should_val {should_val} step {step} {args.zeroshot_steps} {args.val_steps}")

            metrics = evaluate(model=model, data=data,
                    epoch=epoch + i_accum / num_batches_per_epoch,
                    args=args,
                    tb_writer=tb_writer,
                    step=completed_step,
                    should_zero_eval=should_zero_eval,
                    should_val=should_val,
                    )
            # print(metrics)
            if should_zero_eval or should_val:
                model.train()

        # Saving checkpoints every n step.
        if args.save_logs and args.save_every_n_steps != 0 and (completed_step % args.save_every_n_steps) == 0:
            checkpoint_dict = {
                "step": completed_step,
                "cur_epoch": epoch,
                "name": args.name,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if scaler is not None:
                checkpoint_dict["scaler"] = scaler.state_dict()

            torch.save(
                checkpoint_dict,
                os.path.join(args.checkpoint_path, f"step_{completed_step}.pt"),
            )

            if args.delete_prev_step_ckpt:
                previous_checkpoint = os.path.join(args.checkpoint_path, f"step_{completed_step - args.save_every_n_steps}.pt")
                if os.path.exists(previous_checkpoint):
                    os.remove(previous_checkpoint)
    
        # torch.cuda.synchronize()
        # cur_step_end = time.time()
        # if is_master(args):
        #     print("------ unwrap log time",cur_step_end- optim_e1)
        #     print("---- cur step time", cur_step_end - cur_step_s1)
        #     print("---- avg step time", (cur_step_end - step_s1) / (i+1))

    # end for


def evaluate(model, data, epoch, args, tb_writer=None, step=None, should_zero_eval=False, should_val=False):
    metrics = {}
    if not is_master(args):
        return metrics
    
    if not should_zero_eval and not should_val:
        return metrics

    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, should_zero_eval)
    # print(step, zero_shot_metrics)

    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    # should_val = (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs))
    if 'val' in data and should_val:
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                images, texts = batch
                images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                with autocast():
                    model_out = model(images, texts)
                    image_features = model_out["image_features"]
                    text_features = model_out["text_features"]
                    logit_scale = model_out["logit_scale"]
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2

                    gen_loss = maybe_compute_generative_loss(model_out)

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Clip Loss: {cumulative_loss / num_samples:.6f}\t")

                    if gen_loss is not None:
                        cumulative_gen_loss += gen_loss * batch_size
                        logging.info(
                            f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")

            val_metrics = get_clip_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "clip_val_loss": loss.item(), "epoch": epoch, "step": step, "num_samples": num_samples}
            )
            if gen_loss is not None:
                gen_loss = cumulative_gen_loss / num_samples
                metrics.update({"val_generative_loss": gen_loss.item()})

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    if args.save_logs:
        for name, val in metrics.items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"val/{name}", val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        for name, val in metrics.items():
            wandb.log({f"val/{name}": val, 'epoch': epoch})

    return metrics


def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)
