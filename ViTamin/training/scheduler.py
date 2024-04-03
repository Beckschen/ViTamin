import numpy as np
import json

def create_optimizer_nnunet(args, model, lr_mul_nnunet, local_rank=0):
    names_encoder = ['conv_blocks_context', 'td']
    names_decoder = ['conv_blocks_localization', 'tu', 'seg_outputs']
    names_unit = ['linear_mask_features', 'input_proj', 'predictor']
    dict_lr_mult = {
        'encoder': lr_mul_nnunet[0], 
        'decoder': lr_mul_nnunet[1],
        'unit': lr_mul_nnunet[2] if len(lr_mul_nnunet) > 2 else 1.0,
        'head': lr_mul_nnunet[3] if len(lr_mul_nnunet) > 3 else 1.0,
        }
    
    parameter_group_vars = {}
    parameter_group_names = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if name.split('.')[0] in names_encoder:
            group_name = 'encoder'
        elif name.split('.')[0] in names_decoder:
            group_name = 'decoder'
        elif name.split('.')[0] in names_unit:
            if name.split('.')[1].startswith('cacls'): 
                group_name = 'head'
            else:
                group_name = 'unit'
        else:
            group_name = 'head'

        if group_name not in parameter_group_names:
            parameter_group_names[group_name] = {
                "weight_decay": args.weight_decay,
                "params": [],
                "lr_scale": dict_lr_mult[group_name]
            }
            parameter_group_vars[group_name] = {
                "weight_decay": args.weight_decay,
                "params": [],
                "lr_scale": dict_lr_mult[group_name]
            }

        parameter_group_vars[group_name]['params'].append(param)
        parameter_group_names[group_name]["params"].append(name)
    
        # big bug, freeze unit first (including heads)?
        if dict_lr_mult[group_name] == 0: # freeze
            param.requires_grad = False
        # else: # unfreeze head...
        #     param.requires_grad = True

 
    if local_rank==0:
        print("################### Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())

def assign_learning_rate(optimizer, new_lr):

    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr

def assign_learning_rate_factor(optimizer, new_lr_factor):
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["initial_lr"] * new_lr_factor

def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def const_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            lr = base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster


def const_lr_cooldown(optimizer, base_lr, warmup_length, steps, cooldown_steps, cooldown_power=1.0, cooldown_end_lr=0.):
    def _lr_adjuster(step):
        start_cooldown_step = steps - cooldown_steps
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            if step < start_cooldown_step:
                lr = base_lr
            else:
                e = step - start_cooldown_step
                es = steps - start_cooldown_step
                # linear decay if power == 1; polynomial decay otherwise;
                decay = (1 - (e/es)) ** cooldown_power
                lr = decay * (base_lr - cooldown_end_lr) + cooldown_end_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster


def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr

        if step == 0:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster

def cosine_lr_multiplier(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr_factor = (step + 1) / warmup_length
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr_factor = 0.5 * (1 + np.cos(np.pi * e / es))

        if step == 0:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])

        assign_learning_rate_factor(optimizer, lr_factor)

    return _lr_adjuster