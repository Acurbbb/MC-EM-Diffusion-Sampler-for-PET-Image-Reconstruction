import os
from functools import partial
import torch
import torch.distributed as dist
import torch.nn as nn
import copy
import importlib

def setup_ddp(rank, world_size):
    """Initialize NCCL process group for DDP"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)


def cleanup_ddp():
    """Cleanup DDP process group"""
    dist.destroy_process_group()


def check_gpu(rank):
    # print a little banner so you know exactly which GPU this rank is using
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    phys = visible.split(",")[rank] if visible else str(rank)
    print(f"[Rank {rank}] running on visible cuda:{rank} → physical GPU {phys}, GPU name: {torch.cuda.get_device_name(rank)}")


def warmup_lr(step, warmup_step):
    """Linear warmup: step/warmup until warmup, then 1.0 afterwards"""
    return min(step, warmup_step) / float(warmup_step)


def warmup_and_anneal_lr(step, warmup_step, plateau_step, plateau_lambda, start_lambda=0):
    """
    Linear warmup and linear annealing to plateau
        warmup_step: step it reaches peak
        plateau_step: step it reaches plateau
    """
    peak_val = 1.
    if step < warmup_step:
        return step / float(warmup_step) * (peak_val - start_lambda) + start_lambda
    else:
        return max(
            (plateau_step - step) / float(plateau_step - warmup_step) * (peak_val - plateau_lambda) + plateau_lambda,
            float(plateau_lambda)
        )

class ScheduleWrapper():
    def __init__(self, lambdalr_fn_type, **kwargs):
        self.type = lambdalr_fn_type

        if self.type == "warmup":
            try: 
                warmup = kwargs["warmup"]
            except KeyError:
                raise ValueError("warmup not found!")
            self.schedule = partial(warmup_lr, warmup=warmup)

        else:
            raise ValueError("scheduler not found!")
        
def get_scheduler_lambda_fn(lambda_fn_type, **kwargs):
    if lambda_fn_type == "warmup":
        return partial(warmup_lr, **kwargs)
    elif lambda_fn_type == "warmup_and_anneal":
        return partial(warmup_and_anneal_lr, **kwargs)
    elif lambda_fn_type == "const":
        return lambda t: 1.0
        
def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def create_ema_copy(model):
    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad_(False)
    device = next(model.parameters()).device
    ema_model.to(device)
    return ema_model


@torch.no_grad()
def ema(source: nn.Module, target: nn.Module, decay: float):
    # assumes source and target have identical architectures
    for (name_s, p_s), (name_t, p_t) in zip(
            source.named_parameters(), target.named_parameters()):
        # sanity check
        assert name_s == name_t, f"Param mismatch: {name_s} vs {name_t}"
        # EMA update in place
        p_t.data.mul_(decay)
        p_t.data.add_(p_s.data, alpha=1 - decay)


@torch.no_grad()
def ema_fp32_from_master(mp_trainer, ema_model: nn.Module, decay: float):
    """
    Update ema_model’s fp32 parameters by EMA’ing the fp32 master_params
    held inside mp_trainer.
    """
    master_state = mp_trainer.master_params_to_state_dict(mp_trainer.master_params)
    for name, p_t in ema_model.named_parameters():
        p_t.data.mul_(decay)
        p_t.data.add_(master_state[name], alpha=1.0 - decay)


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def log_tensorboard(tb, step, prefix="", **kwargs):
    for k, v in kwargs.items():
        full_tag = f"{prefix}/{k}" if prefix else k
        tb.add_scalar(full_tag, v, step)