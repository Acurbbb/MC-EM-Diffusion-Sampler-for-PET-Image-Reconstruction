import os 
import sys
import datetime
import time
import copy
import math
import argparse
import warnings
from contextlib import nullcontext
import numpy as np
import functools
from omegaconf import OmegaConf
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Tuple, Sequence, Optional, Any, List

from utils.utils import RandomPaddedPatchDataset3D, show_orthogonal_views
from utils.utils_train import setup_ddp, cleanup_ddp, check_gpu, create_ema_copy, ema_fp32_from_master, get_scheduler_lambda_fn, instantiate_from_config, log_tensorboard
from diffusion.unet.fp16_util import MixedPrecisionTrainer, master_params_to_model_params
from diffusion.patch_diffusion_alone import GaussianDiffusion

def get_cli_conf():
    parser = argparse.ArgumentParser(description="Train your model")
    # define whatever args you want to override
    parser.add_argument("--output_dir", type=str, 
                        default="./results/padis_3d_tmp", help="output dir")
    parser.add_argument("--base", type=str, 
                        default="./configs/train_patch_3d.yaml", help="path to base configs. Parameters can be overwritten or added with command-line options of the form `--key value`.")
    parser.add_argument("--parallel", action="store_true", 
                        default=False)
    parser.add_argument("--lr", type=float, 
                        default=1e-4)
    parser.add_argument("--total_steps", type=int, 
                        default=100000)
    parser.add_argument("--ckpt_path", type=str,
                        default="")
    parser.add_argument("--save_step", type=int, 
                        default=10000)
    parser.add_argument("--ema_decay", type=float,
                        default=0.9999)
    parser.add_argument("--use_fp16", action="store_true",
                        default=False)
    parser.add_argument("--accumulate_grad_batches", type=int, 
                        default=2, help="Number of micro-batches to accumulate before an optimizer step.")    
    args = parser.parse_args()

    # turn the Namespace into a dict, filter out args not set (None)
    cli_args = {k: v for k, v in vars(args).items() if v is not None}

    # create an OmegaConf from that dict
    return OmegaConf.create(cli_args)

def save_flags(cfg, savedir):
    OmegaConf.save(cfg, os.path.join(savedir, "hparams.yaml"))

def setup_optim_sched_and_resume(
    *,
    cfg,
    rank: int,
    mp_trainer,         # your MixedPrecisionTrainer (already constructed with the model)
    ema_model,          # your EMA fp32 copy (already constructed)
    make_optimizer=None # optional factory: fn(params) -> optimizer; defaults to Adam(lr=cfg.lr)
):
    """
    Returns: optimizer, scheduler, start_epoch, start_step, resumed(bool)

    Behavior:
    - If cfg.ckpt_path exists: 
        * Restores weights (FP32 masters -> model if fp16; or model.load_state_dict if fp32)
        * Recreates optimizer BOUND to current master params
        * Loads optimizer/scheduler/EMA states
        * Restores loss scale (fp16)
    - Else (train from scratch):
        * Leaves model as-is (mp_trainer already set it up)
        * Creates fresh optimizer/scheduler
        * start_epoch = start_step = 0
    """
    def _default_make_optimizer(params):
        return torch.optim.Adam(params, lr=cfg.lr)

    make_optimizer = make_optimizer or _default_make_optimizer

    # Build scheduler lambda
    lambda_fn = get_scheduler_lambda_fn(**cfg.scheduler_config)

    start_epoch = 0
    start_step  = 0
    resumed     = False

    ckpt_path = getattr(cfg, "ckpt_path", None)
    ckpt = None
    if ckpt_path and os.path.exists(ckpt_path):
        # --- Load checkpoint payload ---
        ckpt = torch.load(ckpt_path, map_location={'cuda:0': f'cuda:{rank}'})
        resumed = True

        # Loss scale (fp16)
        mp_trainer.lg_loss_scale = ckpt.get('lg_loss_scale', mp_trainer.lg_loss_scale)
        # Rebuild NEW FP32 masters from ckpt's model-state
        mp_trainer.master_params = mp_trainer.state_dict_to_master_params(ckpt['master_state'])
        # Copy masters -> (fp16) model params
        master_params_to_model_params(mp_trainer.param_groups_and_shapes, mp_trainer.master_params)

        # EMA
        try:
            ema_model.load_state_dict(ckpt['ema_state'])
        except Exception as e:
            warnings.warn(f"[rank {rank}] Failed to load EMA; continuing with current EMA. {e}")

        # Epoch/step
        start_epoch = ckpt.get('epoch', 0)
        start_step  = ckpt.get('step', 0)

    # --- Create optimizer bound to *current* master params (whether resumed or scratch) ---
    optimizer = make_optimizer(mp_trainer.master_params)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_fn)

    # If resuming, now that optimizer is bound to the correct tensors, load its state
    if ckpt is not None:
        try:
            optimizer.load_state_dict(ckpt['optim_state'])
        except Exception as e:
            warnings.warn(f"[rank {rank}] Failed to load optimizer state; starting fresh. {e}")
        try:
            scheduler.load_state_dict(ckpt['sched_state'])
        except Exception as e:
            warnings.warn(f"[rank {rank}] Failed to load scheduler state; starting fresh. {e}")

    return optimizer, scheduler, start_epoch, start_step, resumed

def save_checkpoint(state, path, rank):
    """Save training state (only on rank 0)"""
    if rank == 0:
        torch.save(state, path)


def train_ddp(rank, world_size, cfg):
    # device
    if torch.cuda.is_available():
        torch.cuda.set_device(rank) # pin this process to its own GPU
        if world_size > 1:
            setup_ddp(rank, world_size)
        device = torch.device(f'cuda:{rank}')
        check_gpu(rank)
    else:
        device = torch.device('cpu')

    # Datasets & Sampler
    ds_train = instantiate_from_config(cfg.data.train)
    
    accum_steps = max(1, int(cfg.accumulate_grad_batches))
    micro_bs    = int(cfg.data.train.batch_size)   # interpret existing config as *micro*-batch per rank
    loader_bs   = micro_bs * accum_steps           # per-rank effective batch per optimizer step

    sampler = DistributedSampler(ds_train, num_replicas=world_size, rank=rank) if world_size>1 else None
    loader_train = DataLoader(
        ds_train, batch_size=loader_bs,
        sampler=sampler, shuffle=(sampler is None),
        num_workers=cfg.data.train.num_workers, pin_memory=True,
    )

    cropper = instantiate_from_config(cfg.data.cropper)

    # create fp32 model first
    net = instantiate_from_config(cfg.model)
    net = net.to(device)
    ema_model = create_ema_copy(net) # ema copy keeps fp32

    # mixed precision trainer
    mp_trainer = MixedPrecisionTrainer(
        model=net,
        use_fp16=cfg.use_fp16,
    ) # type cast happens

    if world_size > 1:
        net = DDP(net, device_ids=[rank])
        net_module = net.module
    else:
        net_module = net

    optimizer, scheduler, start_epoch, start_step, resumed = setup_optim_sched_and_resume(cfg=cfg, rank=rank, mp_trainer=mp_trainer, ema_model=ema_model)
    print(f"start epoch: {start_epoch}, start step: {start_step}")
    
    # TensorBoard (only rank 0)
    if rank == 0:
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        tb = SummaryWriter(log_dir=os.path.join(cfg.output_dir, 'logs'+now)) 
    savedir = os.path.join(cfg.output_dir, "ckpt")
    os.makedirs(savedir, exist_ok=True)

    global_step = start_step

    # Compute steps per epoch
    steps_per_epoch = len(loader_train)
    num_epochs = math.ceil(cfg.total_steps / steps_per_epoch)
    
    # Prepare diffusion model
    diffusion = instantiate_from_config(cfg.diffusion)

    epoch_bar = tqdm(range(start_epoch, num_epochs), desc="Outer train loop", disable=(rank!=0), position=0)
    for epoch in epoch_bar:
        if sampler is not None:
            sampler.set_epoch(epoch)

        train_bar = tqdm(loader_train, desc=f"Rank {rank}, epoch {epoch}", position=rank+1, leave=True)
        for big_bat_x, big_bat_pos in train_bar:
            if global_step >= cfg.total_steps:
                break
            # Forward
            net.train()
            mp_trainer.zero_grad()

            # crop into smaller patches (optional)
            cropper.choose(big_bat_x.shape)
            big_bat_x = cropper.crop(big_bat_x)
            big_bat_pos = cropper.crop(big_bat_pos)

            # Split large batch into micro-batches on the fly (CPU -> GPU per micro)
            B = big_bat_x.shape[0]
            num_micros = (B + micro_bs - 1) // micro_bs
            sizes = [min(micro_bs, B - i * micro_bs) for i in range(num_micros)]

            running_loss = 0.
            for mi in range(num_micros):
                s = slice(mi * micro_bs, min((mi + 1) * micro_bs, B)) 
                bat_x   = big_bat_x[s].to(device)
                bat_pos = big_bat_pos[s].to(device)

                x = bat_x
                cond = {
                    "x_pos": bat_pos
                }

                t = torch.randint(0, diffusion.num_timesteps, (bat_x.shape[0],), device=device).long()

                compute_losses = functools.partial(
                    diffusion.p_losses,
                    net,
                    x,
                    t,
                    model_kwargs=cond,
                )

                is_last_micro = (mi == num_micros - 1)
                sync_ctx = (net.no_sync() if (world_size > 1 and isinstance(net, DDP) and not is_last_micro) else nullcontext())
                with sync_ctx:
                    loss_micro = compute_losses()
                    loss_scaled = loss_micro * (sizes[mi] / B)
                    mp_trainer.backward(loss_scaled)
                    running_loss += float(loss_scaled.detach().item())

            # Optimizer update
            success = mp_trainer.optimize(optimizer)
            if not success:
                continue
            # Scheduler update
            scheduler.step()

            # Update EMA
            ema_fp32_from_master(mp_trainer, ema_model, cfg.ema_decay)

            # Logging
            if rank == 0:
                loss_train = running_loss
                lr_log = scheduler.get_last_lr()[0]
                grad_norm = mp_trainer.last_grad_norm
                log_tensorboard(tb, step=global_step, prefix="train", loss=loss_train, lr=lr_log, 
                                grad_norm=grad_norm, lg_loss_scale=mp_trainer.lg_loss_scale)
                train_bar.set_postfix(loss=f"{loss_train:.2e}", lr=f"{lr_log:.2e}", grad_norm=f"{grad_norm:.2e}")

            # Checkpoint & samples
            if cfg.save_step>0 and global_step % cfg.save_step==0 and rank==0:
                patch_size = (ds_train.pD, ds_train.pH, ds_train.pW)
                unpadded_shape = ds_train._shapes[0]  # take the first subject
                pad_margin = ds_train._pad_margins[0] # take the first subject
                padded_shape = tuple(map(lambda N, M: N + 2 * M, unpadded_shape, pad_margin))
         
                ema_model.eval()
                with torch.no_grad():
                    volume_sample = diffusion.sample(model=ema_model, padded_shape=padded_shape, 
                                                     patch_size=patch_size, pad_margin=pad_margin,
                                                     save_steps=0).squeeze().cpu()
                    volume_sample = ds_train.unnormalize(volume_sample)
                ema_model.train()
                
                show_orthogonal_views(volume_sample, vmin=0., vmax=1., 
                                      save_path=os.path.join(savedir, f"sample_step{global_step}.png"))
                # save checkpoint
                save_checkpoint({
                    'epoch': epoch,
                    'step':  global_step,
                    'optim_state': optimizer.state_dict(),
                    'sched_state': scheduler.state_dict(),
                    'master_state': mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
                    'ema_state':    ema_model.state_dict(),
                    'lg_loss_scale': mp_trainer.lg_loss_scale,}, 
                    os.path.join(savedir,f"ckpt_{global_step}.pt"), rank)
                    
                if world_size > 1:
                    dist.barrier()

            global_step += 1

    # Final checkpoint
    if rank==0:
        save_checkpoint({
            'epoch': epoch,
            'step':  global_step,
            'optim_state': optimizer.state_dict(),
            'sched_state': scheduler.state_dict(),
            'master_state': mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
            'ema_state':    ema_model.state_dict(),
            'lg_loss_scale': mp_trainer.lg_loss_scale,}, 
            os.path.join(savedir,f"ckpt_{global_step}.pt"), rank)
        tb.close()
    
    if world_size>1:
        cleanup_ddp()


def main():
    # CLI arguments
    cli_conf = get_cli_conf()
    # load base config
    yaml_conf = OmegaConf.load(cli_conf.base)
    # merge
    cfg = OmegaConf.merge(yaml_conf, cli_conf)

    os.makedirs(cfg.output_dir, exist_ok=True)
    save_flags(cfg, cfg.output_dir)

    if cfg.parallel and torch.cuda.device_count()>1:
        world_size = torch.cuda.device_count()
        mp.spawn(train_ddp, args=(world_size,cfg), nprocs=world_size, join=True)
    else:
        # single-process (rank=0, world_size=1)
        train_ddp(rank=0, world_size=1, cfg=cfg)        
    

if __name__ == "__main__":
    main()
