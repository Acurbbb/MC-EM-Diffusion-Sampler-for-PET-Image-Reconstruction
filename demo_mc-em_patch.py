from functools import partial
import os
import argparse
import random

import numpy as np
import torch
from omegaconf import OmegaConf

from utils.utils import show_orthogonal_views, get_cond_schedule, get_pad_margin
from utils.utils_train import instantiate_from_config
from utils.utils_cli import _normalize_overrides, print_config_report
from utils.scanner import PETScanner
from utils.operator_pet import PETOperatorLM


def set_seed(seed):
    torch.manual_seed(seed)           # Set seed for CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # Set seed for GPU (if available)
        torch.cuda.manual_seed_all(seed)  # Set seed for all GPUs (if using multi-GPU)
    random.seed(seed)                 # Set seed for Python random module
    np.random.seed(seed)              # Set seed for NumPy random module
    torch.backends.cudnn.deterministic = True  # Make CuDNN deterministic
    torch.backends.cudnn.benchmark = False     # Disable benchmark mode for reproducibility


def get_cli_conf() -> OmegaConf:
    parser = argparse.ArgumentParser(description="Run your recon")
    parser.add_argument("--output_dir", type=str,
                        default="./results/mc-em_patch/ER323", help="Output directory.")
    parser.add_argument("--base", type=str,
                        default="./configs/recon_mc-em_patch.yaml", help="Path to base YAML config.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    args, unknown = parser.parse_known_args()

    known_cfg = OmegaConf.create({k: v for k, v in vars(args).items() if v is not None})

    # normalize before parsing with OmegaConf
    overrides = _normalize_overrides(unknown)
    override_cfg = OmegaConf.from_cli(overrides)

    # Merge known first, then overrides (CLI wins)
    return OmegaConf.merge(known_cfg, override_cfg)

def _get_sample_fn(method, diffusion, pet_op, **kwargs):
    if method == "mc-em":
        return diffusion.sample_mcem_ddim_patch, partial(pet_op.em_update)
    elif method =="pet-dds":
        return diffusion.sample_pet_dds_patch, partial(pet_op.em_update_prox, lamda=kwargs.get("lamda"))
    elif method == "dps":
        return diffusion.sample_dps_patch, partial(pet_op.grad_fn, step_size=kwargs.get("step_size"))
    else:
        raise ValueError("Invalid condition method!")
    
def _get_filename(diffusion, cfg):
    gen_steps = diffusion.sampling_timesteps
    eta_str = f"eta{diffusion.eta}_" if gen_steps < 1000 else ""
    step_cfg = cfg.condition.cond_step_schedule
    cond_cfg = cfg.condition

    if cond_cfg.method == "pet-dds":
        params_str = f"lam{cond_cfg.params.lamda}_"
    elif cond_cfg.method == "dps":
        params_str = f"ss{cond_cfg.params.step_size}_"
    else:
        params_str = ""

    if step_cfg.name == 'linear':
        cond_sched_str = f"gen{gen_steps}_{step_cfg.t_start}-{step_cfg.t_end}-{step_cfg.interval_start}-{step_cfg.interval_end}"
    elif step_cfg.name == 'subiter':
        cond_sched_str = f"gen{gen_steps}_subit{step_cfg.subiter}"

    file_out = (
        f"{cond_cfg.method}_{params_str}{eta_str}{cond_sched_str}_{cfg.seed}"
    )
    return file_out

def recon(cfg):
    set_seed(cfg.seed)
    
    # Device setting
    device_str = f"cuda:0" if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)  

    # PET-related
    imgsize = tuple(cfg.recon.imgsize)
    voxsize = tuple(cfg.recon.voxsize)
    patch_size = cfg.recon.patch_size
    pad_margin = tuple(map(get_pad_margin, imgsize, patch_size))
    imgsize_padded = tuple(map(lambda N, M: N + 2*M, imgsize, pad_margin))
    num_subsets = cfg.recon.subsets
    scanner = PETScanner(cfg.recon.scanner)
    
    lmdata = np.fromfile(cfg.lm_data, 
                         dtype=np.int16).reshape((-1,5))
    add_fac = np.fromfile(cfg.add_fac, 
                          dtype=np.float32)
    sens = np.fromfile(cfg.sens,
                       dtype=np.float32).reshape(imgsize_padded)
    sens = sens.transpose((0,2,1)) # hard-coded transpose for Fortran layout
    
    pet_op = PETOperatorLM(
        scanner=scanner,
        num_subsets=num_subsets,
        imgsize=imgsize_padded,
        voxsize=voxsize,
        lmdata=lmdata,
        add_fac=add_fac,
        sens=sens,
    )

    # get scale z
    x_mlem = pet_op.osem(n_iter=cfg.recon.pre_recon_iter).reshape(imgsize_padded)
    # blur image
    x_mlem = pet_op.gaussian_blur(x_mlem, psf_sigma_mm=2, voxel_size_mm=voxsize)
    # show_orthogonal_views(x_mlem, save_path='x_mlem_3d.png')

    perct = cfg.recon.max_intensity_perct
    max_intensity = np.percentile(x_mlem.flatten(), perct)
    print(f"Max intensity [{max_intensity.item()}] inferred at [{perct}] percentile. OSEM iteration [{cfg.recon.pre_recon_iter}]")

    # Prepare conditioning method
    ckpt = torch.load(cfg.ckpt_path, map_location="cpu")
    model = instantiate_from_config(cfg.model)
    model.load_state_dict(ckpt['ema_state'])
    model = model.to(device).eval()

    diffusion = instantiate_from_config(cfg.diffusion)
    gen_steps = diffusion.sampling_timesteps
   
    # Load diffusion sampler
    sample_fn, em_update_fn = _get_sample_fn(cfg.condition.method, diffusion, pet_op, **cfg.condition.params)
   
    # Working directory
    savedir = cfg.output_dir
    os.makedirs(savedir, exist_ok=True)

    # Sampling
    x_T = torch.randn((1, 1, *imgsize_padded), device=device)

    cond_schedule = get_cond_schedule(gen_steps, **cfg.condition.cond_step_schedule)
    img = sample_fn(
        model,
        x_T, 
        patch_size=patch_size,
        pad_margin=pad_margin,
        scale=max_intensity.item(),
        em_update_fn=em_update_fn,
        n_subsets=num_subsets,
        cond_schedule=cond_schedule,
        clip_denoised=cfg.diffusion.clip_denoised,
        save_steps=cfg.save_steps,
    )

    file_out = _get_filename(diffusion, cfg)
    OmegaConf.save(cfg, os.path.join(savedir, file_out+".yaml"))
    np.savez(
        os.path.join(savedir, file_out+".npz"), 
        x=img, 
        # x0s=x0s, 
        cond_steps=np.sum(np.array(cond_schedule)),
        gen_steps=len(cond_schedule),
        max_intensity=max_intensity.item(),
    )


def main():
    # 1) CLI arguments (known + OmegaConf overrides)
    cli_conf = get_cli_conf()

    # 2) Load base YAML
    yaml_conf = OmegaConf.load(cli_conf.base)

    # 3) Merge: YAML first, then CLI (CLI wins)
    cfg = OmegaConf.merge(yaml_conf, cli_conf)

    # 4) Print report (moved into a separate function)
    print_config_report(yaml_conf, cfg)

    # 5) Make output dir & run
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Run your program
    recon(cfg)

if __name__ == "__main__":
    main()