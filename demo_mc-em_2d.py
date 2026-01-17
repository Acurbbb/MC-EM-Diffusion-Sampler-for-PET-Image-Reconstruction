from functools import partial
import os
import argparse
import random

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.sparse import load_npz, eye

from utils.utils import get_cond_schedule, save_gray_with_colorbar
from utils.utils_train import instantiate_from_config
from utils.utils_cli import _normalize_overrides, print_config_report
from utils.operator_pet import get_projection, PETOperator



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
                        default="./results/mc-em_2d/", help="Output directory.")
    parser.add_argument("--base", type=str,
                        default="./configs/recon_mc-em_2d.yaml", help="Path to base YAML config.")
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
        return diffusion.sample_mcem_ddim, partial(pet_op.em_update)
    elif method =="pet-dds":
        return diffusion.sample_pet_dds, partial(pet_op.em_update_prox, lamda=kwargs.get("lamda"))
    elif method == "dps":
        return diffusion.sample_dps, partial(pet_op.grad_fn, step_size=kwargs.get("step_size"))
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
        f"{cond_cfg.method}_{params_str}{eta_str}{cond_sched_str}_"
        f"phan{os.path.splitext(os.path.basename(cfg.phantom))[0]}_{cfg.seed}"
    )
    return file_out

def recon(cfg):
    set_seed(cfg.seed)
    
    # Device setting
    device_str = f"cuda:0" if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)  

    # PET-related
    G = load_npz(cfg.system_matrix)
    if hasattr(cfg, "ipsf_model"):
        P = load_npz(cfg.ipsf_model)
    else:
        P = eye(G.shape[1], format='coo')

    load_file = np.load(cfg.phantom)
    x_true = load_file['x0']
    imgsize = tuple(cfg.recon.imgsize)
    x_true = x_true.ravel()    # phantom
    u = load_file['u'].ravel() # attenuation

    # get projection
    count = int(cfg.count * 1e6)
    yi, ni, ri = get_projection(G, P, x_true, u, count, random=0.2)

    # MLEM
    pet_op = PETOperator(G=G, P=P, yi=yi, ni=ni, ri=ri)
    x_mlem, _, lkl_mlem = pet_op.do_mlem(x=None, maxit=20)

    print('MLEM finished...')
    # save_gray_with_colorbar(x_mlem.reshape(imgsize), 'mlem.png')
    # save_gray_with_colorbar(u.reshape(imgsize), 'u.png')
    # save_gray_with_colorbar(x_true.reshape(imgsize), 'x_true.png')

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
    x_T = torch.randn((1, 1, *imgsize), device=device)

    cond_schedule = get_cond_schedule(gen_steps, **cfg.condition.cond_step_schedule)
    img = sample_fn(
        model,
        x_T, 
        em_update_fn=em_update_fn,
        cond_schedule=cond_schedule,
        clip_denoised=cfg.diffusion.clip_denoised,
        save_steps=cfg.save_steps,
    )

    file_out = _get_filename(diffusion, cfg)
    OmegaConf.save(cfg, os.path.join(savedir, file_out+".yaml"))
    np.savez(
        os.path.join(savedir, file_out+".npz"), 
        x=img, 
        cond_steps=np.sum(np.array(cond_schedule)),
        gen_steps=len(cond_schedule),
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