"""
Standalone diffusion module modified from lucidrains' implementation
"""
import math
from typing import List
from functools import partial
from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader

from einops import rearrange, reduce

from tqdm.auto import tqdm

from utils.utils import ListPaddedPatchDataset3D, ListPaddedPatchDataset2D

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.to(t.device).gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)
    

    
class GaussianDiffusion():
    def __init__(
        self,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_noise',
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        min_snr_loss_weight = False,
        min_snr_gamma = 5,
        eta = 0.,
        clip_x0=(-1., 1.),
        bg_val = -1., # background value in the normalized images, usually -1
    ):
        super().__init__()

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        
        self.clip_x0_lower, self.clip_x0_upper = clip_x0

        self.bg_val = bg_val

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.eta = eta if self.is_ddim_sampling else None

        self.betas = betas.to(torch.float32)
        self.alphas_cumprod = alphas_cumprod.to(torch.float32)
        self.alphas_cumprod_prev = alphas_cumprod_prev.to(torch.float32)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(torch.float32)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod).to(torch.float32)
        self.log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod).to(torch.float32)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod).to(torch.float32)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1).to(torch.float32)

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.posterior_variance = posterior_variance.to(torch.float32)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.posterior_log_variance_clipped = torch.log(posterior_variance.clamp(min =1e-20)).to(torch.float32)
        self.posterior_mean_coef1 = (betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)).to(torch.float32)
        self.posterior_mean_coef2 = ((1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)).to(torch.float32)

        # loss weight

        snr = alphas_cumprod / (1 - alphas_cumprod)

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            loss_weight = maybe_clipped_snr / snr
        elif objective == 'pred_x0':
            loss_weight = maybe_clipped_snr
        elif objective == 'pred_v':
            loss_weight = maybe_clipped_snr / (snr + 1)

        self.loss_weight = loss_weight

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, model, x, t, model_kwargs=None, clip_x_start=False, rederive_pred_noise=False):
        model_output = model(x, t, **model_kwargs)
        maybe_clip = partial(torch.clamp, min=self.clip_x0_lower, max=self.clip_x0_upper) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)
    
    def model_predictions_3d(self, model, x, t:int, patch_size, pad_margin, clip_denoised=False, rederive_pred_noise=False, patches_ds=None, batch_size=4):
        device = x.device
        
        patches_ds = ListPaddedPatchDataset3D(
            padded=x,
            patch_size=patch_size,
            pad_margin=pad_margin
        ) if patches_ds is None else patches_ds

        patches_loader = DataLoader(
            patches_ds,
            batch_size=batch_size,
            sampler=None,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        x_start_all = []
        pred_noise_all = []
        for x_bat, pos_bat in patches_loader:
            batch_size = x_bat.shape[0]
            model_kwargs = {
                "x_pos": pos_bat,
            }
            batched_times = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            preds = self.model_predictions(model, x_bat, batched_times, model_kwargs, clip_x_start=clip_denoised, rederive_pred_noise=rederive_pred_noise)

            x_start_bat = preds.pred_x_start
            x_start_all.append(x_start_bat)

            pred_noise_bat = preds.pred_noise
            pred_noise_all.append(pred_noise_bat)
            
        x_start_all = torch.cat(x_start_all, dim=0)
        x_start_all = patches_ds.reconstruct_volume(x_start_all, margin_val=self.bg_val)

        # pred noise when x0 = 0
        pred_noise_from_0 = self.predict_noise_from_start(x_t=x, t=torch.full((1,), t, device=device, dtype=torch.long),
                                                          x0=self.bg_val * torch.ones_like(x_start_all, device=device, dtype=torch.float32))
        pred_noise_all = torch.cat(pred_noise_all, dim=0)
        pred_noise_all = patches_ds.reconstruct_volume(pred_noise_all, 
                                                       margin=pred_noise_from_0)
        return ModelPrediction(pred_noise_all, x_start_all)

    def model_predictions_2d(self, model, x, t:int, patch_size, pad_margin, clip_denoised=False, rederive_pred_noise=False):
        device = x.device
        
        patches_ds = ListPaddedPatchDataset2D(
            padded_batch=x,
            patch_size=patch_size,
            pad_margin=pad_margin
        )
        patches_loader = DataLoader(
            patches_ds,
            batch_size=64,
            sampler=None,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        x_start_all = []
        pred_noise_all = []
        for x_bat, pos_bat in patches_loader:
            batch_size = x_bat.shape[0]
            model_kwargs = {
                "x_pos": pos_bat,
            }
            batched_times = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            preds = self.model_predictions(model, x_bat, batched_times, model_kwargs, clip_x_start=clip_denoised, rederive_pred_noise=rederive_pred_noise)

            x_start_bat = preds.pred_x_start
            x_start_all.append(x_start_bat)

            pred_noise_bat = preds.pred_noise
            pred_noise_all.append(pred_noise_bat)
            
        x_start_all = torch.cat(x_start_all, dim=0)
        x_start_all = patches_ds.reconstruct_image(x_start_all, margin_val=self.bg_val)

        # pred noise when x0 = 0
        pred_noise_from_0 = self.predict_noise_from_start(x_t=x, t=torch.full((1,), t, device=device, dtype=torch.long),
                                                          x0=self.bg_val * torch.ones_like(x_start_all, device=device, dtype=torch.float32))
        pred_noise_all = torch.cat(pred_noise_all, dim=0)
        pred_noise_all = patches_ds.reconstruct_image(pred_noise_all, 
                                                       margin=pred_noise_from_0)
        return ModelPrediction(pred_noise_all, x_start_all)

    def p_mean_variance(self, model, x, t, model_kwargs = None, clip_denoised = True):
        preds = self.model_predictions(model, x, t, model_kwargs)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(self.clip_x0_lower, self.clip_x0_upper)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

        
    @torch.no_grad()
    def p_sample(self, model, x, t: int, patch_size, pad_margin, clip_denoised=True):
        dim = 3 if len(x.shape) == 5 else 2
        device = x.device
        
        if dim == 3:
            preds = self.model_predictions_3d(
                model=model, x=x, t=t,
                patch_size=patch_size,
                pad_margin=pad_margin,
            )
        else:
            preds = self.model_predictions_2d(
                model=model, x=x, t=t,
                patch_size=patch_size,
                pad_margin=pad_margin,
            )

        # DDPM clips the denoised after model evaluation (pred_noise is not affected by clipping)
        if clip_denoised:
            preds.pred_x_start.clamp_(self.clip_x0_lower, self.clip_x0_upper)

        model_mean, _, posterior_log_variance = self.q_posterior(
            x_start=preds.pred_x_start, x_t=x, 
            t=torch.full((1,), t, device=device, dtype=torch.long)
        )

        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * posterior_log_variance).exp() * noise
        return pred_img, preds.pred_x_start

    @torch.no_grad()
    def p_sample_loop(self, model, shape, patch_size, pad_margin, save_steps=0, clip_denoised=True):
        device = next(model.parameters()).device

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            img, x_start = self.p_sample(model, img, t, patch_size=patch_size, pad_margin=pad_margin, clip_denoised=clip_denoised)

            if save_steps > 0 and t % save_steps == 0:
                imgs.append(img)

        ret = img if save_steps == 0 else torch.stack(imgs, dim = 1)
        return ret

    @torch.no_grad()
    def ddim_sample(self, model, shape, patch_size, pad_margin, save_steps=0, clip_denoised=True):
        total_timesteps, sampling_timesteps = self.num_timesteps, self.sampling_timesteps
        dim = 3 if len(shape) == 5 else 2
        device = next(model.parameters()).device

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
        imgs = []

        x_start = None

        for i, (time, time_next) in enumerate(tqdm(time_pairs, desc = 'sampling loop time step')):
            if dim == 3:
                pred_noise, x_start = self.model_predictions_3d(
                    model=model, 
                    x=img, 
                    t=time,
                    patch_size=patch_size, 
                    pad_margin=pad_margin,
                    clip_denoised=clip_denoised,
                    rederive_pred_noise=True
                )
            else:
                pred_noise, x_start = self.model_predictions_2d(
                    model=model, 
                    x=img, 
                    t=time,
                    patch_size=patch_size, 
                    pad_margin=pad_margin,
                    clip_denoised=clip_denoised,
                    rederive_pred_noise=True
                )

            if time_next < 0:
                img = x_start
                if save_steps > 0:
                    imgs.append(img.cpu().numpy())
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            eta = self.eta

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
            
            if save_steps > 0 and i % save_steps == 0:
                imgs.append(img.cpu().numpy())

        ret = img if save_steps == 0 else np.concatenate(imgs, axis=0)
        return ret

    @torch.no_grad()
    def sample(self, model, padded_shape, patch_size, pad_margin, save_steps=0, **extra_kwargs):
        batch_padded_shape = (1, 1, *padded_shape)
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        padded_volume = sample_fn(model, batch_padded_shape, patch_size, pad_margin, save_steps=save_steps, **extra_kwargs)
        return padded_volume[..., pad_margin[0]:-pad_margin[0],
                                  pad_margin[1]:-pad_margin[1],
                                  pad_margin[2]:-pad_margin[2]]
    
    @torch.no_grad()
    def sample_2d(self, model, padded_shape, patch_size, pad_margin, save_steps=0, **extra_kwargs):
        batch_padded_shape = padded_shape
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        padded_volume = sample_fn(model, batch_padded_shape, patch_size, pad_margin, save_steps=save_steps, **extra_kwargs)
        return padded_volume[..., pad_margin[0]:-pad_margin[0],
                                  pad_margin[1]:-pad_margin[1]]

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    @autocast('cuda', enabled = False)
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, model, x_start, t, model_kwargs=None, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # predict and take gradient step
        model_out = model(x, t, **model_kwargs)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()


class ConditionalDiffusion(GaussianDiffusion):
    """
    Add conditional sampling to GaussianDiffusion
    """
    def sample_mcem_ddim_patch(
            self,
            model,
            x_T,
            patch_size,
            pad_margin,
            scale,
            em_update_fn,
            n_subsets,
            cond_schedule: List | None,
            clip_denoised,
            rederive_pred_noise=True,
            batch_size=4,
            save_steps=0,
        ):
        """
        Args:
            x_T:           3D initial input, shape [Z, Y, X], scaled to [-1,1]
            em_update_fn:  measurement update
            batch_size:    batchsize for mc-em calculation
            cond_schedule: List[int], number of measurement steps at each time step
            rederive_pred_noise: pred_noise only
            batch_size:    batch size of patches
        """
        total_timesteps, sampling_timesteps= self.num_timesteps, self.sampling_timesteps
        device = next(model.parameters()).device

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        
        cond_schedule = len(time_pairs) * [1] if cond_schedule is None else cond_schedule
        assert len(cond_schedule) == len(time_pairs), "Condition update schedule should be a list with the same length of time pairs"

        img = x_T # img is [-1,1]
        imgs, x0s = [], []
        count_em = 0
        for i, (time, time_next) in enumerate(tqdm(time_pairs, desc = 'sampling loop time step')):
            # 1: batchify the volume
            xt_ds = ListPaddedPatchDataset3D(
                padded=img.detach(),
                patch_size=patch_size,
                pad_margin=pad_margin
            )

            # 2: get unconditional x_0_hat
            with torch.no_grad():
                pred_noise, x_start = self.model_predictions_3d(
                    model=model,
                    x=img,
                    t=time,
                    patch_size=None,
                    pad_margin=None,
                    clip_denoised=clip_denoised,
                    rederive_pred_noise=rederive_pred_noise,
                    patches_ds=xt_ds,
                    batch_size=len(xt_ds),
                )

            # 3: Get EM update vector
            x_start_scaled = unnormalize_to_zero_to_one(x_start.cpu().numpy().squeeze()) * scale
            x_start_scaled_0 = np.copy(x_start_scaled)
            
            for _ in range(cond_schedule[-1-i]):
                subset_id = count_em % n_subsets
                count_em += 1
                x_start_scaled = em_update_fn(x_start_scaled, subset_id)

            increm = x_start_scaled - x_start_scaled_0
            increm = increm / scale * 2 # 2 for [-1,1] normalization
            increm = torch.from_numpy(increm.astype(np.float32)).to(device)[None, None, ...]
            
            # 4: Manifold constraint
            increm_ds = xt_ds.clone_with_image(increm) 
            assert increm_ds.coords[0] == xt_ds.coords[0], "Increment patches not aligned with image patches"
            
            dataloader_kwargs = {"sampler": None, "shuffle": False, "num_workers": 0, "pin_memory": False}
            increm_loader = DataLoader(
                increm_ds,
                batch_size=batch_size,
                **dataloader_kwargs,
            )
            xt_loader = DataLoader(
                xt_ds,
                batch_size=batch_size,
                **dataloader_kwargs,
            )

            increm_mcem_batches = []
            for (x_bat, pos_bat), (increm_bat, _) in zip(xt_loader, increm_loader):
                batch_size = x_bat.shape[0]
                model_kwargs = {
                    "x_pos": pos_bat.detach_().requires_grad_(False),
                }
                x_bat.detach_().requires_grad_(True)
                batched_times = torch.full((batch_size,), time, device=device, dtype=torch.long)
                
                _, x_start_bat = self.model_predictions(
                    model=model, 
                    x=x_bat, 
                    t=batched_times, 
                    model_kwargs=model_kwargs, 
                    clip_x_start=clip_denoised, 
                    rederive_pred_noise=rederive_pred_noise
                )

                increm_mcem_batches.append(
                    torch.autograd.grad(
                        outputs=x_start_bat,
                        inputs=x_bat,
                        grad_outputs=increm_bat)[0]
                )

            alpha = self.alphas_cumprod[time]
            increm_mcem_batches = torch.cat(increm_mcem_batches, dim=0) * torch.sqrt(alpha) # scaling factor \sqrt{\bar{alpha}} in the Jacobian
            increm_mcem_batches = increm_mcem_batches.clamp(min=-5., max=5.)
            increm_mcem_vol = increm_ds.reconstruct_volume(increm_mcem_batches, margin_val=0.)

            if time_next < 0:
                img = x_start + increm_mcem_vol
                if save_steps > 0:
                    imgs.append(img.detach().cpu().numpy().squeeze())
                    # x0s.append(x_start.cpu().numpy().squeeze())
                continue
            
            # 5: DDIM
            alpha_next = self.alphas_cumprod[time_next]
            eta = self.eta

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x_start)

            img = (x_start + increm_mcem_vol) * alpha_next.sqrt() + \
                    c * pred_noise + \
                    sigma * noise
            
            if save_steps > 0 and i % save_steps == 0:
                imgs.append(img.cpu().numpy().squeeze())
                # x0s.append(x_start.cpu().numpy().squeeze())
            
        if save_steps == 0:
            ret = img.cpu().numpy().squeeze()
        else:
            ret = np.stack(imgs, axis=0) 
            # x0s = np.stack(x0s, axis=0)

        ret = unnormalize_to_zero_to_one(ret) * scale
        ret = ret[..., 
                  pad_margin[0]:-pad_margin[0],
                  pad_margin[1]:-pad_margin[1],
                  pad_margin[2]:-pad_margin[2]]
        # x0s = x0s[..., 
        #           pad_margin[0]:-pad_margin[0],
        #           pad_margin[1]:-pad_margin[1],
        #           pad_margin[2]:-pad_margin[2]]
        return ret
    

    def sample_pet_dds_patch(
            self,
            model,
            x_T,
            patch_size,
            pad_margin,
            scale,
            em_update_fn,
            n_subsets,
            cond_schedule: List | None,
            clip_denoised,
            rederive_pred_noise=True,
            batch_size=4,
            save_steps=0,
    ):
        """
            x_T: 3D initial input, shape [Z, Y, X], scaled to [-1,1]
            batch_size: batchsize for mc-em calculation
        """
        total_timesteps, sampling_timesteps= self.num_timesteps, self.sampling_timesteps
        device = next(model.parameters()).device

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        
        cond_schedule = len(time_pairs) * [1] if cond_schedule is None else cond_schedule
        assert len(cond_schedule) == len(time_pairs), "Condition update schedule should be a list with the same length of time pairs"

        img = x_T # img is [-1,1]
        imgs = []
        count_em = 0
        for i, (time, time_next) in enumerate(tqdm(time_pairs, desc = 'sampling loop time step')):
            # 1: batchify the volume
            xt_ds = ListPaddedPatchDataset3D(
                padded=img.detach(),
                patch_size=patch_size,
                pad_margin=pad_margin
            )

            # 2: get unconditional x_0_hat
            with torch.no_grad():
                pred_noise, x_start = self.model_predictions_3d(
                    model=model,
                    x=img,
                    t=time,
                    patch_size=None,
                    pad_margin=None,
                    clip_denoised=clip_denoised,
                    rederive_pred_noise=rederive_pred_noise,
                    patches_ds=xt_ds,
                    batch_size=len(xt_ds),
                )

            # 3: Get Proximal EM update 
            x_start_scaled = unnormalize_to_zero_to_one(x_start.cpu().numpy().squeeze()) * scale
            x_start_scaled_0 = np.copy(x_start_scaled)
            
            for _ in range(cond_schedule[-1-i]):
                subset_id = count_em % n_subsets
                count_em += 1
                x_start_scaled = em_update_fn(x_start_scaled, subset_id, x0=x_start_scaled_0)

            increm = x_start_scaled - x_start_scaled_0
            increm = increm / scale * 2 # 2 for [-1,1] normalization
            increm = torch.from_numpy(increm.astype(np.float32)).to(device)[None, None, ...]
            
            if time_next < 0:
                img = x_start + increm
                if save_steps > 0:
                    imgs.append(img.detach().cpu().numpy().squeeze())
                continue
            
            # 4: DDIM
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            eta = self.eta

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x_start)

            img = (x_start + increm) * alpha_next.sqrt() + \
                    c * pred_noise + \
                    sigma * noise
            
            if save_steps > 0 and i % save_steps == 0:
                imgs.append(img.cpu().numpy().squeeze())
            
        if save_steps == 0:
            ret = img.cpu().numpy().squeeze()
        else:
            ret = np.stack(imgs, axis=0) 

        ret = unnormalize_to_zero_to_one(ret) * scale
        ret = ret[..., 
                  pad_margin[0]:-pad_margin[0],
                  pad_margin[1]:-pad_margin[1],
                  pad_margin[2]:-pad_margin[2]]
        return ret
    

    def sample_dps_patch(
            self,
            model,
            x_T,
            patch_size,
            pad_margin,
            scale,
            em_update_fn,
            n_subsets,
            cond_schedule: List | None,
            clip_denoised,
            rederive_pred_noise=True,
            batch_size=3,
            save_steps=0,
        ):
        """
            x_T: 3D initial input, shape [Z, Y, X], scaled to [-1,1]
            batch_size: batchsize for mc-em calculation
        """
        total_timesteps, sampling_timesteps= self.num_timesteps, self.sampling_timesteps
        device = next(model.parameters()).device

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        
        cond_schedule = len(time_pairs) * [1] if cond_schedule is None else cond_schedule
        assert len(cond_schedule) == len(time_pairs), "Condition update schedule should be a list with the same length of time pairs"

        img = x_T # img is [-1,1]
        imgs, x0s = [], []
        count_em = 0
        for i, (time, time_next) in enumerate(tqdm(time_pairs, desc = 'sampling loop time step')):
            # 1: batchify the volume
            xt_ds = ListPaddedPatchDataset3D(
                padded=img.detach(),
                patch_size=patch_size,
                pad_margin=pad_margin
            )

            # 2: get unconditional x_0_hat
            with torch.no_grad():
                pred_noise, x_start = self.model_predictions_3d(
                    model=model,
                    x=img,
                    t=time,
                    patch_size=None,
                    pad_margin=None,
                    clip_denoised=False,
                    rederive_pred_noise=rederive_pred_noise,
                    patches_ds=xt_ds,
                    batch_size=len(xt_ds),
                )
            if clip_denoised:
                x_start.clamp_(self.clip_x0_lower, self.clip_x0_upper)

            # 3: Get EM update vector
            x_start_scaled = unnormalize_to_zero_to_one(x_start.cpu().numpy().squeeze()) * scale
            
            subset_id = count_em % n_subsets
            count_em += 1
            grad = em_update_fn(x_start_scaled, subset_id)

            increm = grad
            increm = increm / scale * 2 # 2 for [-1,1] normalization
            increm = torch.from_numpy(increm.astype(np.float32)).to(device)[None, None, ...]
            
            # 4: Manifold constraint
            increm_ds = xt_ds.clone_with_image(increm) 
            assert increm_ds.coords[0] == xt_ds.coords[0], "Increment patches not aligned with image patches"
            
            dataloader_kwargs = {"sampler": None, "shuffle": False, "num_workers": 0, "pin_memory": False}
            increm_loader = DataLoader(
                increm_ds,
                batch_size=batch_size,
                **dataloader_kwargs,
            )
            xt_loader = DataLoader(
                xt_ds,
                batch_size=batch_size,
                **dataloader_kwargs,
            )

            increm_mcem_batches = []
            for (x_bat, pos_bat), (increm_bat, _) in zip(xt_loader, increm_loader):
                batch_size = x_bat.shape[0]
                model_kwargs = {
                    "x_pos": pos_bat.detach_().requires_grad_(False),
                }
                x_bat.detach_().requires_grad_(True)
                batched_times = torch.full((batch_size,), time, device=device, dtype=torch.long)
                
                _, x_start_bat = self.model_predictions(
                    model=model, 
                    x=x_bat, 
                    t=batched_times, 
                    model_kwargs=model_kwargs, 
                    clip_x_start=clip_denoised, 
                    rederive_pred_noise=rederive_pred_noise
                )

                increm_mcem_batches.append(
                    torch.autograd.grad(
                        outputs=x_start_bat,
                        inputs=x_bat,
                        grad_outputs=increm_bat)[0]
                )

            increm_mcem_batches = torch.cat(increm_mcem_batches, dim=0).clamp(min=-5., max=5.)
            increm_mcem_vol = increm_ds.reconstruct_volume(increm_mcem_batches, margin_val=0.)

            # 5: Sampling
            model_mean, _, posterior_log_variance = self.q_posterior(
                x_start=x_start, x_t=img, 
                t=torch.full((1,), time, device=device, dtype=torch.long)
            )

            noise = torch.randn_like(img) if time > 0 else 0. # no noise if t == 0
            img = model_mean + (0.5 * posterior_log_variance).exp() * noise
            img += increm_mcem_vol
            
            if save_steps > 0 and i % save_steps == 0:
                imgs.append(img.cpu().numpy().squeeze())
            
        if save_steps == 0:
            ret = img.cpu().numpy().squeeze()
        else:
            ret = np.stack(imgs, axis=0) 

        ret = unnormalize_to_zero_to_one(ret) * scale
        ret = ret[..., 
                  pad_margin[0]:-pad_margin[0],
                  pad_margin[1]:-pad_margin[1],
                  pad_margin[2]:-pad_margin[2]]
        return ret
    