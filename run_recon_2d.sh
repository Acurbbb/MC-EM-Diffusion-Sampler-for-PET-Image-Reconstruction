# !/usr/bin/env bash

conda activate guided-diff

nohup python demo_mc-em_2d.py \
  --base configs/recon_mc-em_2d.yaml \
  --output_dir ./results/temp \
  --phantom  ./phantom/bern_putamen/350_020.npz  \
  --seed 0 \
  --diffusion.params.sampling_timesteps 100 \
  --diffusion.params.eta 0.1 \
  --condition.cond_step_schedule.subiter 2 \
  > output.log 2>&1 &

        