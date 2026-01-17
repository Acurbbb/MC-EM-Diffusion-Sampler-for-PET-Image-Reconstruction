# !/usr/bin/env bash

conda activate guided-diff

nohup python ./demo_mc-em_patch.py \
    --base ./configs/recon_mc-em_patch.yaml \
    --output_dir ./results/temp \
    --seed 0 \
    --condition.cond_step_schedule.subiter 1 \
    --diffusion.params.sampling_timesteps 200 \
    > output_recon.log 2>&1 &

