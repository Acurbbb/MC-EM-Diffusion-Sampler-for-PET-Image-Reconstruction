conda activate guided-diff

nohup python ./train_patch.py \
    --base ./configs/train_patch_3d.yaml \
    --output_dir ./results/temp \
    --lr 8e-5 \
    --total_steps 400001 \
    --save_step 10000 \
    --accumulate_grad_batches 4 \
    --ema_decay 0.999 \
    --use_fp16 \
    > output2.log 2>&1 &

