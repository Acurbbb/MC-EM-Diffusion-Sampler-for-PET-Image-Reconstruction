conda activate guided-diff

nohup python ./train_2d.py \
    --base ./configs/train_2d.yaml \
    --output_dir ./results/temp \
    --lr 1e-4 \
    --total_steps 100001 \
    --save_step 10000 \
    --accumulate_grad_batches 1 \
    --ema_decay 0.999 \
    --use_fp16 \
    > output_2d.log 2>&1 &