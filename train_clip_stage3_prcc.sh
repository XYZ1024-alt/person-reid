#!/bin/bash
# 阶段3：CLIP PRCC微调

python scripts/train.py \
    --backbone clip_vit_l \
    --backbone-lr 5e-6 \
    --head-lr 5e-5 \
    --mode prcc \
    --epochs 12 \
    --batch-size 64 \
    --lr-scheduler cosine \
    --prcc-root prcc \
    --cal-weight 0.03 \
    --cal-warmup-epochs 1 \
    --use-prcc-sketch \
    --rgb-sketch-consistency-weight 0.1 \
    --sketch-warmup-epochs 0 \
    --sketch-ramp-epochs 3 \
    --cross-clothes-contrastive-weight 0.5 \
    --contrastive-temperature 0.10 \
    --use-part-branch false \
    --triplet-weight 1.0 \
    --triplet-margin 0.3 \
    --weight-decay 0.01 \
    --freeze-backbone-layers "" \
    --pretrained-checkpoint outputs/transfer/expT4_clip_l/best.pth \
    --teacher-checkpoint outputs/transfer/expT4_clip_l/best.pth \
    --distill-weight 0.02 \
    --best-metric mAP \
    --best-dataset prcc \
    --eval-period 1 \
    --color-jitter-probability 0.5 \
    --random-grayscale-probability 0.2 \
    --output-dir outputs/transfer/expT5_clip_l
