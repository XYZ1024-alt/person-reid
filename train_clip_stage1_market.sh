#!/bin/bash
# 阶段1：CLIP Market预训练（替代ExpT1-T3）

python scripts/train.py \
    --backbone clip_vit_l \
    --backbone-lr 1e-5 \
    --head-lr 1e-4 \
    --mode market \
    --epochs 60 \
    --batch-size 64 \
    --lr-scheduler step \
    --lr-milestones 40,50 \
    --lr-gamma 0.1 \
    --market-root Market-1501 \
    --use-part-branch false \
    --triplet-weight 1.0 \
    --triplet-margin 0.3 \
    --weight-decay 0.01 \
    --freeze-backbone-epochs 10 \
    --pretrained-checkpoint "" \
    --teacher-checkpoint "" \
    --distill-weight 0 \
    --best-metric mAP \
    --best-dataset market \
    --eval-period 2 \
    --color-jitter-probability 0.5 \
    --random-grayscale-probability 0.1 \
    --output-dir outputs/transfer/expT_market_clip
