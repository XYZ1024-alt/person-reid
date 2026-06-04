#!/usr/bin/env bash
set -euo pipefail

GPUS="${GPUS:-2}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-12}"
START_STAGE="${START_STAGE:-1}"
TORCHRUN="${TORCHRUN:-torchrun}"
PYTHON="${PYTHON:-python}"

EXP_ROOT="${EXP_ROOT:-outputs/transfer}"
EXP1="${EXP_ROOT}/expT1_market_clean"
EXP2="${EXP_ROOT}/expT2_market_dark"
EXP3="${EXP_ROOT}/expT3_market_occlusion"
EXP4="${EXP_ROOT}/expT4_market_to_joint_prcc"
EXP5="${EXP_ROOT}/expT5_prcc_finetune"

run_stage() {
  local stage="$1"
  shift
  if (( stage < START_STAGE )); then
    echo "skip ExpT${stage}"
    return
  fi
  echo "run ExpT${stage}"
  "$@"
}

evaluate_market() {
  local checkpoint="$1"
  "$PYTHON" -m scripts.evaluate --checkpoint "$checkpoint" --dataset market
}

evaluate_prcc() {
  local checkpoint="$1"
  "$PYTHON" -m scripts.evaluate --checkpoint "$checkpoint" --dataset prcc
}

train_distributed() {
  "$TORCHRUN" --nproc_per_node="$GPUS" -m scripts.train --distributed "$@"
}

run_stage 1 train_distributed \
  --mode market \
  --epochs 120 \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --cal-weight 0 \
  --no-use-prcc-sketch \
  --best-metric mAP \
  --eval-period 10 \
  --color-jitter-probability 0 \
  --random-grayscale-probability 0 \
  --dark-augment-probability 0 \
  --occlusion-augment-probability 0 \
  --output-dir "$EXP1"
run_stage 1 evaluate_market "$EXP1/best.pth"

run_stage 2 train_distributed \
  --mode market \
  --epochs 30 \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --lr 0.0001 \
  --cal-weight 0 \
  --no-use-prcc-sketch \
  --best-metric mAP \
  --eval-period 10 \
  --color-jitter-probability 0.1 \
  --random-grayscale-probability 0 \
  --dark-augment-probability 0.15 \
  --occlusion-augment-probability 0 \
  --pretrained-checkpoint "$EXP1/best.pth" \
  --output-dir "$EXP2"
run_stage 2 evaluate_market "$EXP2/best.pth"

run_stage 3 train_distributed \
  --mode market \
  --epochs 30 \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --lr 0.0001 \
  --cal-weight 0 \
  --no-use-prcc-sketch \
  --best-metric mAP \
  --eval-period 10 \
  --color-jitter-probability 0.1 \
  --random-grayscale-probability 0 \
  --dark-augment-probability 0 \
  --occlusion-augment-probability 0.2 \
  --pretrained-checkpoint "$EXP2/best.pth" \
  --output-dir "$EXP3"
run_stage 3 evaluate_market "$EXP3/best.pth"

run_stage 4 train_distributed \
  --mode joint \
  --epochs 80 \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --lr 0.0001 \
  --cal-weight 0.03 \
  --cal-warmup-epochs 20 \
  --cal-ramp-epochs 20 \
  --sketch-loss-weight 0 \
  --rgb-sketch-consistency-weight 0.02 \
  --sketch-warmup-epochs 10 \
  --sketch-ramp-epochs 10 \
  --prcc-identities-ratio 0.5 \
  --best-metric mAP \
  --eval-period 10 \
  --freeze-backbone-epochs 10 \
  --freeze-backbone-layers stem,layer1,layer2 \
  --color-jitter-probability 0.5 \
  --random-grayscale-probability 0.2 \
  --dark-augment-probability 0.05 \
  --occlusion-augment-probability 0.1 \
  --pretrained-checkpoint "$EXP3/best.pth" \
  --output-dir "$EXP4"
run_stage 4 evaluate_market "$EXP4/best.pth"
run_stage 4 evaluate_prcc "$EXP4/best.pth"

run_stage 5 train_distributed \
  --mode prcc \
  --epochs 50 \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --lr 0.0001 \
  --cal-weight 0.03 \
  --cal-warmup-epochs 10 \
  --cal-ramp-epochs 10 \
  --sketch-loss-weight 0 \
  --rgb-sketch-consistency-weight 0.02 \
  --sketch-warmup-epochs 5 \
  --sketch-ramp-epochs 10 \
  --best-metric mAP \
  --eval-period 10 \
  --color-jitter-probability 0.5 \
  --random-grayscale-probability 0.25 \
  --dark-augment-probability 0.05 \
  --occlusion-augment-probability 0.1 \
  --pretrained-checkpoint "$EXP4/best.pth" \
  --output-dir "$EXP5"
run_stage 5 evaluate_prcc "$EXP5/best.pth"
