#!/usr/bin/env bash
set -euo pipefail

GPUS="${GPUS:-1}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-12}"
START_STAGE="${START_STAGE:-1}"
STOP_STAGE="${STOP_STAGE:-4}"
RUN_EXPT4_NODISTILL="${RUN_EXPT4_NODISTILL:-0}"
RUN_EXPT4_DEV_ABLATIONS="${RUN_EXPT4_DEV_ABLATIONS:-1}"
PRCC_DEV_IDENTITIES="${PRCC_DEV_IDENTITIES:-30}"
PRCC_DEV_SEED="${PRCC_DEV_SEED:-42}"
TORCHRUN="${TORCHRUN:-torchrun}"
PYTHON="${PYTHON:-python}"

if [[ -z "${OMP_NUM_THREADS:-}" || ! "${OMP_NUM_THREADS}" =~ ^[0-9]+$ || "${OMP_NUM_THREADS}" -lt 1 ]]; then
  echo "set OMP_NUM_THREADS=1 (was '${OMP_NUM_THREADS:-unset}')"
  export OMP_NUM_THREADS=1
fi

EXP_ROOT="${EXP_ROOT:-outputs/transfer}"
EXP1="${EXP_ROOT}/expT1_market_clean"
EXP2="${EXP_ROOT}/expT2_market_dark"
EXP3="${EXP_ROOT}/expT3_market_occlusion"
EXP4="${EXP_ROOT}/expT4_market_to_joint_prcc"
EXP4_NODISTILL="${EXP_ROOT}/expT4_market_to_joint_prcc_nodistill"
EXP4_DEV_CONTROL="${EXP_ROOT}/expT4_dev_control"
EXP4_DEV_FEATURE_MATCH="${EXP_ROOT}/expT4_dev_feature_match"
EXP4_DEV_OBJECTIVE_SHIFT="${EXP_ROOT}/expT4_dev_objective_shift"
if [[ "$RUN_EXPT4_DEV_ABLATIONS" == "1" ]]; then
  DEFAULT_EXP4_FOR_EXP5="$EXP4_DEV_OBJECTIVE_SHIFT"
else
  DEFAULT_EXP4_FOR_EXP5="$EXP4"
fi
EXP4_FOR_EXP5="${EXP4_FOR_EXP5:-$DEFAULT_EXP4_FOR_EXP5}"
EXP5="${EXP_ROOT}/expT5_prcc_finetune"

run_stage() {
  local stage="$1"
  shift
  if (( stage < START_STAGE )); then
    echo "skip ExpT${stage}"
    return
  fi
  if (( stage > STOP_STAGE )); then
    echo "skip ExpT${stage}"
    return
  fi
  echo "run ExpT${stage}"
  "$@"
}

evaluate_market() {
  local checkpoint="$1"
  local feature_key="${2:-bn_features}"
  "$PYTHON" -m scripts.evaluate --checkpoint "$checkpoint" --dataset market --feature-key "$feature_key"
}

evaluate_prcc() {
  local checkpoint="$1"
  local feature_key="${2:-bn_features}"
  "$PYTHON" -m scripts.evaluate --checkpoint "$checkpoint" --dataset prcc --feature-key "$feature_key"
}

evaluate_prcc_dev() {
  local checkpoint="$1"
  local feature_key="${2:-bn_features}"
  "$PYTHON" -m scripts.evaluate \
    --checkpoint "$checkpoint" \
    --dataset prcc_dev \
    --feature-key "$feature_key" \
    --prcc-dev-identities "$PRCC_DEV_IDENTITIES" \
    --prcc-dev-seed "$PRCC_DEV_SEED"
}

train_model() {
  if (( GPUS > 1 )); then
    "$TORCHRUN" --nproc_per_node="$GPUS" -m scripts.train --distributed "$@"
    return
  fi
  "$PYTHON" -m scripts.train "$@"
}

train_expt4() {
  local output_dir="$1"
  local distill_weight="$2"
  local distill_final_weight="$3"
  shift 3
  train_model \
    --mode joint \
    --epochs 40 \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --lr 0.0001 \
    --cal-weight 0 \
    --cal-warmup-epochs 0 \
    --cal-ramp-epochs 0 \
    --sketch-loss-weight 0 \
    --rgb-sketch-consistency-weight 0.02 \
    --sketch-warmup-epochs 5 \
    --sketch-ramp-epochs 10 \
    --prcc-identities-ratio 0.75 \
    --use-part-branch \
    --num-parts 6 \
    --part-embedding-dim 256 \
    --part-triplet-weight 0.3 \
    --cloth-invariant-weight 0.5 \
    --combined-global-weight 0.7 \
    --combined-part-weight 0.3 \
    --teacher-checkpoint "$EXP3/best.pth" \
    --distill-weight "$distill_weight" \
    --distill-final-weight "$distill_final_weight" \
    --distill-hold-epochs 0 \
    --distill-ramp-epochs 3 \
    --feature-key combined_features \
    --best-metric mAP \
    --best-variant standard \
    --eval-period 1 \
    --lr-milestones 20,30 \
    --freeze-backbone-epochs 40 \
    --freeze-backbone-layers stem,layer1,layer2 \
    --color-jitter-probability 0.5 \
    --random-grayscale-probability 0.3 \
    --dark-augment-probability 0.05 \
    --occlusion-augment-probability 0.1 \
    --pretrained-checkpoint "$EXP3/best.pth" \
    "$@" \
    --output-dir "$output_dir"
}

train_expt4_dev_common() {
  local output_dir="$1"
  shift
  train_expt4 "$output_dir" 0.05 0.02 \
    --prcc-dev-identities "$PRCC_DEV_IDENTITIES" \
    --prcc-dev-seed "$PRCC_DEV_SEED" \
    --best-dataset prcc_dev \
    "$@"
}

train_expt4_dev_control() {
  train_expt4_dev_common "$EXP4_DEV_CONTROL"
}

train_expt4_dev_feature_match() {
  train_expt4_dev_common "$EXP4_DEV_FEATURE_MATCH" \
    --triplet-feature-key combined_features
}

train_expt4_dev_objective_shift() {
  train_expt4_dev_common "$EXP4_DEV_OBJECTIVE_SHIFT" \
    --triplet-feature-key combined_features \
    --prcc-ce-weight 0.2 \
    --prcc-ce-final-weight 0 \
    --prcc-ce-ramp-epochs 5 \
    --cross-clothes-contrastive-weight 0.2 \
    --contrastive-temperature 0.07
}

run_stage 1 train_model \
  --mode market \
  --epochs 120 \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --cal-weight 0 \
  --no-use-prcc-sketch \
  --use-part-branch \
  --num-parts 6 \
  --part-embedding-dim 256 \
  --part-triplet-weight 0.3 \
  --combined-global-weight 0.7 \
  --combined-part-weight 0.3 \
  --feature-key combined_features \
  --best-metric mAP \
  --best-variant standard \
  --eval-period 5 \
  --lr-milestones 40,70,100 \
  --color-jitter-probability 0.5 \
  --random-grayscale-probability 0 \
  --dark-augment-probability 0.10 \
  --occlusion-augment-probability 0.10 \
  --output-dir "$EXP1"
run_stage 1 evaluate_market "$EXP1/best.pth" combined_features

run_stage 2 train_model \
  --mode market \
  --epochs 30 \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --lr 0.0001 \
  --cal-weight 0 \
  --no-use-prcc-sketch \
  --use-part-branch \
  --num-parts 6 \
  --part-embedding-dim 256 \
  --part-triplet-weight 0.3 \
  --combined-global-weight 0.7 \
  --combined-part-weight 0.3 \
  --feature-key combined_features \
  --best-metric mAP \
  --best-variant dark \
  --eval-period 10 \
  --lr-milestones 10,20 \
  --color-jitter-probability 0.1 \
  --random-grayscale-probability 0 \
  --dark-augment-probability 0.15 \
  --occlusion-augment-probability 0 \
  --pretrained-checkpoint "$EXP1/best.pth" \
  --output-dir "$EXP2"
run_stage 2 evaluate_market "$EXP2/best.pth" combined_features

run_stage 3 train_model \
  --mode market \
  --epochs 30 \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --lr 0.0001 \
  --cal-weight 0 \
  --no-use-prcc-sketch \
  --use-part-branch \
  --num-parts 6 \
  --part-embedding-dim 256 \
  --part-triplet-weight 0.3 \
  --combined-global-weight 0.7 \
  --combined-part-weight 0.3 \
  --feature-key combined_features \
  --best-metric mAP \
  --best-variant occluded \
  --eval-period 10 \
  --lr-milestones 10,20 \
  --color-jitter-probability 0.1 \
  --random-grayscale-probability 0 \
  --dark-augment-probability 0 \
  --occlusion-augment-probability 0.2 \
  --pretrained-checkpoint "$EXP2/best.pth" \
  --output-dir "$EXP3"
run_stage 3 evaluate_market "$EXP3/best.pth" combined_features

if [[ "$RUN_EXPT4_DEV_ABLATIONS" == "1" ]]; then
  run_stage 4 train_expt4_dev_control
  run_stage 4 evaluate_prcc_dev "$EXP4_DEV_CONTROL/best.pth" combined_features
  run_stage 4 train_expt4_dev_feature_match
  run_stage 4 evaluate_prcc_dev "$EXP4_DEV_FEATURE_MATCH/best.pth" combined_features
  run_stage 4 train_expt4_dev_objective_shift
  run_stage 4 evaluate_prcc_dev "$EXP4_DEV_OBJECTIVE_SHIFT/best.pth" combined_features
else
  run_stage 4 train_expt4 "$EXP4" 0.05 0.02
  run_stage 4 evaluate_market "$EXP4/best.pth" combined_features
  run_stage 4 evaluate_prcc "$EXP4/best.pth" combined_features

  if [[ "$RUN_EXPT4_NODISTILL" == "1" ]]; then
    run_stage 4 train_expt4 "$EXP4_NODISTILL" 0 0
    run_stage 4 evaluate_market "$EXP4_NODISTILL/best.pth" combined_features
    run_stage 4 evaluate_prcc "$EXP4_NODISTILL/best.pth" combined_features
  fi
fi

run_stage 5 train_model \
  --mode prcc \
  --epochs 3 \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --lr 0.00003 \
  --cal-weight 0 \
  --cal-warmup-epochs 0 \
  --cal-ramp-epochs 0 \
  --no-use-prcc-sketch \
  --sketch-loss-weight 0 \
  --rgb-sketch-consistency-weight 0 \
  --sketch-warmup-epochs 0 \
  --sketch-ramp-epochs 0 \
  --use-part-branch \
  --num-parts 6 \
  --part-embedding-dim 256 \
  --part-triplet-weight 0.3 \
  --cloth-invariant-weight 0.1 \
  --combined-global-weight 0.7 \
  --combined-part-weight 0.3 \
  --teacher-checkpoint "$EXP4_FOR_EXP5/best.pth" \
  --distill-weight 0.1 \
  --distill-final-weight 0.1 \
  --distill-hold-epochs 0 \
  --distill-ramp-epochs 0 \
  --freeze-backbone-all-epochs \
  --feature-key combined_features \
  --best-metric mAP \
  --best-variant standard \
  --eval-period 1 \
  --lr-milestones 1,2 \
  --color-jitter-probability 0.5 \
  --random-grayscale-probability 0.25 \
  --dark-augment-probability 0.05 \
  --occlusion-augment-probability 0.1 \
  --pretrained-checkpoint "$EXP4_FOR_EXP5/best.pth" \
  --output-dir "$EXP5"
run_stage 5 evaluate_prcc "$EXP5/best.pth" combined_features
