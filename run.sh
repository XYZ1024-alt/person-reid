#!/usr/bin/env bash
set -euo pipefail

# Foundation Model direct-transfer training pipeline.
# Stage 1: Market-1501 pretraining.
# Stage 3: PRCC fine-tuning from the Stage 1 checkpoint.

GPUS="${GPUS:-1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-12}"
START_STAGE="${START_STAGE:-1}"
STOP_STAGE="${STOP_STAGE:-3}"

if [[ "$GPUS" -gt 1 ]]; then
  PYTHON="torchrun --nproc_per_node=${GPUS}"
  DISTRIBUTED_ARG="--distributed"
else
  PYTHON="python"
  DISTRIBUTED_ARG=""
fi

USE_MLFLOW="${USE_MLFLOW:-0}"
MLFLOW_EXPERIMENT="${MLFLOW_EXPERIMENT:-pedestrian_reid_clip}"
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file:./outputs/mlruns}"

BACKBONE="${BACKBONE:-clip_vit_l}"
BACKBONE_LR_STAGE1="${BACKBONE_LR_STAGE1:-1e-5}"
BACKBONE_LR_STAGE3="${BACKBONE_LR_STAGE3:-5e-6}"
HEAD_LR_STAGE1="${HEAD_LR_STAGE1:-1e-4}"
HEAD_LR_STAGE3="${HEAD_LR_STAGE3:-5e-5}"
PRECISION="${PRECISION:-fp16}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PYTHONPATH:-}:${SCRIPT_DIR}"
echo "PYTHONPATH=${PYTHONPATH}"

if [[ -z "${OMP_NUM_THREADS:-}" || ! "${OMP_NUM_THREADS}" =~ ^[0-9]+$ || "${OMP_NUM_THREADS}" -lt 1 ]]; then
  export OMP_NUM_THREADS=1
  echo "OMP_NUM_THREADS=1"
fi

EXP_ROOT="${EXP_ROOT:-outputs/transfer}"
CLIP_STAGE1="${EXP_ROOT}/stage1_market_clip"
CLIP_STAGE3="${EXP_ROOT}/stage3_prcc_direct"

if [[ "$USE_MLFLOW" == "1" ]]; then
  MLFLOW_ARGS="--use-mlflow --mlflow-experiment ${MLFLOW_EXPERIMENT} --mlflow-tracking-uri ${MLFLOW_TRACKING_URI}"
else
  MLFLOW_ARGS=""
fi

if [[ "$START_STAGE" -le 1 && "$STOP_STAGE" -ge 1 ]]; then
  echo "============================================================================"
  echo "Stage 1: Market-1501 pretraining"
  echo "============================================================================"

  $PYTHON scripts/train.py \
    --backbone "${BACKBONE}" \
    --backbone-lr "${BACKBONE_LR_STAGE1}" \
    --head-lr "${HEAD_LR_STAGE1}" \
    --mode market \
    --epochs 60 \
    --batch-size "${BATCH_SIZE}" \
    --lr-scheduler step \
    --lr-milestones 40,50 \
    --lr-gamma 0.1 \
    --market-root Market-1501 \
    --no-use-part-branch \
    --cal-weight 0 \
    --triplet-weight 1.0 \
    --triplet-margin 0.3 \
    --weight-decay 0.01 \
    --freeze-backbone-epochs 0 \
    --pretrained-checkpoint "" \
    --teacher-checkpoint "" \
    --distill-weight 0 \
    --best-metric mAP \
    --best-dataset market \
    --eval-period 2 \
    --color-jitter-probability 0.5 \
    --random-grayscale-probability 0.1 \
    --num-workers "${NUM_WORKERS}" \
    --precision "${PRECISION}" \
    --output-dir "${CLIP_STAGE1}" \
    ${DISTRIBUTED_ARG} \
    ${MLFLOW_ARGS}

  echo "Stage 1 complete. Evaluating Market-1501..."
  $PYTHON scripts/evaluate.py \
    --checkpoint "${CLIP_STAGE1}/best.pth" \
    --dataset market \
    --root Market-1501 \
    --batch-size 128 \
    --num-workers "${NUM_WORKERS}"
fi

if [[ "$START_STAGE" -le 3 && "$STOP_STAGE" -ge 3 ]]; then
  echo "============================================================================"
  echo "Stage 3: PRCC fine-tuning from Stage 1"
  echo "============================================================================"

  $PYTHON scripts/train.py \
    --backbone "${BACKBONE}" \
    --backbone-lr "${BACKBONE_LR_STAGE3}" \
    --head-lr "${HEAD_LR_STAGE3}" \
    --mode prcc \
    --epochs 30 \
    --batch-size "${BATCH_SIZE}" \
    --lr-scheduler cosine \
    --prcc-root prcc \
    --use-sketch-fusion \
    --cal-weight 0.03 \
    --cal-warmup-epochs 10 \
    --cal-sigmoid-ramp \
    --use-prcc-sketch \
    --rgb-sketch-consistency-weight 0.1 \
    --sketch-warmup-epochs 0 \
    --sketch-ramp-epochs 5 \
    --cross-clothes-contrastive-weight 0.5 \
    --contrastive-temperature 0.10 \
    --cross-clothes-hard-negative-weight 2.5 \
    --no-use-part-branch \
    --triplet-weight 1.0 \
    --triplet-margin 0.3 \
    --weight-decay 0.01 \
    --pretrained-checkpoint "${CLIP_STAGE1}/best.pth" \
    --teacher-checkpoint "" \
    --distill-weight 0 \
    --best-metric mAP \
    --best-dataset prcc \
    --eval-period 2 \
    --color-jitter-probability 0.5 \
    --random-grayscale-probability 0.2 \
    --num-workers "${NUM_WORKERS}" \
    --precision "${PRECISION}" \
    --output-dir "${CLIP_STAGE3}" \
    ${DISTRIBUTED_ARG} \
    ${MLFLOW_ARGS}

  echo "Stage 3 complete. Evaluating PRCC..."
  $PYTHON scripts/evaluate.py \
    --checkpoint "${CLIP_STAGE3}/best.pth" \
    --dataset prcc \
    --root prcc \
    --batch-size 128 \
    --num-workers "${NUM_WORKERS}"

  echo "============================================================================"
  echo "Training complete"
  echo "Final model: ${CLIP_STAGE3}/best.pth"
  echo "============================================================================"
fi
