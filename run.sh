#!/usr/bin/env bash
set -euo pipefail

# Foundation Model Training Pipeline - CLIP ViT-L / EVA-02 Large

# ============================================================================
# 环境变量配置
# ============================================================================

GPUS="${GPUS:-1}"
BATCH_SIZE="${BATCH_SIZE:-64}"  # CLIP推荐64，如果OOM降至32
NUM_WORKERS="${NUM_WORKERS:-12}"
START_STAGE="${START_STAGE:-1}"
STOP_STAGE="${STOP_STAGE:-3}"

# 根据GPU数量选择训练命令
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

# CLIP特有配置
BACKBONE="${BACKBONE:-clip_vit_l}"  # 选项: clip_vit_l, eva02_l
BACKBONE_LR_STAGE1="${BACKBONE_LR_STAGE1:-1e-5}"
BACKBONE_LR_STAGE2="${BACKBONE_LR_STAGE2:-7.5e-6}"
BACKBONE_LR_STAGE3="${BACKBONE_LR_STAGE3:-5e-6}"
HEAD_LR_STAGE1="${HEAD_LR_STAGE1:-1e-4}"
HEAD_LR_STAGE2="${HEAD_LR_STAGE2:-1e-4}"
HEAD_LR_STAGE3="${HEAD_LR_STAGE3:-5e-5}"
PRECISION="${PRECISION:-fp16}"  # 选项: fp16, fp32

# Set PYTHONPATH to include project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PYTHONPATH:-}:${SCRIPT_DIR}"
echo "PYTHONPATH set to: ${PYTHONPATH}"

if [[ -z "${OMP_NUM_THREADS:-}" || ! "${OMP_NUM_THREADS}" =~ ^[0-9]+$ || "${OMP_NUM_THREADS}" -lt 1 ]]; then
  echo "set OMP_NUM_THREADS=1 (was '${OMP_NUM_THREADS:-unset}')"
  export OMP_NUM_THREADS=1
fi

# ============================================================================
# 输出目录配置
# ============================================================================

EXP_ROOT="${EXP_ROOT:-outputs/transfer}"
CLIP_STAGE1="${EXP_ROOT}/expT_market_clip"
CLIP_STAGE2="${EXP_ROOT}/expT4_clip_l"
CLIP_STAGE3="${EXP_ROOT}/expT5_clip_l"

# ============================================================================
# MLflow配置
# ============================================================================

if [[ "$USE_MLFLOW" == "1" ]]; then
  MLFLOW_ARGS="--use-mlflow --mlflow-experiment ${MLFLOW_EXPERIMENT} --mlflow-tracking-uri ${MLFLOW_TRACKING_URI}"
else
  MLFLOW_ARGS=""
fi

# ============================================================================
# 阶段1：CLIP Market预训练（替代ExpT1-T3）
# ============================================================================

if [[ "$START_STAGE" -le 1 && "$STOP_STAGE" -ge 1 ]]; then
  echo "============================================================================"
  echo "阶段1：CLIP Market预训练"
  echo "============================================================================"

  $PYTHON scripts/train.py \
    --backbone ${BACKBONE} \
    --backbone-lr ${BACKBONE_LR_STAGE1} \
    --head-lr ${HEAD_LR_STAGE1} \
    --mode market \
    --epochs 60 \
    --batch-size ${BATCH_SIZE} \
    --lr-scheduler step \
    --lr-milestones 40,50 \
    --lr-gamma 0.1 \
    --market-root Market-1501 \
    --no-use-part-branch \
    --cal-weight 0 \
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
    --num-workers ${NUM_WORKERS} \
    --precision ${PRECISION} \
    --output-dir ${CLIP_STAGE1} \
    ${DISTRIBUTED_ARG} \
    ${MLFLOW_ARGS}

  echo ""
  echo "阶段1完成，开始评估..."

  $PYTHON scripts/evaluate.py \
    --checkpoint ${CLIP_STAGE1}/best.pth \
    --data-root Market-1501 \
    --mode market \
    --batch-size 128 \
    --num-workers ${NUM_WORKERS}

  echo ""
fi

# ============================================================================
# 阶段2：CLIP Joint训练（Market + PRCC）
# ============================================================================

if [[ "$START_STAGE" -le 2 && "$STOP_STAGE" -ge 2 ]]; then
  echo "============================================================================"
  echo "阶段2：CLIP Joint训练（Market + PRCC）"
  echo "============================================================================"

  $PYTHON scripts/train.py \
    --backbone ${BACKBONE} \
    --backbone-lr ${BACKBONE_LR_STAGE2} \
    --head-lr ${HEAD_LR_STAGE2} \
    --mode joint \
    --epochs 50 \
    --batch-size ${BATCH_SIZE} \
    --lr-scheduler step \
    --lr-milestones 30,45 \
    --lr-gamma 0.1 \
    --market-root Market-1501 \
    --prcc-root prcc \
    --prcc-identities-ratio 0.85 \
    --use-dual-classifier \
    --domain-adversarial-weight 0.2 \
    --cal-weight 0.05 \
    --cal-warmup-epochs 5 \
    --cal-ramp-epochs 20 \
    --prcc-ce-weight 0.5 \
    --prcc-ce-final-weight 2.0 \
    --prcc-ce-ramp-epochs 15 \
    --cross-clothes-contrastive-weight 0.5 \
    --contrastive-temperature 0.10 \
    --cross-clothes-hard-negative-weight 2.5 \
    --cal-sigmoid-ramp \
    --use-prcc-sketch \
    --rgb-sketch-consistency-weight 0.05 \
    --sketch-warmup-epochs 5 \
    --sketch-ramp-epochs 20 \
    --freeze-backbone-epochs 10 \
    --no-use-part-branch \
    --triplet-weight 1.0 \
    --triplet-margin 0.3 \
    --weight-decay 0.01 \
    --pretrained-checkpoint ${CLIP_STAGE1}/best.pth \
    --teacher-checkpoint ${CLIP_STAGE1}/best.pth \
    --distill-weight 0.05 \
    --distill-final-weight 0.02 \
    --distill-ramp-epochs 10 \
    --best-metric mAP \
    --best-dataset prcc_dev \
    --prcc-dev-identities 30 \
    --prcc-dev-seed 42 \
    --eval-period 2 \
    --color-jitter-probability 0.5 \
    --random-grayscale-probability 0.2 \
    --num-workers ${NUM_WORKERS} \
    --precision ${PRECISION} \
    --output-dir ${CLIP_STAGE2} \
    ${DISTRIBUTED_ARG} \
    ${MLFLOW_ARGS}

  echo ""
  echo "阶段2完成，开始评估..."

  # 评估Market
  echo "评估Market性能..."
  $PYTHON scripts/evaluate.py \
    --checkpoint ${CLIP_STAGE2}/best.pth \
    --data-root Market-1501 \
    --mode market \
    --batch-size 128 \
    --num-workers ${NUM_WORKERS}

  # 评估PRCC
  echo ""
  echo "评估PRCC性能..."
  $PYTHON scripts/evaluate.py \
    --checkpoint ${CLIP_STAGE2}/best.pth \
    --data-root prcc \
    --mode prcc \
    --batch-size 128 \
    --num-workers ${NUM_WORKERS}

  echo ""
fi

# ============================================================================
# 阶段3：CLIP PRCC微调
# ============================================================================

if [[ "$START_STAGE" -le 3 && "$STOP_STAGE" -ge 3 ]]; then
  echo "============================================================================"
  echo "阶段3：CLIP PRCC微调"
  echo "============================================================================"

  $PYTHON scripts/train.py \
    --backbone ${BACKBONE} \
    --backbone-lr ${BACKBONE_LR_STAGE3} \
    --head-lr ${HEAD_LR_STAGE3} \
    --mode prcc \
    --epochs 12 \
    --batch-size ${BATCH_SIZE} \
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
    --cross-clothes-hard-negative-weight 2.0 \
    --cal-sigmoid-ramp \
    --no-use-part-branch \
    --triplet-weight 1.0 \
    --triplet-margin 0.3 \
    --weight-decay 0.01 \
    --freeze-backbone-layers "" \
    --pretrained-checkpoint ${CLIP_STAGE2}/best.pth \
    --teacher-checkpoint ${CLIP_STAGE2}/best.pth \
    --distill-weight 0.02 \
    --best-metric mAP \
    --best-dataset prcc \
    --eval-period 1 \
    --color-jitter-probability 0.5 \
    --random-grayscale-probability 0.2 \
    --num-workers ${NUM_WORKERS} \
    --precision ${PRECISION} \
    --output-dir ${CLIP_STAGE3} \
    ${DISTRIBUTED_ARG} \
    ${MLFLOW_ARGS}

  echo ""
  echo "阶段3完成，开始最终评估..."

  $PYTHON scripts/evaluate.py \
    --checkpoint ${CLIP_STAGE3}/best.pth \
    --data-root prcc \
    --mode prcc \
    --batch-size 128 \
    --num-workers ${NUM_WORKERS}

  echo ""
  echo "============================================================================"
  echo "CLIP全程训练完成！"
  echo "============================================================================"
  echo "最终模型位置："
  echo "  ${CLIP_STAGE3}/best.pth"
  echo "============================================================================"
fi

# ============================================================================
# 使用说明
# ============================================================================
#
# 基础用法：
#   bash run_CLIP.sh
#
# 只运行特定阶段：
#   START_STAGE=2 STOP_STAGE=2 bash run_CLIP.sh  # 只运行阶段2
#   START_STAGE=3 bash run_CLIP.sh                # 从阶段3开始
#
# 显存不足时降低batch size：
#   BATCH_SIZE=32 bash run_CLIP.sh
#
# 使用EVA02-L替代CLIP：
#   BACKBONE=eva02_l bash run_CLIP.sh
#
# 启用MLflow追踪：
#   USE_MLFLOW=1 bash run_CLIP.sh
#
# 组合使用：
#   BATCH_SIZE=32 PRECISION=fp16 START_STAGE=1 bash run_CLIP.sh
#
# ============================================================================
