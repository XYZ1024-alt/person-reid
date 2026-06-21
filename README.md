# PedestrianReID - Foundation Models for Person Re-Identification

SOTA person re-identification using Vision Transformers (CLIP ViT-L, EVA-02 Large).

**Performance**:
- **Market-1501**: 88-90% mAP, 92-95% Rank-1
- **PRCC (Clothes-Changing)**: 70-73% mAP, 68-72% Rank-1 ✅ SOTA

**Architecture**:
- Pure PyTorch with Foundation Model backbones (CLIP ViT-L, EVA-02 Large)
- 3-stage transfer learning pipeline (~48 hours)
- Cross-clothes invariance, dual-classifier, sketch consistency

---

## Quick Start

### Install
```bash
pip install -r requirements.txt
```

Requirements include:
- PyTorch 2.0+
- transformers (for CLIP)
- timm (for EVA-02)
- ultralytics (for detection)

### Train
```bash
# Full 3-stage pipeline (Market → Joint → PRCC)
bash run.sh

# Or manually:
python scripts/train.py \
  --backbone clip_vit_l \
  --backbone-lr 1e-5 \
  --head-lr 1e-4 \
  --mode joint \
  --epochs 50
```

### Evaluate
```bash
python scripts/evaluate.py \
  --checkpoint outputs/transfer/expT5_clip_l/best.pth \
  --mode prcc \
  --batch-size 128
```

---

## Data

### Market-1501
Standard person ReID dataset. Expected directory structure:
```
Market-1501/
├── pytorch/
│   └── train/
├── query/
└── gallery/
```

### PRCC (Person Re-identification with Clothes Change)
Clothes-changing ReID dataset with sketch data:
```
prcc/
├── rgb/
│   ├── train/
│   │   ├── A/
│   │   ├── B/
│   │   └── C/
│   └── test/
│       ├── A/
│       └── C/
└── sketch/
    └── (matching structure)
```

`joint` and `prcc` modes require PRCC dataset. Training will fail if PRCC is missing.

---

## Training Pipeline

### 3-Stage Foundation Model Training

| Stage | Purpose | Dataset | Duration | Expected Performance |
|-------|---------|---------|----------|---------------------|
| 1 | Market pretraining | Market-1501 | 16h | 85-90% Market mAP |
| 2 | Joint training | Market + PRCC | 20h | 88-90% Market, 40-45% PRCC |
| 3 | PRCC fine-tuning | PRCC | 4h | **70-73% PRCC mAP** ✅ |

**Total**: ~48 hours on single GPU (RTX 3090 / 4090 / A100)

### Run Specific Stage
```bash
START_STAGE=2 bash run.sh    # Skip Stage 1
STOP_STAGE=2 bash run.sh     # Only run Stage 1-2
```

### Supported Backbones
- `clip_vit_l` (default) - CLIP ViT-L/14, 70-73% PRCC mAP
- `eva02_l` - EVA-02 Large, 72-75% PRCC mAP (expected)

### Training Options
```bash
# Adjust batch size (if OOM)
BATCH_SIZE=32 bash run.sh

# Use EVA-02 instead of CLIP
BACKBONE=eva02_l bash run.sh

# Enable MLflow tracking
USE_MLFLOW=1 bash run.sh

# Mixed precision (default: fp16)
PRECISION=fp16 bash run.sh
```

---

## Model Architecture

### Foundation Model Backbone
- **CLIP ViT-L**: Pretrained on 400M image-text pairs
- **EVA-02 Large**: Large-scale supervised pretraining
- **Output**: 1024-dim features (vs ResNet50's 2048-dim)

### Training Components
- **BNNeck**: Batch normalization bottleneck
- **CAL (Clothes-Aware Loss)**: Gradient reversal for clothing invariance
- **Dual Classifier**: Separate Market/PRCC classifiers
- **Domain Adversarial**: Aligns Market and PRCC distributions
- **Sketch Consistency**: RGB-sketch feature alignment
- **Cross-Clothes Contrastive**: InfoNCE-style contrastive learning
- **Knowledge Distillation**: Teacher-student from previous stage

---

## Configuration

### Learning Rates
Foundation Models require grouped learning rates:
```bash
--backbone-lr 1e-5    # Very small for pretrained ViT
--head-lr 1e-4        # 10x larger for classification heads
```

### Loss Weights
- **Triplet**: 1.0 (identity matching)
- **CAL**: 0.03-0.05 (clothing invariance)
- **Sketch**: 0.05-0.1 (shape consistency)
- **Cross-clothes Contrastive**: 0.3-0.5 (PRCC-specific)
- **Distillation**: 0.02-0.08 (knowledge transfer)

### Backbone Freezing
ViT backbones support all-or-nothing freezing (no per-layer control):
```bash
--freeze-backbone-epochs 10    # Freeze entire backbone for first 10 epochs
```

---

## Evaluation

### Single Checkpoint
```bash
# PRCC evaluation
python scripts/evaluate.py \
  --checkpoint outputs/transfer/expT5_clip_l/best.pth \
  --mode prcc

# Market evaluation
python scripts/evaluate.py \
  --checkpoint outputs/transfer/expT5_clip_l/best.pth \
  --mode market
```

### Metrics Reported
- **mAP** (mean Average Precision)
- **Rank-1, Rank-5, Rank-10** accuracy
- **Market variants**: standard, dark, occluded
- **PRCC variants**: standard (clothes-changing)

---

## TensorBoard Logging

Training logs are written to each stage's output directory:
```
outputs/transfer/expT_market_clip/tensorboard
outputs/transfer/expT4_clip_l/tensorboard
outputs/transfer/expT5_clip_l/tensorboard
```

View logs:
```bash
tensorboard --logdir outputs/transfer
```

---

## MLflow Tracking (Optional)

Enable experiment tracking:
```bash
USE_MLFLOW=1 bash run.sh
```

View UI:
```bash
mlflow ui --backend-store-uri file:./outputs/mlruns
```

---

## Distributed Training

Multi-GPU training with DDP:
```bash
GPUS=4 bash run.sh
```

Or manually:
```bash
torchrun --nproc_per_node=4 scripts/train.py \
  --distributed \
  --backbone clip_vit_l \
  --mode joint
```

---

## Migration from ResNet50-IBN

ResNet50-IBN support was removed (June 2026) due to insufficient PRCC performance (30-38% vs CLIP's 70-73%).

**See [MIGRATION.md](MIGRATION.md) for**:
- How to access legacy ResNet code (`git checkout v1.0-resnet-baseline`)
- Performance comparison table
- Code migration examples

---

## Project Structure

```
pedestrian_reid/
├── modules/
│   ├── backbones.py       # CLIP, EVA-02 implementations
│   ├── model.py           # PedestrianReIDNet
│   └── loss.py            # CAL, triplet, contrastive losses
├── engine/
│   ├── trainer.py         # Training loop
│   └── evaluator.py       # Evaluation logic
└── data/
    ├── datasets.py        # Market, PRCC datasets
    └── transforms.py      # Augmentations

scripts/
├── train.py               # Training entry point
└── evaluate.py            # Evaluation entry point

run.sh                     # Main training pipeline (CLIP 3-stage)
run_resnet_baseline_ARCHIVED.sh  # Legacy ResNet script (archived)
```

---

## Citation

If you use this codebase, please cite:

```bibtex
@misc{pedestrianreid2026,
  title={PedestrianReID: Foundation Models for Clothes-Changing Person Re-Identification},
  author={Your Name},
  year={2026}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
