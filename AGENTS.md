# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

**PedestrianReID** is a person re-identification system using Vision Transformer foundation models (CLIP ViT-L, EVA-02 Large). Achieves SOTA performance on clothes-changing scenarios (PRCC: 70-73% mAP).

**Core Architecture:**
- Pure PyTorch with Foundation Model backbones (not ResNet50-IBN, removed June 2026)
- 3-stage transfer learning: Market pretraining → Joint training → PRCC fine-tuning
- Training components: BNNeck, CAL (gradient reversal), dual classifiers, domain adversarial, sketch consistency, cross-clothes contrastive, knowledge distillation

## Commands

### Training
```bash
# Full 3-stage pipeline (~48 hours)
bash run.sh

# Run specific stages
START_STAGE=2 bash run.sh      # Skip Stage 1
STOP_STAGE=2 bash run.sh       # Only Stage 1-2

# Manual training (single stage)
python scripts/train.py \
  --backbone clip_vit_l \
  --backbone-lr 1e-5 \
  --head-lr 1e-4 \
  --mode joint \
  --epochs 50
```

### Evaluation
```bash
# Evaluate checkpoint
python scripts/evaluate.py \
  --checkpoint outputs/transfer/expT5_clip_l/best.pth \
  --mode prcc \
  --batch-size 128

# Supported modes: market, prcc
```

### Environment
```bash
pip install -r requirements.txt

# Key dependencies:
# - torch>=2.0.0
# - transformers>=4.30.0 (CLIP)
# - timm>=0.9.0 (EVA-02)
# - ultralytics (detection)
```

## Code Architecture

### Module Hierarchy

**`pedestrian_reid/modules/`** - Core neural network components
- `backbones.py`: Foundation Model implementations (`CLIPViTBackbone`, `EVA02LBackbone`)
  - Abstract interface: `BaseBackbone` with `output_format()` method
  - **Critical**: Backbones have different output formats:
    - ViT: `'sequence'` → [B, D] CLS token (no spatial pooling needed)
    - CNN: `'spatial'` → [B, C, H, W] feature maps (requires AdaptiveAvgPool2d)
  - **Input resize**: CLIP requires 224×224, EVA-02 requires 448×448 (handled internally)
- `model.py`: Main ReID model (`PedestrianReIDNet`)
  - Dynamic backbone creation via `create_backbone(backbone_type)`
  - Conditional pooling based on `backbone.output_format()`
  - Part branch only works with CNN backbones (disabled for ViT)
  - Dual classifier support (separate Market/PRCC heads)
- `losses.py`: Triplet loss (`batch_hard_triplet_loss`)
- `metrics.py`: Evaluation metrics (mAP, Rank-N)

**`pedestrian_reid/data/`** - Dataset and transforms
- `datasets.py`: Market-1501, PRCC loaders
  - `ReidSample` dataclass: unified sample interface
  - Sketch support for PRCC (RGB-sketch pairs)
- `transforms.py`: Augmentations (flip, color jitter, grayscale)
- `samplers.py`: Identity-balanced sampling for training

**`pedestrian_reid/engine/`** - Training/evaluation loops
- `trainer.py`: Multi-stage training logic
  - Handles loss component weighting (CAL, sketch, distillation)
  - Progressive weight ramping (CAL warmup, PRCC CE ramping)
  - Teacher-student distillation from previous stage
- `evaluator.py`: CMC + mAP evaluation

**`pedestrian_reid/config/`** - Configuration management
- Config dataclasses for training hyperparameters

**`scripts/`** - Entry points
- `train.py`: Training CLI
- `evaluate.py`: Evaluation CLI
- `extract.py`: Feature extraction

### Training Pipeline Architecture

**3-Stage Transfer Learning:**
1. **Stage 1** (Market pretraining): Market-1501 only, 60 epochs, freeze backbone 10 epochs
2. **Stage 2** (Joint training): Market + PRCC, 50 epochs, dual classifiers, CAL/sketch/contrastive losses
3. **Stage 3** (PRCC fine-tuning): PRCC only, 12 epochs, aggressive contrastive learning

**Key Training Mechanisms:**
- **Backbone freezing**: `--freeze-backbone-epochs N` freezes entire ViT (all-or-nothing, no per-layer control)
- **Grouped learning rates**: `--backbone-lr 1e-5 --head-lr 1e-4` (ViT needs 10× smaller LR than heads)
- **Loss ramping**: CAL/sketch/distillation weights ramp up over N epochs from 0 to target
- **Knowledge distillation**: Teacher model from previous stage guides current stage
- **PRCC dev split**: `--prcc-dev-identities N` reserves N identities for validation (stage 2 uses this for best model selection)

## Important Implementation Details

### Backbone Format Handling
When adding new backbones or modifying `PedestrianReIDNet.forward()`:
- **Always check** `self.backbone.output_format()` before pooling
- ViT backbones return CLS token directly → use `nn.Identity()` pooling
- CNN backbones return spatial maps → use `nn.AdaptiveAvgPool2d(1)`
- Part branch (`PartFeatureBranch`) only compatible with spatial format

Example from `model.py:125-134`:
```python
if self.backbone.output_format() == 'spatial':
    pooled = self.pool(feature_map).flatten(1)  # CNN path
else:
    pooled = feature_map  # ViT path (already CLS token)
```

### Dual Classifier System
When `--use-dual-classifier` is enabled:
- `classifier`: Market-1501 identity classifier
- `prcc_classifier`: PRCC identity classifier
- Separate class counts: `num_market_classes`, `num_prcc_classes`
- Forward pass returns both `logits` (Market) and `prcc_logits`

### Loss Component Weights
Stage-dependent defaults (see `run.sh`):
- **Triplet**: Always 1.0
- **CAL**: 0 (Stage 1) → 0.05 (Stage 2) → 0.03 (Stage 3)
- **Sketch**: 0.05 (Stage 2) → 0.1 (Stage 3)
- **Cross-clothes contrastive**: 0.3 (Stage 2) → 0.5 (Stage 3)
- **Distillation**: 0.05→0.02 (Stage 2) → 0.02 (Stage 3)

### Dataset Structure
Expected directory layout:
```
Market-1501/
├── pytorch/train/
├── query/
└── gallery/

prcc/
├── rgb/
│   ├── train/{A,B,C}/
│   └── test/{A,C}/
└── sketch/ (same structure)
```

## Testing Strategy

**No pytest framework** - use manual verification:
```bash
# Test model forward pass
python scripts/train.py --epochs 1 --batch-size 8 --mode market

# Test evaluation
python scripts/evaluate.py --checkpoint <path> --mode market
```

**TensorBoard monitoring:**
```bash
tensorboard --logdir outputs/transfer
```

## Common Pitfalls

1. **ResNet50-IBN references**: Removed in June 2026. Use `clip_vit_l` or `eva02_l` only.
2. **Part branch with ViT**: Incompatible. Part branch requires spatial feature maps (CNNs only).
3. **Backbone LR too high**: ViT needs ~1e-5, not 1e-4. Use `--backbone-lr` and `--head-lr` separately.
4. **Missing PRCC for joint/prcc modes**: Training fails if `prcc/` directory missing.
5. **OOM errors**: Reduce `--batch-size` (64 → 32 for CLIP).
6. **Backbone layer freezing**: ViT supports all-or-nothing freezing via `--freeze-backbone-epochs`, not per-layer via `--freeze-backbone-layers` (that's CNN-specific, ignored for ViT).

## Migration Notes

- **Legacy ResNet50-IBN code**: Access via `git checkout v1.0-resnet-baseline` (archived June 2026)
- **Performance gap**: ResNet50-IBN achieved 30-38% PRCC mAP vs CLIP's 70-73%
- See `MIGRATION.md` for detailed comparison
