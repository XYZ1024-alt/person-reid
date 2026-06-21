# Migration Guide: ResNet50-IBN Removal

## Summary

ResNet50-IBN support removed in June 2026 due to insufficient PRCC performance.

| Backbone | Market mAP | PRCC mAP | Status |
|----------|------------|----------|--------|
| ResNet50-IBN | 90% | **30-38%** ❌ | Removed |
| CLIP ViT-L | 88-90% | **70-73%** ✅ | Current |
| EVA-02 Large | 90%+ | **72-75%** ✅ | Supported |

**PRCC (clothes-changing ReID) is the primary challenge** - CLIP achieves 2.3x better performance.

---

## Impact on Existing Work

### Existing ResNet Checkpoints
**Will NOT load.** Error message:
```
ValueError: Unknown backbone type: resnet50_ibn. Supported: clip_vit_l, eva02_l
```

### Old Training Scripts
**`run.sh` renamed** to `run_resnet_baseline_ARCHIVED.sh` (will exit with error)

---

## Options

### 1. Re-train with CLIP (Recommended)
```bash
bash run.sh    # New 3-stage CLIP pipeline
```

Expected: 70-73% PRCC mAP in ~48 hours

### 2. Access Legacy ResNet Code
```bash
git checkout v1.0-resnet-baseline
```

Git tag `v1.0-resnet-baseline` contains the last commit with full ResNet50-IBN support.

### 3. View Original Results
- **Git history**: `git show v1.0-resnet-baseline:README.md`
- **Backup outputs**: `outputs/transfer_backup_YYYYMMDD/`

---

## Code Migration Examples

### Old (ResNet)
```bash
# 5-stage pipeline
bash run.sh    # ExpT1-T5, 60 hours

# Training
python scripts/train.py \
  --backbone resnet50_ibn \
  --freeze-backbone-layers stem,layer1,layer2 \
  --epochs 120
```

### New (CLIP)
```bash
# 3-stage pipeline
bash run.sh    # Market → Joint → PRCC, 48 hours

# Training
python scripts/train.py \
  --backbone clip_vit_l \
  --backbone-lr 1e-5 \
  --head-lr 1e-4 \
  --epochs 60
```

### Key Differences
- **No per-layer freezing** for ViT (entire backbone or nothing)
- **Grouped learning rates** (`--backbone-lr` << `--head-lr`)
- **No part branch** for ViT (uses global features)
- **Fewer stages** (3 vs 5) but longer per-stage

---

## Performance Comparison

| Metric | ResNet50-IBN | CLIP ViT-L | Improvement |
|--------|--------------|------------|-------------|
| Market mAP | 90% | 88-90% | Maintained |
| Market Rank-1 | 93% | 92-95% | Maintained |
| PRCC mAP | **30-38%** | **70-73%** | +92-132% |
| PRCC Rank-1 | 35-42% | 68-72% | +71-83% |
| Training time | 60h | 48h | -20% |
| GPU memory | 8GB | 16GB | +100% |

**Conclusion**: PRCC performance gain (2x+) far outweighs Market slight decline (2-3%).
