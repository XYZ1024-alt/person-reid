# PedestrianReID

Foundation-model person re-identification with CLIP ViT-L and EVA-02 Large backbones.

The current pipeline uses direct transfer:

1. Market-1501 pretraining.
2. PRCC fine-tuning from the Market checkpoint.

The previous Market + PRCC `joint` stage has been removed because it caused PRCC degradation in this codebase.

## Install

```bash
pip install -r requirements.txt
```

Core dependencies:

- PyTorch 2.0+
- transformers for CLIP
- timm for EVA-02
- ultralytics for detection demos

## Train

Run the full direct-transfer pipeline:

```bash
bash run.sh
```

Run a single stage:

```bash
START_STAGE=1 STOP_STAGE=1 bash run.sh
START_STAGE=3 bash run.sh
```

Manual Market pretraining:

```bash
python scripts/train.py \
  --backbone clip_vit_l \
  --backbone-lr 1e-5 \
  --head-lr 1e-4 \
  --mode market \
  --market-root Market-1501 \
  --epochs 60
```

Manual PRCC fine-tuning:

```bash
python scripts/train.py \
  --backbone clip_vit_l \
  --backbone-lr 5e-6 \
  --head-lr 5e-5 \
  --mode prcc \
  --prcc-root prcc \
  --use-sketch-fusion \
  --use-prcc-sketch \
  --pretrained-checkpoint outputs/transfer/stage1_market_clip/best.pth \
  --epochs 30
```

## Evaluate

```bash
python scripts/evaluate.py \
  --checkpoint outputs/transfer/stage3_prcc_direct/best.pth \
  --dataset prcc \
  --root prcc \
  --batch-size 128
```

For Market-1501:

```bash
python scripts/evaluate.py \
  --checkpoint outputs/transfer/stage1_market_clip/best.pth \
  --dataset market \
  --root Market-1501 \
  --batch-size 128
```

## Data Layout

Market-1501:

```text
Market-1501/
  pytorch/
    train/
    query/
    gallery/
```

PRCC:

```text
prcc/
  rgb/
    train/
      A/
      B/
      C/
    test/
      A/
      C/
  sketch/
    train/
      A/
      B/
      C/
    test/
      A/
      C/
```

`prcc` mode requires PRCC RGB data. Sketch fusion also requires matching sketch files.

## Important Options

- `--backbone clip_vit_l` or `--backbone eva02_l`
- `--backbone-lr` for the foundation backbone
- `--head-lr` for classifier and embedding heads
- `--freeze-backbone-epochs` freezes the entire backbone for the first N epochs
- `--use-sketch-fusion` enables the PRCC sketch fusion head
- `--cross-clothes-hard-negative-weight` weights different-ID same-clothes negatives in cross-clothes contrastive loss
- `--use-part-branch` is not supported with the current CLIP/EVA sequence backbones

## Checkpoints

Checkpoints store the model architecture in `model_config`, including `backbone_type` and `use_sketch_fusion`.
Evaluation requires these fields so the checkpoint is loaded with the same architecture used during training.
Older checkpoints without these fields must be retrained or explicitly migrated.

## Outputs

Default outputs:

```text
outputs/transfer/stage1_market_clip/
outputs/transfer/stage3_prcc_direct/
```

Each training directory includes:

- `best.pth`
- `last.pth`
- `run_config.json`
- `training_metrics.csv`
- `evaluation_metrics.csv`
- TensorBoard logs when enabled

## Distributed Training

```bash
GPUS=4 bash run.sh
```

Equivalent manual launch:

```bash
torchrun --nproc_per_node=4 scripts/train.py \
  --distributed \
  --backbone clip_vit_l \
  --mode prcc \
  --prcc-root prcc
```

## Project Structure

```text
pedestrian_reid/
  data/
  engine/
  modules/
scripts/
run.sh
```

## License

MIT License. See `LICENSE`.
