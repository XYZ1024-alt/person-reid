# RobustPersonReID

This project now has a standalone pure PyTorch ReID path under `robust_person_reid`.
It uses a pure PyTorch ResNet50-IBN backbone with BNNeck and CAL.
It initializes the custom ResNet50-IBN backbone from ImageNet ResNet50 weights.
It does not use external ReID frameworks.

## Data

- Market-1501: use the existing `Market-1501/pytorch/train`, `query`, and `gallery` folders.
- PRCC: use the existing `prcc` folder. The default layout is `rgb/train/A|B|C`,
  `rgb/test/A|C`, and paired `sketch` folders with matching filenames.

`joint` and `prcc` modes require PRCC. If PRCC is missing, training fails explicitly.

## Install

```powershell
pip install -r requirements.txt
```

## Train

```powershell
python -m scripts.train --mode joint --epochs 80 --batch-size 256 --num-workers 8 --cal-weight 0.05 --cal-warmup-epochs 20 --cal-ramp-epochs 20
python -m scripts.train --mode prcc --epochs 60
python -m scripts.train --mode market --cal-weight 0 --epochs 60
```

CUDA training uses FP16 mixed precision by default. DataLoader uses pinned
memory by default and keeps workers persistent when `--num-workers` is greater
than 0. To disable these speed options:

```powershell
--precision fp32 --no-pin-memory --no-persistent-workers
```

For single-process multi-GPU training, add:

```powershell
--multi-gpu
```

`--multi-gpu` uses PyTorch `DataParallel` over all visible CUDA GPUs and fails
explicitly if fewer than two GPUs are visible.

CAL requires clothes labels, so `--cal-weight` defaults to `0.5` and should be
used with PRCC or joint training. Market-1501 does not provide clothes labels.
Joint training uses source-balanced identity sampling by default, with half of
each batch's identities from PRCC. PRCC sketch images are used as training-only
pose/shape supervision by default; evaluation and deployment still use RGB only.
CAL uses PRCC outfit-level labels: each person's A/B images are one outfit and
C images are another outfit.

Useful PRCC options:

```powershell
--sketch-loss-weight 0.5 --rgb-sketch-consistency-weight 0.2
--prcc-identities-ratio 0.5 --cal-warmup-epochs 20 --cal-ramp-epochs 20 --disable-source-balanced-sampling
```

## Ablation

Run PRCC-focused ablation with separate output folders:

```powershell
python -m scripts.train --mode joint --epochs 80 --batch-size 256 --num-workers 8 --cal-weight 0 --no-use-prcc-sketch --disable-source-balanced-sampling --output-dir outputs/ablation/baseline
python -m scripts.train --mode joint --epochs 80 --batch-size 256 --num-workers 8 --cal-weight 0 --sketch-loss-weight 0.5 --rgb-sketch-consistency-weight 0 --disable-source-balanced-sampling --output-dir outputs/ablation/sketch_id
python -m scripts.train --mode joint --epochs 80 --batch-size 256 --num-workers 8 --cal-weight 0 --sketch-loss-weight 0.5 --rgb-sketch-consistency-weight 0.2 --disable-source-balanced-sampling --output-dir outputs/ablation/sketch_consistency
python -m scripts.train --mode joint --epochs 80 --batch-size 256 --num-workers 8 --cal-weight 0.05 --cal-warmup-epochs 20 --cal-ramp-epochs 20 --sketch-loss-weight 0.5 --rgb-sketch-consistency-weight 0.2 --prcc-identities-ratio 0.5 --output-dir outputs/ablation/full
```

Evaluate both PRCC and Market-1501 for each run:

```powershell
python -m scripts.evaluate --checkpoint outputs/ablation/baseline/best.pth --dataset prcc
python -m scripts.evaluate --checkpoint outputs/ablation/baseline/best.pth --dataset market
```

Repeat the evaluation command for `sketch_id`, `sketch_consistency`, and `full`.

## Accuracy Tuning Stage 1

The current strongest baseline is RGB-only joint training. First test small CAL
weights without sketch or source-balanced sampling:

```powershell
python -m scripts.train --mode joint --epochs 80 --batch-size 512 --num-workers 12 --multi-gpu --cal-weight 0 --no-use-prcc-sketch --disable-source-balanced-sampling --output-dir outputs/stage1/expA_rgb_baseline
python -m scripts.train --mode joint --epochs 80 --batch-size 512 --num-workers 12 --multi-gpu --cal-weight 0.02 --cal-warmup-epochs 20 --cal-ramp-epochs 20 --no-use-prcc-sketch --disable-source-balanced-sampling --output-dir outputs/stage1/expB_cal002
python -m scripts.train --mode joint --epochs 80 --batch-size 512 --num-workers 12 --multi-gpu --cal-weight 0.03 --cal-warmup-epochs 20 --cal-ramp-epochs 20 --no-use-prcc-sketch --disable-source-balanced-sampling --output-dir outputs/stage1/expC_cal003
python -m scripts.train --mode joint --epochs 80 --batch-size 512 --num-workers 12 --multi-gpu --cal-weight 0.05 --cal-warmup-epochs 20 --cal-ramp-epochs 20 --no-use-prcc-sketch --disable-source-balanced-sampling --output-dir outputs/stage1/expD_cal005
```

Evaluate each run on PRCC and Market before trying sketch again.

## Accuracy Tuning Stage 2

Stage 2 fine-tunes PRCC from the best stage 1 RGB checkpoint. It loads only the
compatible backbone, embedding, and BNNeck weights from `--pretrained-checkpoint`;
the PRCC identity classifier is trained from scratch. Sketch consistency is a
small structure-only target and does not use sketch identity CE/triplet when
`--sketch-loss-weight 0` is set.

```powershell
python -m scripts.train --mode prcc --epochs 40 --batch-size 512 --num-workers 12 --multi-gpu --lr 0.0001 --cal-weight 0 --no-use-prcc-sketch --pretrained-checkpoint outputs/stage1/expA_rgb_baseline/best.pth --output-dir outputs/stage2/expE_prcc_finetune_rgb
python -m scripts.train --mode prcc --epochs 40 --batch-size 512 --num-workers 12 --multi-gpu --lr 0.0001 --cal-weight 0 --sketch-loss-weight 0 --rgb-sketch-consistency-weight 0.01 --sketch-warmup-epochs 10 --sketch-ramp-epochs 10 --pretrained-checkpoint outputs/stage1/expA_rgb_baseline/best.pth --output-dir outputs/stage2/expF_prcc_sketch_con001
python -m scripts.train --mode prcc --epochs 40 --batch-size 512 --num-workers 12 --multi-gpu --lr 0.0001 --cal-weight 0 --sketch-loss-weight 0 --rgb-sketch-consistency-weight 0.02 --sketch-warmup-epochs 10 --sketch-ramp-epochs 10 --pretrained-checkpoint outputs/stage1/expA_rgb_baseline/best.pth --output-dir outputs/stage2/expG_prcc_sketch_con002
python -m scripts.train --mode prcc --epochs 40 --batch-size 512 --num-workers 12 --multi-gpu --lr 0.0001 --cal-weight 0 --sketch-loss-weight 0 --rgb-sketch-consistency-weight 0.05 --sketch-warmup-epochs 10 --sketch-ramp-epochs 10 --pretrained-checkpoint outputs/stage1/expA_rgb_baseline/best.pth --output-dir outputs/stage2/expH_prcc_sketch_con005
```

Evaluate each stage 2 run on PRCC:

```powershell
python -m scripts.evaluate --checkpoint outputs/stage2/expE_prcc_finetune_rgb/best.pth --dataset prcc
python -m scripts.evaluate --checkpoint outputs/stage2/expF_prcc_sketch_con001/best.pth --dataset prcc
python -m scripts.evaluate --checkpoint outputs/stage2/expG_prcc_sketch_con002/best.pth --dataset prcc
python -m scripts.evaluate --checkpoint outputs/stage2/expH_prcc_sketch_con005/best.pth --dataset prcc
```

## Evaluate

```powershell
python -m scripts.evaluate --checkpoint outputs/robust_person_reid/best.pth --dataset market
python -m scripts.evaluate --checkpoint outputs/robust_person_reid/best.pth --dataset prcc
```

Evaluation reports standard, dark-query, and occluded-query Rank-1/Rank-5/mAP.

## Plot Figures

Training writes metrics to:

```text
outputs/robust_person_reid/training_metrics.csv
outputs/robust_person_reid/evaluation_metrics.csv
```

Generate paper figures after training:

```powershell
python -m scripts.plot_metrics --dataset prcc
python -m scripts.plot_metrics --dataset market
```

Figures are saved under:

```text
outputs/robust_person_reid/figures
```
