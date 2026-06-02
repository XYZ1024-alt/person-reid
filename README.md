# RobustPersonReID

This project now has a standalone pure PyTorch ReID path under `robust_person_reid`.
It uses a pure PyTorch ResNet50-IBN backbone with BNNeck and CAL.
It initializes the custom ResNet50-IBN backbone from ImageNet ResNet50 weights.
It does not use external ReID frameworks.

## Data

- Market-1501: use the existing `Market-1501/pytorch/train`, `query`, and `gallery` folders.
- PRCC: use the existing `prcc` folder. The default layout is `rgb/train/A|B|C` and `rgb/test/A|C`.

`joint` and `prcc` modes require PRCC. If PRCC is missing, training fails explicitly.

## Install

```powershell
pip install -r requirements.txt
```

## Train

```powershell
python -m scripts.train --mode joint --epochs 80 --batch-size 128 --num-workers 8 --cal-weight 0.1
python -m scripts.train --mode prcc --epochs 60
python -m scripts.train --mode market --cal-weight 0 --epochs 60
```

CAL requires clothes labels, so `--cal-weight` defaults to `0.5` and should be
used with PRCC or joint training. Market-1501 does not provide clothes labels.
Joint training uses source-balanced identity sampling by default, with half of
each batch's identities from PRCC. CAL uses PRCC clothes-state labels: A/B are
same-clothes and C is changed-clothes.

Useful PRCC options:

```powershell
--prcc-identities-ratio 0.5 --cal-warmup-epochs 10 --cal-ramp-epochs 10 --disable-source-balanced-sampling
```

## Ablation

Run PRCC-focused ablation with separate output folders:

```powershell
python -m scripts.train --mode joint --epochs 80 --batch-size 128 --num-workers 8 --cal-weight 0 --disable-source-balanced-sampling --output-dir outputs/ablation/baseline
python -m scripts.train --mode joint --epochs 80 --batch-size 128 --num-workers 8 --cal-weight 0.1 --cal-warmup-epochs 0 --cal-ramp-epochs 0 --disable-source-balanced-sampling --output-dir outputs/ablation/correct_cal
python -m scripts.train --mode joint --epochs 80 --batch-size 128 --num-workers 8 --cal-weight 0.1 --cal-warmup-epochs 10 --cal-ramp-epochs 10 --disable-source-balanced-sampling --output-dir outputs/ablation/cal_schedule
python -m scripts.train --mode joint --epochs 80 --batch-size 128 --num-workers 8 --cal-weight 0.1 --cal-warmup-epochs 10 --cal-ramp-epochs 10 --prcc-identities-ratio 0.5 --output-dir outputs/ablation/full
```

Evaluate both PRCC and Market-1501 for each run:

```powershell
python -m scripts.evaluate --checkpoint outputs/ablation/baseline/best.pth --dataset prcc
python -m scripts.evaluate --checkpoint outputs/ablation/baseline/best.pth --dataset market
```

Repeat the evaluation command for `correct_cal`, `cal_schedule`, and `full`.

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
