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
python -m scripts.train --mode joint --epochs 60
python -m scripts.train --mode prcc --epochs 60
python -m scripts.train --mode market --cal-weight 0 --epochs 60
```

CAL requires clothes labels, so `--cal-weight` defaults to `0.5` and should be
used with PRCC or joint training. Market-1501 does not provide clothes labels.

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
