# PedestrianReID

This project now has a standalone pure PyTorch ReID path under `pedestrian_reid`.
It uses a pure PyTorch ResNet50-IBN backbone with BNNeck, CAL, and an optional
PCB-style part branch for changed-clothes PRCC transfer.
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

Training uses `MultiStepLR` by default with `--lr-milestones 40,70,100`
and `--lr-gamma 0.1`. Each run writes `run_config.json` to the output
directory with the full argument set, dataset summary, loader summary, DDP
summary, scheduler settings, and pretrained parameter count.

For distributed multi-GPU training, launch with `torchrun` and add
`--distributed`:

```powershell
torchrun --nproc_per_node=2 -m scripts.train --distributed
```

`--distributed` uses PyTorch `DistributedDataParallel`. It requires `torchrun`
and fails explicitly if the distributed environment is missing. In distributed
training, `--batch-size` is the global batch size and is split evenly across
GPUs. The older `--multi-gpu` flag still uses single-process `DataParallel` and
is kept only for compatibility.
`--ddp-find-unused-parameters auto` disables DDP unused-parameter detection for
simple stages such as Market-only pretraining, and enables it when warmup or
conditional sketch/CAL paths can temporarily leave parameters unused.

CAL requires clothes labels, so `--cal-weight` defaults to `0.5` and should be
used with PRCC or joint training. Market-1501 does not provide clothes labels.
Joint training uses source-balanced identity sampling by default. The transfer
recipe below uses a PRCC-heavy ratio so 75% of each batch's identities come from
PRCC. PRCC sketch images are used as training-only pose/shape supervision by
default; evaluation and deployment still use RGB only.
CAL uses PRCC outfit-level labels: each person's A/B images are one outfit and
C images are another outfit.
PRCC sampling is clothes-aware: for each sampled PRCC identity, the `--instances`
images must cover at least two clothes labels, so `--instances 4` samples from
both outfit states instead of four same-outfit images.

Useful PRCC options:

```powershell
--sketch-loss-weight 0.5 --rgb-sketch-consistency-weight 0.2
--prcc-identities-ratio 0.75 --cal-warmup-epochs 25 --cal-ramp-epochs 15 --disable-source-balanced-sampling
--color-jitter-probability 0.5 --random-grayscale-probability 0.2
--dark-augment-probability 0.15 --occlusion-augment-probability 0.2
```

## Transfer Training

The recommended training path is a five-stage transfer sequence:

```text
ExpT1: Market clean pretraining
ExpT2: Market dark adaptation
ExpT3: Market occlusion adaptation
ExpT4: Market to joint PRCC transfer
ExpT5: PRCC fine-tuning
```

Market first teaches standard RGB ReID. Dark and occlusion adaptation then add
scene robustness without disrupting the first clean pretraining stage. Joint and
PRCC transfer uses PRCC sketch consistency, PRCC-balanced sampling,
cross-clothes invariance, and a weak teacher-distillation constraint to reduce
clothing-color dependence without collapsing the Market-trained representation.

`--pretrained-checkpoint` loads every checkpoint parameter whose name and shape
match the current model. Market-to-Market stages keep the identity classifier,
while transfer stages skip classifiers automatically when class counts change.
Use `--best-metric mAP` for paper runs so `best.pth` is selected by retrieval
quality across the ranked list instead of only the first match.
Use `--best-variant dark` or `--best-variant occluded` when a stage is meant
to optimize that evaluation condition; otherwise use `standard`.
During PRCC transfer, `--freeze-backbone-epochs 10 --freeze-backbone-layers stem,layer1,layer2`
keeps low-level ResNet50-IBN features fixed at the start, then automatically
unfreezes them after epoch 10.

Run the default transfer experiment through ExpT4:

```bash
bash run.sh
```

Resume from a later stage after previous checkpoints already exist:

```bash
START_STAGE=4 bash run.sh
```

`run.sh` uses one GPU by default. Set `GPUS=2` or higher to launch with
`torchrun --distributed`. Useful script overrides:

```bash
GPUS=2 BATCH_SIZE=128 NUM_WORKERS=12 EXP_ROOT=outputs/transfer bash run.sh
RUN_EXPT4_NODISTILL=1 START_STAGE=4 STOP_STAGE=4 bash run.sh
STOP_STAGE=5 bash run.sh
```

### ExpT1: Market Clean Pretraining

This stage learns standard Market-1501 ReID without dark or occlusion
augmentation:

```powershell
torchrun --nproc_per_node=2 -m scripts.train --distributed --mode market --epochs 120 --batch-size 128 --num-workers 12 --cal-weight 0 --no-use-prcc-sketch --use-part-branch --num-parts 6 --part-embedding-dim 256 --part-triplet-weight 0.3 --combined-global-weight 0.7 --combined-part-weight 0.3 --feature-key combined_features --best-metric mAP --best-variant standard --eval-period 5 --lr-milestones 40,70,100 --color-jitter-probability 0.5 --random-grayscale-probability 0 --dark-augment-probability 0.10 --occlusion-augment-probability 0.10 --output-dir outputs/transfer/expT1_market_clean
```

Evaluate ExpT1:

```powershell
python -m scripts.evaluate --checkpoint outputs/transfer/expT1_market_clean/best.pth --dataset market --feature-key combined_features
```

### ExpT2: Market Dark Adaptation

This stage loads ExpT1 and adapts the Market model to low-light queries:

```powershell
torchrun --nproc_per_node=2 -m scripts.train --distributed --mode market --epochs 30 --batch-size 128 --num-workers 12 --lr 0.0001 --cal-weight 0 --no-use-prcc-sketch --use-part-branch --num-parts 6 --part-embedding-dim 256 --part-triplet-weight 0.3 --combined-global-weight 0.7 --combined-part-weight 0.3 --feature-key combined_features --best-metric mAP --best-variant dark --eval-period 10 --lr-milestones 10,20 --color-jitter-probability 0.1 --random-grayscale-probability 0 --dark-augment-probability 0.15 --occlusion-augment-probability 0 --pretrained-checkpoint outputs/transfer/expT1_market_clean/best.pth --output-dir outputs/transfer/expT2_market_dark
```

Evaluate ExpT2:

```powershell
python -m scripts.evaluate --checkpoint outputs/transfer/expT2_market_dark/best.pth --dataset market --feature-key combined_features
```

### ExpT3: Market Occlusion Adaptation

This stage loads ExpT2 and adapts the Market model to occluded queries:

```powershell
torchrun --nproc_per_node=2 -m scripts.train --distributed --mode market --epochs 30 --batch-size 128 --num-workers 12 --lr 0.0001 --cal-weight 0 --no-use-prcc-sketch --use-part-branch --num-parts 6 --part-embedding-dim 256 --part-triplet-weight 0.3 --combined-global-weight 0.7 --combined-part-weight 0.3 --feature-key combined_features --best-metric mAP --best-variant occluded --eval-period 10 --lr-milestones 10,20 --color-jitter-probability 0.1 --random-grayscale-probability 0 --dark-augment-probability 0 --occlusion-augment-probability 0.2 --pretrained-checkpoint outputs/transfer/expT2_market_dark/best.pth --output-dir outputs/transfer/expT3_market_occlusion
```

Evaluate ExpT3:

```powershell
python -m scripts.evaluate --checkpoint outputs/transfer/expT3_market_occlusion/best.pth --dataset market --feature-key combined_features
```

### ExpT4: Market to Joint PRCC Transfer

This stage uses Market + PRCC, PRCC-heavy source-balanced identity sampling,
PRCC sketch consistency, clothes-aware PRCC identity sampling, a PCB-style
local part branch, PRCC cross-clothes invariance, and weak teacher
distillation from ExpT3. CAL is disabled in the main transfer run:

```powershell
python -m scripts.train --mode joint --epochs 25 --batch-size 128 --num-workers 12 --lr 0.00005 --cal-weight 0 --cal-warmup-epochs 0 --cal-ramp-epochs 0 --sketch-loss-weight 0 --rgb-sketch-consistency-weight 0.02 --sketch-warmup-epochs 10 --sketch-ramp-epochs 10 --prcc-identities-ratio 0.75 --use-part-branch --num-parts 6 --part-embedding-dim 256 --part-triplet-weight 0.3 --cloth-invariant-weight 0.2 --combined-global-weight 0.7 --combined-part-weight 0.3 --teacher-checkpoint outputs/transfer/expT3_market_occlusion/best.pth --distill-weight 0.1 --distill-final-weight 0.05 --distill-hold-epochs 0 --distill-ramp-epochs 3 --feature-key combined_features --best-metric mAP --best-variant standard --eval-period 1 --lr-milestones 12,18,22 --freeze-backbone-epochs 10 --freeze-backbone-layers stem,layer1,layer2 --color-jitter-probability 0.5 --random-grayscale-probability 0.2 --dark-augment-probability 0.05 --occlusion-augment-probability 0.1 --pretrained-checkpoint outputs/transfer/expT3_market_occlusion/best.pth --output-dir outputs/transfer/expT4_market_to_joint_prcc
```

For the no-distillation control run, keep every option above and set both
distillation weights to 0, or run
`RUN_EXPT4_NODISTILL=1 START_STAGE=4 STOP_STAGE=4 bash run.sh`.

Evaluate ExpT4:

```powershell
python -m scripts.evaluate --checkpoint outputs/transfer/expT4_market_to_joint_prcc/best.pth --dataset market --feature-key combined_features
python -m scripts.evaluate --checkpoint outputs/transfer/expT4_market_to_joint_prcc/best.pth --dataset prcc --feature-key combined_features
```

### ExpT5: PRCC Fine-tuning

This optional diagnostic stage freezes the backbone and briefly fine-tunes the
part/head layers on PRCC. It is not run by default because recent results showed
ExpT5 did not improve over ExpT4:

```powershell
python -m scripts.train --mode prcc --epochs 3 --batch-size 128 --num-workers 12 --lr 0.00003 --cal-weight 0 --cal-warmup-epochs 0 --cal-ramp-epochs 0 --no-use-prcc-sketch --sketch-loss-weight 0 --rgb-sketch-consistency-weight 0 --sketch-warmup-epochs 0 --sketch-ramp-epochs 0 --use-part-branch --num-parts 6 --part-embedding-dim 256 --part-triplet-weight 0.3 --cloth-invariant-weight 0.1 --combined-global-weight 0.7 --combined-part-weight 0.3 --teacher-checkpoint outputs/transfer/expT4_market_to_joint_prcc/best.pth --distill-weight 0.1 --distill-final-weight 0.1 --distill-hold-epochs 0 --distill-ramp-epochs 0 --freeze-backbone-all-epochs --feature-key combined_features --best-metric mAP --best-variant standard --eval-period 1 --lr-milestones 1,2 --color-jitter-probability 0.5 --random-grayscale-probability 0.25 --dark-augment-probability 0.05 --occlusion-augment-probability 0.1 --pretrained-checkpoint outputs/transfer/expT4_market_to_joint_prcc/best.pth --output-dir outputs/transfer/expT5_prcc_finetune
```

Evaluate ExpT5:

```powershell
python -m scripts.evaluate --checkpoint outputs/transfer/expT5_prcc_finetune/best.pth --dataset prcc --feature-key combined_features
```

Historical full/sketch ablations were kept for comparison only; the transfer
path above is the current main experiment line.

## Evaluate

```powershell
python -m scripts.evaluate --checkpoint outputs/pedestrian_reid/best.pth --dataset market
python -m scripts.evaluate --checkpoint outputs/pedestrian_reid/best.pth --dataset prcc
```

Evaluation reports standard, dark-query, and occluded-query Rank-1/Rank-5/mAP.

Market evaluation follows the standard protocol: junk images (pid -1) are
ignored while 0000 distractor images stay in the gallery. Results produced
before this protocol fix are not directly comparable.

## Plot Figures

Training writes metrics to:

```text
outputs/pedestrian_reid/training_metrics.csv
outputs/pedestrian_reid/evaluation_metrics.csv
```

Generate paper figures after training:

```powershell
python -m scripts.plot_metrics --dataset prcc
python -m scripts.plot_metrics --dataset market
```

Figures are saved under:

```text
outputs/pedestrian_reid/figures
```
