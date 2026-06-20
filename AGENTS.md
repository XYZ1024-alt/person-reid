# Agent Guide for PedestrianReID

This file is intended for AI coding agents that work on this repository. It captures the project structure, conventions, build/test commands, and other practical details needed to make safe, useful changes.

## Project Overview

PedestrianReID is a pure-PyTorch person re-identification (ReID) project. It trains a custom ResNet50-IBN backbone with BNNeck, an optional PCB-style part branch, clothes-aware learning (CAL), and cross-domain transfer from Market-1501 to the changed-clothes PRCC dataset. It also contains a small demo pipeline that combines YOLO person detection with the trained ReID model for video matching.

The project does **not** use external ReID frameworks (no `torchreid`, `fastreid`, etc.). All training, evaluation, model definition, data loading, and metrics live under the `pedestrian_reid` package.

## Technology Stack

- **Language:** Python 3 (the code uses `from __future__ import annotations`, type hints, and dataclasses).
- **Deep Learning:** PyTorch + torchvision.
- **Detection:** Ultralytics YOLO (used only in the demo pipeline).
- **Data / Image processing:** Pillow, OpenCV.
- **Logging:** TensorBoard (`tensorboard` package), CSV files.
- **Utilities:** tqdm, matplotlib.
- **No formal packaging:** There is no `pyproject.toml`, `setup.py`, `setup.cfg`, or `package.json`. Dependencies are listed in `requirements.txt` and installed directly with `pip`.

## Project Structure

```text
.
├── requirements.txt              # Python dependencies
├── README.md                     # Human-facing training/eval recipes
├── run.sh                        # Bash orchestration for the 5-stage transfer recipe
├── .gitignore                    # Excludes outputs/, datasets, __pycache__, .idea, mcps
│
├── pedestrian_reid/              # Core ReID library
│   ├── builders.py               # Dataset / DataLoader construction; mode constants
│   ├── runtime.py                # torch.multiprocessing sharing strategy setup
│   ├── config/__init__.py        # Module marker (defaults live in code)
│   ├── data/
│   │   ├── datasets.py           # Market-1501 / PRCC parsing and ReIDSample dataclass
│   │   ├── samplers.py           # Identity-balanced and clothes-aware batch samplers
│   │   └── transforms.py         # Training/evaluation transforms (dark/occlusion variants)
│   ├── engine/
│   │   ├── trainer.py            # Full training loop, checkpointing, DDP, mixed precision
│   │   └── evaluator.py          # Evaluation protocol and checkpoint loading
│   └── modules/
│       ├── model.py              # ResNet50-IBN backbone, BNNeck, part branch, CAL head
│       ├── losses.py             # Batch-hard triplet loss
│       └── metrics.py            # Rank-1/Rank-5/mAP computation and FeatureBank
│
├── scripts/                      # CLI entry points
│   ├── train.py                  # `python -m scripts.train ...`
│   ├── evaluate.py               # `python -m scripts.evaluate ...`
│   ├── extract.py                # Extract a single image embedding
│   ├── plot_metrics.py           # Plot training/eval curves and similarity matrix
│   └── diagnose_checkpoint.py    # Diagnose checkpoint quality on Market train split
│
├── train.py, evaluate.py,        # Thin wrappers around scripts.* for convenience
├── extract.py, plot_metrics.py   # (they just import `main` and run it)
├── main.py                       # Demo: YOLO detection + ReID matching on a video
│
├── modules/                      # Demo-only helpers
│   ├── detector.py               # YOLO-based person cropping
│   └── reid_engine.py            # Feature extraction and cosine similarity
├── models/                       # Demo-only predictor loader
│   └── loader.py                 # PedestrianReIDPredictor + YOLO init helpers
│
├── tests/                        # Unit tests
│   └── test_prcc_objective_shift.py
│
├── Market-1501/                  # Dataset (gitignored)
├── prcc/                         # Dataset (gitignored)
├── outputs/                      # Training outputs (gitignored)
└── mcps/                         # MCP server tool schemas (gitignored, not core)
```

## Build and Test Commands

### Install dependencies

```powershell
pip install -r requirements.txt
```

There is no build step; the project runs as plain Python modules.

### Run tests

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

The only current test file is `tests/test_prcc_objective_shift.py`. It exercises PRCC dev-split construction, PRCC CE weight ramping, feature-key validation, and cross-clothes contrastive loss behavior.

### Training

The recommended entry point for experiments is the module form:

```powershell
python -m scripts.train --mode joint --epochs 80 --batch-size 256 --num-workers 8
```

Convenience wrapper (identical behavior):

```powershell
python train.py --mode joint --epochs 80 --batch-size 256 --num-workers 8
```

Distributed multi-GPU:

```powershell
torchrun --nproc_per_node=2 -m scripts.train --distributed
```

The full transfer recipe is driven by `run.sh`:

```bash
bash run.sh                 # default: stages 1-4 with joint_v1 recipe
START_STAGE=4 bash run.sh   # resume from stage 4
GPUS=2 bash run.sh          # distributed multi-GPU
```

### Evaluation

```powershell
python -m scripts.evaluate --checkpoint outputs/pedestrian_reid/best.pth --dataset market
python -m scripts.evaluate --checkpoint outputs/pedestrian_reid/best.pth --dataset prcc
```

### Plotting

```powershell
python -m scripts.plot_metrics --dataset prcc
python -m scripts.plot_metrics --dataset market
```

Figures are written to `outputs/pedestrian_reid/figures`.

### Single-image embedding

```powershell
python -m scripts.extract --checkpoint outputs/pedestrian_reid/best.pth --image path/to/image.jpg
```

### Demo pipeline

```powershell
python main.py
```

This expects `outputs/pedestrian_reid/best.pth`, `data/target.jpg`, and `data/video.mp4`.

## Code Style Guidelines

- Start every module with `from __future__ import annotations`.
- Use type hints throughout; prefer built-in generics (`list[...]`, `dict[...]`) and union syntax (`str | None`).
- Use dataclasses for configuration and structured state objects.
- Constants are `UPPER_CASE` and declared near the top of the module.
- CLI arguments are defined with `argparse` in the `scripts/` entry point, not inside `pedestrian_reid`.
- Prefer explicit validation and early failure (`ValueError`/`RuntimeError` with clear messages).
- Keep loss/metric logic deterministic and reproducible: use explicit seeds and avoid hidden global state.
- Avoid committing checkpoints, dataset files, TensorBoard logs, or `__pycache__`.

## Testing Instructions

- Tests live under `tests/` and use the standard `unittest` framework.
- Run the full suite with `python -m unittest discover -s tests -p "test_*.py" -v`.
- Many tests require `torch` to be installed.
- The existing tests are focused on PRCC dev-split correctness and loss-component math; add similar unit-style tests when introducing new data splits or loss terms.

## Security Considerations

- **Do not commit credentials:** The repository contains no `.env`, secret files, or API keys. Keep it that way.
- **Do not commit large artifacts:** `.gitignore` already excludes `outputs/`, `Market-1501/`, `prcc/`, `mcps/`, `__pycache__/`, and `*.pyc`. Do not remove these exclusions.
- **Avoid modifying datasets or outputs unless asked:** These directories are considered user data / experiment results.
- **No network services:** The codebase does not expose any server or endpoint; it is purely offline training/evaluation tooling.
- **Checkpoint safety:** Checkpoints are saved as `.pth` files via `torch.save`. Loading checkpoints uses `map_location` to support CPU/CUDA transfers safely.

## Key Development Conventions

- **Modes:** `market`, `prcc`, `joint`. `prcc_dev` is used only for evaluation of a held-out identity split.
- **Datasets:**
  - Market-1501 expects `Market-1501/pytorch/{train,query,gallery}`.
  - PRCC expects `prcc/rgb/{train,test/A,test/C}` and optionally paired `prcc/sketch/...`.
- **Checkpoints:** Each run writes `last.pth`, `best.pth`, `run_config.json`, `training_metrics.csv`, `evaluation_metrics.csv`, and a `tensorboard/` directory under `--output-dir`.
- **Best checkpoint selection:** Controlled by `--best-metric` (`rank1` or `mAP`), `--best-dataset`, and `--best-variant` (`standard`, `dark`, `occluded`).
- **Mixed precision:** FP16 is the default when CUDA is available; override with `--precision fp32`.
- **Backbone freezing:** Use `--freeze-backbone-epochs` and `--freeze-backbone-layers stem,layer1,layer2` to keep low-level ResNet layers frozen during transfer stages.
- **Feature keys:**
  - `bn_features` (default global BNNeck feature).
  - `features` (pre-BN embedding).
  - `combined_features` (global + part branch; requires `--use-part-branch`).
- **Distributed training:** Use `--distributed` with `torchrun`. `--multi-gpu` is a legacy single-process `DataParallel` flag kept for compatibility; do not mix with `--distributed`.

## Notes for Agents

- When editing training behavior, start in `pedestrian_reid/engine/trainer.py` and `pedestrian_reid/builders.py`.
- When editing model architecture, start in `pedestrian_reid/modules/model.py`.
- When editing data loading or sampling, modify `pedestrian_reid/data/`.
- When adding a new CLI flag, add it to the relevant `scripts/*.py` file and propagate defaults through `pedestrian_reid/builders.py` or `pedestrian_reid/engine/trainer.py` as needed.
- Before large refactors, run the unit tests and, if possible, a short training smoke test with `--epochs 1 --eval-period 1`.
- Keep the `README.md` and this `AGENTS.md` in sync when changing user-facing commands or project conventions.
