from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from robust_person_reid.builders import MODE_MARKET, MODE_PRCC, build_eval_loader
from robust_person_reid.data.transforms import VARIANT_STANDARD
from robust_person_reid.engine.evaluator import extract_feature_bank, load_model


DEFAULT_OUTPUT_DIR = "outputs/robust_person_reid"
DEFAULT_MARKET_ROOT = "Market-1501"
DEFAULT_PRCC_ROOT = "prcc"
DEFAULT_BATCH_SIZE = 64
DEFAULT_WORKERS = 4
DEFAULT_MATRIX_SIZE = 64
TRAIN_METRICS_CSV = "training_metrics.csv"
EVAL_METRICS_CSV = "evaluation_metrics.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training curves and ReID similarity matrix")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--dataset", choices=[MODE_MARKET, MODE_PRCC], default=MODE_PRCC)
    parser.add_argument("--root", default="")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--matrix-size", type=int, default=DEFAULT_MATRIX_SIZE)
    parser.add_argument("--fig-dir", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.root = args.root or default_root(args.dataset)
    args.checkpoint = args.checkpoint or str(Path(args.output_dir) / "best.pth")
    args.fig_dir = args.fig_dir or str(Path(args.output_dir) / "figures")
    return args


def default_root(dataset: str) -> str:
    if dataset == MODE_PRCC:
        return DEFAULT_PRCC_ROOT
    return DEFAULT_MARKET_ROOT


def main() -> None:
    args = parse_args()
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    train_rows = read_csv_rows(Path(args.output_dir) / TRAIN_METRICS_CSV)
    eval_rows = read_csv_rows(Path(args.output_dir) / EVAL_METRICS_CSV)
    plot_training_curves(train_rows, fig_dir)
    plot_eval_metric(eval_rows, fig_dir, "rank1")
    plot_eval_metric(eval_rows, fig_dir, "mAP")
    plot_similarity_matrix(args, fig_dir)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No metric rows found in {path}")
    return rows


def plot_training_curves(rows: list[dict[str, str]], fig_dir: Path) -> None:
    epochs = [int(row["epoch"]) for row in rows]
    plt.figure(figsize=(8, 5))
    for metric in ["loss", "ce", "triplet", "cal"]:
        plt.plot(epochs, _float_column(rows, metric), marker="o", label=metric)
    _save_line_plot(fig_dir / "training_loss_curves.png", "Training Loss Curves", "Epoch", "Loss")


def plot_eval_metric(rows: list[dict[str, str]], fig_dir: Path, metric: str) -> None:
    plt.figure(figsize=(8, 5))
    for label, group in _group_eval_rows(rows).items():
        epochs = [int(row["epoch"]) for row in group]
        plt.plot(epochs, _float_column(group, metric), marker="o", label=label)
    title = f"Evaluation {metric} Curves"
    _save_line_plot(fig_dir / f"evaluation_{metric}_curves.png", title, "Epoch", metric)


def plot_similarity_matrix(args: argparse.Namespace, fig_dir: Path) -> None:
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    query = _feature_bank(model, args.root, args.dataset, "query", device, args)
    gallery = _feature_bank(model, args.root, args.dataset, "gallery", device, args)
    matrix = _similarity_matrix(query.features, gallery.features, args.matrix_size)
    plt.figure(figsize=(7, 6))
    plt.imshow(matrix.numpy(), cmap="viridis", aspect="auto", vmin=-1.0, vmax=1.0)
    plt.colorbar(label="Cosine similarity")
    plt.xlabel("Gallery samples")
    plt.ylabel("Query samples")
    plt.title(f"{args.dataset} Query-Gallery Similarity Matrix")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{args.dataset}_similarity_matrix.png", dpi=300)
    plt.close()


def _feature_bank(model, root: str, dataset: str, split: str, device: torch.device, args: argparse.Namespace):
    loader = build_eval_loader(root, dataset, split, VARIANT_STANDARD, args)
    return extract_feature_bank(model, loader, device)


def _similarity_matrix(query_features: torch.Tensor, gallery_features: torch.Tensor, size: int) -> torch.Tensor:
    query = F.normalize(query_features[:size], dim=1)
    gallery = F.normalize(gallery_features[:size], dim=1)
    return query @ gallery.t()


def _float_column(rows: list[dict[str, str]], key: str) -> list[float]:
    return [float(row[key]) for row in rows]


def _group_eval_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        label = f"{row['dataset']}/{row['variant']}"
        groups.setdefault(label, []).append(row)
    return groups


def _save_line_plot(path: Path, title: str, xlabel: str, ylabel: str) -> None:
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
