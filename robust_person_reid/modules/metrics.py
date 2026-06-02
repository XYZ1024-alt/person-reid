from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


TOP_K = 5
PROTOCOL_STANDARD = "standard"
PROTOCOL_CLOTH_CHANGE = "cloth_change"


@dataclass(frozen=True)
class FeatureBank:
    features: torch.Tensor
    pids: torch.Tensor
    camids: torch.Tensor
    clothes_ids: torch.Tensor
    is_junk: torch.Tensor
    paths: list[str]


def extract_feature_bank(model, loader, device: torch.device) -> FeatureBank:
    model.eval()
    features, pids, camids, clothes_ids, is_junk, paths = [], [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch["image"].to(device))
            features.append(outputs["features"].cpu())
            pids.append(batch["pid"])
            camids.append(batch["camid"])
            clothes_ids.append(batch["clothes_id"])
            is_junk.append(batch["is_junk"])
            paths.extend(batch["path"])
    return FeatureBank(*_cat_bank(features, pids, camids, clothes_ids, is_junk), paths)


def evaluate_reid(query: FeatureBank, gallery: FeatureBank, protocol: str = PROTOCOL_STANDARD) -> dict[str, float]:
    distances = 1.0 - F.normalize(query.features, dim=1) @ F.normalize(gallery.features, dim=1).t()
    cmc_total = torch.zeros(TOP_K)
    average_precisions = []
    for index in range(len(query.pids)):
        valid = _valid_gallery(index, query, gallery, protocol)
        ranked = torch.argsort(distances[index])
        ranked = ranked[valid[ranked]]
        cmc_total += _cmc(ranked, gallery.pids, query.pids[index])
        average_precisions.append(_average_precision(ranked, gallery.pids, query.pids[index]))
    return _metrics(cmc_total, average_precisions, len(query.pids))


def _cat_bank(features, pids, camids, clothes_ids, is_junk):
    return (
        torch.cat(features),
        torch.cat(pids),
        torch.cat(camids),
        torch.cat(clothes_ids),
        torch.cat(is_junk).bool(),
    )


def _valid_gallery(index: int, query: FeatureBank, gallery: FeatureBank, protocol: str) -> torch.Tensor:
    same_pid = gallery.pids.eq(query.pids[index])
    valid = ~gallery.is_junk & ~(same_pid & gallery.camids.eq(query.camids[index]))
    if protocol == PROTOCOL_CLOTH_CHANGE:
        valid &= ~(same_pid & gallery.clothes_ids.eq(query.clothes_ids[index]))
    if not same_pid[valid].any():
        raise ValueError(f"No valid positive gallery match for query: {query.paths[index]}")
    return valid


def _cmc(indices: torch.Tensor, gallery_pids: torch.Tensor, query_pid: torch.Tensor) -> torch.Tensor:
    hits = gallery_pids[indices].eq(query_pid)
    first_hit = torch.nonzero(hits, as_tuple=False)[0].item()
    cmc = torch.zeros(TOP_K)
    if first_hit < TOP_K:
        cmc[first_hit:] = 1.0
    return cmc


def _average_precision(indices: torch.Tensor, gallery_pids: torch.Tensor, query_pid: torch.Tensor) -> float:
    hits = gallery_pids[indices].eq(query_pid).float()
    hit_count = int(hits.sum().item())
    if hit_count == 0:
        raise ValueError("Average precision requires at least one positive match")
    precision = torch.cumsum(hits, dim=0) / torch.arange(1, len(hits) + 1, dtype=torch.float32)
    return (precision * hits).sum().item() / hit_count


def _metrics(cmc_total: torch.Tensor, aps: list[float], query_count: int) -> dict[str, float]:
    divisor = float(query_count)
    metrics = {f"rank{rank + 1}": (cmc_total[rank] / divisor).item() for rank in range(TOP_K)}
    metrics["mAP"] = float(sum(aps) / len(aps))
    return metrics

