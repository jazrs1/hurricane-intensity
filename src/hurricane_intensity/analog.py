"""Building blocks for the analog-forecasting experiments: storm-capped batches, embedding/regression models,
storm-deduplicated kNN retrieval and intensity metrics by Saffir-Simpson category."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler

# ImageNet statistics. The three input channels are BT frames (t-6 h, t-3 h, t) scaled to 0-1, not RGB; they are
# normalized with the ImageNet per-channel mean/std that the pretrained backbone expects (all arms alike).
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Saffir-Simpson categories by intensity (kt): TD < 34, TS 34-63, C1 64-82, C2 83-95, C3 96-112, C4 113-136, C5 >= 137
CATEGORY_EDGES = (34, 64, 83, 96, 113, 137)
CATEGORIES = ("TD", "TS", "C1", "C2", "C3", "C4", "C5")


def category(kt):
    return np.asarray(CATEGORIES)[np.searchsorted(CATEGORY_EDGES, np.asarray(kt), side="right")]


class StormCappedBatchSampler(Sampler):
    """Shuffled batches of `batch_size` with at most `max_per_storm` windows from any one storm.

    Each epoch: shuffle all indices and fill batches greedily; a window whose storm is at its cap in the current
    batch is deferred and offered first to the next batch. Windows still deferred when no storm can fill another
    batch end up in final, smaller batches (still capped); batches smaller than `min_batch` are dropped and
    counted in `self.dropped` for that epoch.
    """

    def __init__(self, storm_of, batch_size=128, max_per_storm=4, min_batch=8, generator=None):
        self.storm_of = np.asarray(storm_of)
        self.batch_size, self.max_per_storm, self.min_batch = batch_size, max_per_storm, min_batch
        self.generator = generator
        self.dropped = 0

    def _batches(self):
        order = torch.randperm(len(self.storm_of), generator=self.generator).tolist()
        pending, batches = [], []
        while order or pending:
            batch, counts, deferred = [], {}, []
            for idx in pending + order:
                if len(batch) == self.batch_size:
                    deferred.append(idx)
                    continue
                s = self.storm_of[idx]
                if counts.get(s, 0) < self.max_per_storm:
                    batch.append(idx)
                    counts[s] = counts.get(s, 0) + 1
                else:
                    deferred.append(idx)
            order, pending = [], deferred
            batches.append(batch)
            if not batch:
                break
        self.dropped = sum(len(b) for b in batches if len(b) < self.min_batch)
        return [b for b in batches if len(b) >= self.min_batch]

    def __iter__(self):
        yield from self._batches()

    def __len__(self):
        return int(np.ceil(len(self.storm_of) / self.batch_size))


def _backbone(pretrained=True):
    from torchvision.models import ResNet34_Weights, resnet34
    net = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
    net.fc = nn.Identity()  # 512-d pooled features
    return net


class EmbeddingNet(nn.Module):
    """ResNet-34 -> Linear 512->256 -> ReLU -> Linear 256->256 -> L2 normalization (arm A)."""

    def __init__(self, pretrained=True, dim=256):
        super().__init__()
        self.backbone = _backbone(pretrained)
        self.head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, dim))

    def forward(self, x):
        z = self.head(self.backbone(normalize_input(x)))
        return z / z.norm(dim=1, keepdim=True).clamp_min(1e-8)


class RegressionNet(nn.Module):
    """ResNet-34 -> Linear 512->256 -> ReLU -> Linear 256->1 (arm B; predicts the z-scored intensity)."""

    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = _backbone(pretrained)
        self.head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 1))

    def forward(self, x):
        return self.head(self.backbone(normalize_input(x))).squeeze(1)


class FrozenFeatures(nn.Module):
    """ImageNet ResNet-34 pooled 512-d features, no training (arm C)."""

    def __init__(self):
        super().__init__()
        self.backbone = _backbone(True).eval()

    @torch.no_grad()
    def forward(self, x):
        return self.backbone(normalize_input(x))


def normalize_input(x):
    return (x - IMAGENET_MEAN.to(x.device, x.dtype)) / IMAGENET_STD.to(x.device, x.dtype)


def knn_predict(query, db, db_labels, db_storms, k, eps=1e-6, chunk=512):
    """Inverse-distance-weighted kNN on Euclidean distance, at most one database window per storm.

    For each query: sort the database by distance, keep each storm's nearest window only, take the k nearest of
    those, and predict sum(w * label) / sum(w) with w = 1 / (d + eps). Returns (pred [Q], neighbor distances
    [Q, k]). Ties in distance are broken by database order.
    """
    query = torch.as_tensor(query, dtype=torch.float32)
    db = torch.as_tensor(db, dtype=torch.float32, device=query.device)
    labels = np.asarray(db_labels, dtype=np.float64)
    storms = np.asarray(db_storms)
    _, storm_code = np.unique(storms, return_inverse=True)
    n_storms = storm_code.max() + 1
    if k > n_storms:
        raise ValueError(f"k={k} but the database holds only {n_storms} storms")
    preds, dists = [], []
    for i in range(0, len(query), chunk):
        d = torch.cdist(query[i:i + chunk], db).cpu().numpy()
        order = np.argsort(d, axis=1, kind="stable")
        for row, o in zip(d, order):
            codes = storm_code[o]
            _, first = np.unique(codes, return_index=True)  # first (nearest) position of each storm in `o`
            nearest = o[np.sort(first)[:k]]
            w = 1.0 / (row[nearest] + eps)
            preds.append(float((w * labels[nearest]).sum() / w.sum()))
            dists.append(row[nearest])
    return np.asarray(preds), np.asarray(dists)


def metrics(pred, true):
    err = np.asarray(pred, float) - np.asarray(true, float)
    if len(err) == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan}
    return {"n": int(len(err)), "mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean())),
            "bias": float(err.mean())}


def metrics_by_category(pred, true):
    """Overall, per Saffir-Simpson category of the true intensity, and for true >= 113 kt (Category 4-5)."""
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    cat = category(true)
    out = {"all": metrics(pred, true)}
    for c in CATEGORIES:
        out[c] = metrics(pred[cat == c], true[cat == c])
    out[">=113"] = metrics(pred[true >= 113], true[true >= 113])
    return out
