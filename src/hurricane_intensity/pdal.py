"""Pairwise Distance Alignment Loss (PDAL), as defined in Celik, Korkmaz and Ozdag, "PDAL: A Geometry-Preserving
Loss for CNN-Based Location Recognition", IGARSS 2026 (docs/references/PDAL.pdf, not committed).

For a minibatch of m samples with embeddings F (m x k) and labels C (m x d), with
    dF_ij = ||F_i - F_j||_2,  dC_ij = ||C_i - C_j||_2,  i, j = 1..m   (paper eq. 2)
the three variants are
    plain            (1/m^2) sum_ij (dF_ij - dC_ij)^2                                   (eq. 1)
    absolute         (1/m^2) sum_ij |dF_ij - dC_ij|                                     (eq. 5)
    scale_invariant  exp(-cosSim(dF, dC)),  cosSim = dF . dC / (max(||dF||, eps) max(||dC||, eps)),
                     eps = 1e-8, over the vectorized m x m distance matrices              (eqs. 3-4)
The sums run over all m^2 pairs, the diagonal included (it is zero in both matrices), as in the paper.

The paper applies the loss to normalized embeddings and labels. Here the caller passes F already L2-normalized
(the model head ends in L2 normalization, paper section VII) and C standardized (this project z-scores labels with
the train-set mean and std; the paper min-max scales coordinates). This module does not normalize either.

Distances are computed as sqrt(clamp(squared, 1e-12)) with the diagonal zeroed, not with torch.cdist, whose
backward is NaN at zero distance (the diagonal, or two identical embeddings).
"""
from __future__ import annotations

import torch

EPS = 1e-8
VARIANTS = ("plain", "absolute", "scale_invariant")


def pairwise_distances(x):
    """[m, k] -> [m, m] Euclidean distances, diagonal exactly 0, finite gradients everywhere."""
    if x.dim() == 1:
        x = x[:, None]
    sq = (x[:, None, :] - x[None, :, :]).pow(2).sum(-1)
    off_diag = 1.0 - torch.eye(x.shape[0], dtype=x.dtype, device=x.device)
    return torch.sqrt(sq.clamp_min(1e-12)) * off_diag


def pdal(F, C):
    """Plain PDAL, paper eq. 1."""
    m = F.shape[0]
    return (pairwise_distances(F) - pairwise_distances(C)).pow(2).sum() / m ** 2


def pdal_absolute(F, C):
    """Absolute-difference PDAL, paper eq. 5."""
    m = F.shape[0]
    return (pairwise_distances(F) - pairwise_distances(C)).abs().sum() / m ** 2


def pdal_scale_invariant(F, C):
    """Scale-invariant PDAL, paper eqs. 3-4: exp(-cosine similarity of the two distance configurations)."""
    dF = pairwise_distances(F).reshape(-1)
    dC = pairwise_distances(C).reshape(-1)
    cos = (dF * dC).sum() / (dF.norm().clamp_min(EPS) * dC.norm().clamp_min(EPS))
    return torch.exp(-cos)


def pdal_loss(F, C, variant="scale_invariant"):
    """Dispatch by variant name ("plain", "absolute" or "scale_invariant")."""
    fn = {"plain": pdal, "absolute": pdal_absolute, "scale_invariant": pdal_scale_invariant}.get(variant)
    if fn is None:
        raise ValueError(f"unknown PDAL variant {variant!r}; expected one of {VARIANTS}")
    return fn(F, C)
