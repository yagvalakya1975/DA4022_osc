"""Weighted Lloyd updates and BF16 finalisation -- OWNED BY MEMBER 2 (part B).

The two functions below are the only pieces of Algorithm 1 that are not yet
implemented. ``fit_aaac_layer`` (in ``fit.py``) already calls them with the
exact signatures shown, so replacing the ``raise NotImplementedError`` bodies is
all that is needed to make the full fit run. See ``README.md`` in this folder,
section "Hand-off to Member 2 (part B)", for the full contract and the tests to
add.
"""

from __future__ import annotations

import torch


def weighted_lloyd_update(
    normalized: torch.Tensor,
    importance: torch.Tensor,
    codebooks: torch.Tensor,
    selectors: torch.Tensor,
    selection_size: int,
    *,
    chunk_elements: int,
) -> torch.Tensor:
    """One inner iteration of Algorithm 1 (weighted Lloyd step). TODO(Member 2B).

    Args:
        normalized:  float32 ``[out, in]`` normalised weights ``u = W / scale``.
        importance:  float32 ``[in]`` non-negative column importance ``I[k]``
                     (same weight for every output row).
        codebooks:   float32 ``[2, 16]`` current tables. Do not modify in place.
        selectors:   bool ``[out, in // S]``; ``True`` means the group uses table 1.
        selection_size: S.
        chunk_elements: memory bound to pass on to ``encode_indices``.

    Must return a NEW float32 ``[2, 16]`` tensor where, for every table ``c`` and
    entry ``j``::

        cell(c, j) = { (o, k) : group of (o, k) uses table c and
                                encode_indices(...)[o, k] == j }
        new[c, j]  = sum_{cell} I[k] * u[o, k] / sum_{cell} I[k]   if sum_{cell} I[k] > 0
        new[c, j]  = codebooks[c, j]                                 otherwise (empty cell: keep)

    Must be deterministic (accumulate in float64; avoid CUDA atomics such as
    ``index_add_`` / ``scatter_add_`` unless running on CPU).
    """
    raise NotImplementedError(
        "weighted_lloyd_update is Member 2 (part B)'s task; see "
        "Quantization_and_Pruning/quantization/aaac/README.md"
    )


def finalize_codebooks(codebooks: torch.Tensor, codebook_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Round the fitted float32 ``[2, 16]`` tables to ``codebook_dtype`` (BF16). TODO(Member 2B).

    Must return a new ``[2, 16]`` tensor of dtype ``codebook_dtype``.
    ``fit_aaac_layer`` re-runs group assignment and index encoding with these
    rounded tables afterwards, so the stored indices always match the stored
    tables.
    """
    raise NotImplementedError(
        "finalize_codebooks is Member 2 (part B)'s task; see "
        "Quantization_and_Pruning/quantization/aaac/README.md"
    )
