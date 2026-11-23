"""AAAC: Activation-Aware Adaptive Codebooks for 4-bit LLM weight quantization."""

from .assignment import assign_selection_groups, encode_indices, nearest_codeword
from .codebook_init import initial_codebooks, quantile_codebook, weighted_quantiles
from .config import NUM_CODEWORDS, NUM_TABLES, AAACConfig
from .fit import AAACLayerFit, fit_aaac_layer, fit_aaac_model
from .scaling import compute_group_scales, expand_group_scales, normalize_weight

__all__ = [
    "NUM_CODEWORDS",
    "NUM_TABLES",
    "AAACConfig",
    "AAACLayerFit",
    "assign_selection_groups",
    "compute_group_scales",
    "encode_indices",
    "expand_group_scales",
    "fit_aaac_layer",
    "fit_aaac_model",
    "initial_codebooks",
    "nearest_codeword",
    "normalize_weight",
    "quantile_codebook",
    "weighted_quantiles",
]
