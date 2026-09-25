"""Evaluation suite: §25-32 of the proposal, one module per metric family."""

from .field_metrics import (HorizonMetrics, predictability_horizon,
                            crossover_horizon)
from .spectra import radial_spectrum, spectral_error, SpectrumAccumulator
from .combustion_metrics import (CombustionMetrics, gradient_error,
                                 reaction_zone_metrics, reaction_mask,
                                 global_quantities)
from .semigroup_metrics import (evaluate_semigroup, semigroup_consistency,
                                identity_consistency, DEFAULT_TAU_PAIRS)
from .timing import (time_callable, benchmark_direct, benchmark_ar, gpu_state)
from .runner import (Predictor, DirectPredictor, ARPredictor,
                     BaselinePredictor, build_predictor, run_horizon_eval)

__all__ = [
    "Predictor", "DirectPredictor", "ARPredictor", "BaselinePredictor",
    "build_predictor", "run_horizon_eval",
    "HorizonMetrics", "predictability_horizon", "crossover_horizon",
    "radial_spectrum", "spectral_error", "SpectrumAccumulator",
    "CombustionMetrics", "gradient_error", "reaction_zone_metrics",
    "reaction_mask", "global_quantities",
    "evaluate_semigroup", "semigroup_consistency", "identity_consistency",
    "DEFAULT_TAU_PAIRS",
    "time_callable", "benchmark_direct", "benchmark_ar", "gpu_state",
]
