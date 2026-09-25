"""Losses: prediction (§18), semigroup (§16), identity (§17). Nothing else (§35)."""

from .prediction import (PredictionLoss, normalized_mse, channel_relative_l2,
                         relative_l2)
from .semigroup import semigroup_loss, SemigroupWeight, LatentCollapseMonitor
from .identity import identity_loss

__all__ = ["PredictionLoss", "normalized_mse", "channel_relative_l2",
           "relative_l2", "semigroup_loss", "SemigroupWeight",
           "LatentCollapseMonitor", "identity_loss"]
