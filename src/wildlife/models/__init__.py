"""Model architectures, temperature calibration, OOD rejection, and Gemini 4 Argon integration."""

from wildlife.models.argon import (
    ARGON_MODEL_ID,
    ArgonMultimodalAssistant,
    build_interpretable_explanation,
)
from wildlife.models.calibration import (
    TemperatureScaler,
    calibrated_softmax,
    compute_brier_score,
    compute_ece,
    compute_nll,
    compute_reliability_bins,
)
from wildlife.models.classifier import (
    CheckpointClassOrderError,
    WildlifeClassifier,
    compute_classification_metrics,
    load_checkpoint,
    save_checkpoint,
)
from wildlife.models.unknown import (
    UnknownSpeciesGate,
    compute_energy_score,
)

__all__ = [
    "ARGON_MODEL_ID",
    "ArgonMultimodalAssistant",
    "build_interpretable_explanation",
    "TemperatureScaler",
    "calibrated_softmax",
    "compute_brier_score",
    "compute_ece",
    "compute_nll",
    "compute_reliability_bins",
    "CheckpointClassOrderError",
    "WildlifeClassifier",
    "compute_classification_metrics",
    "load_checkpoint",
    "save_checkpoint",
    "UnknownSpeciesGate",
    "compute_energy_score",
]
