"""Verejne API balicku ``dataio`` pro cviceni 10.

Nacteni datasetu Breast Cancer Wisconsin, typovana konfigurace, metriky
binarni klasifikace a vykreslovani vysledku uceni.

Verejne API
-----------
- ``load_breast_cancer_data`` -- ``(x_train, y_train, x_test, y_test)``, standardizovane, 1 = maligni
- ``load_config`` / ``validate_config`` -- typovana konfigurace nad ``config.yaml``
- ``DataConfig`` / ``NetworkConfig`` / ``TrainingConfig`` / ``ExperimentConfig`` -- dataclassy
- ``confusion_matrix_2x2`` / ``classification_metrics`` -- matice zamen a metriky
- ``plot_loss_curve`` -- krivka uceni (trenovaci a testovaci ztrata, krok uceni)
- ``plot_confusion_matrix`` -- matice zamen 2x2 s metrikami

Cely balicek ``dataio/`` je v tomto cviceni **predvyplneny** -- zadny
studentsky ukol, zadny ``NotImplementedError``.

**Tento soubor neupravujte.**
"""

from __future__ import annotations

from dataio.config_manager import (
    DataConfig,
    ExperimentConfig,
    NetworkConfig,
    TrainingConfig,
    load_config,
    validate_config,
)
from dataio.loader import load_breast_cancer_data
from dataio.metrics import classification_metrics, confusion_matrix_2x2
from dataio.plotting import plot_confusion_matrix, plot_loss_curve

__all__ = [
    "load_breast_cancer_data",
    "load_config",
    "validate_config",
    "DataConfig",
    "NetworkConfig",
    "TrainingConfig",
    "ExperimentConfig",
    "confusion_matrix_2x2",
    "classification_metrics",
    "plot_loss_curve",
    "plot_confusion_matrix",
]
