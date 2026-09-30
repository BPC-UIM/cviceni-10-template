"""Typovana sprava konfigurace nad ``config.yaml`` pro cviceni 10.

Modul definuje vnorene dataclassy odpovidajici sekcim ``config.yaml`` a dve
funkce: ``load_config`` (naparsuje YAML, sestavi dataclassy, zvaliduje a
vrati) a ``validate_config`` (rozsahove kontroly s ceskymi chybovymi
hlaskami). Cely modul je predvyplneny -- tovarni funkce, ktere podle jmen
z konfigurace vytvari inicializator a ztratu (``make_weights_initializer``,
``make_loss``), jsou rovnez predvyplnene a ziji v ``src``.

K hodnotam se pristupuje pres atributy (napr. ``cfg.training.learning_rate``),
nikdy ne pres klice slovniku -- preklep v atributu odhali editor/typovy
kontroler staticky, zatimco ``cfg["training"]["learning_rate"]`` spadne az
za behu.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import yaml

ALLOWED_INITIALIZERS: tuple[str, ...] = ("random_uniform", "xavier", "he", "pytorch_default")
ALLOWED_LOSSES: tuple[str, ...] = ("mse", "bce")


@dataclass
class DataConfig:
    """Nastaveni dat (sekce ``data``)."""

    test_size: float
    random_state: int | None


@dataclass
class NetworkConfig:
    """Architektura site (sekce ``network``).

    Atributy
    --------
    hidden_units:
        Pocty jednotek skrytych vrstev, napr. ``[9, 4]`` pro sit 30-9-4-1.
    initializer:
        Jmeno inicializatoru vah, jedno z ``ALLOWED_INITIALIZERS``.
    sigmoid_temperature:
        Teplota sigmoidy ve vsech vrstvach.
    """

    hidden_units: list[int]
    initializer: str
    sigmoid_temperature: float


@dataclass
class TrainingConfig:
    """Hyperparametry uceni (sekce ``training``) -- predavaji se tride ``Trainer``."""

    learning_rate: float
    epochs: int
    batch_size: int
    max_loss: float
    lr_decay: float
    lr_decay_every: int


@dataclass
class ExperimentConfig:
    """Korenova konfigurace experimentu slozena ze vsech dilcich sekci."""

    data: DataConfig
    network: NetworkConfig
    training: TrainingConfig
    loss: str


def load_config(filepath: str = "config.yaml") -> ExperimentConfig:
    """Nacte a zvaliduje konfiguraci z YAML souboru.

    Parametry
    ---------
    filepath:
        Cesta k YAML souboru s konfiguraci.

    Navratova hodnota
    -----------------
    ``ExperimentConfig`` s vnorenymi dataclassami ``DataConfig``,
    ``NetworkConfig`` a ``TrainingConfig`` a jmenem ztraty ``loss``.

    Vyjimky
    -------
    ``FileNotFoundError``:
        Pokud soubor neexistuje.
    ``ValueError``:
        Pokud chybi sekce/klic, nebo hodnota nesplnuje kontroly
        ve ``validate_config``.
    """
    with open(filepath, "r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)

    try:
        seed_raw = raw["data"]["random_state"]
        cfg = ExperimentConfig(
            data=DataConfig(
                test_size=float(raw["data"]["test_size"]),
                random_state=int(seed_raw) if seed_raw is not None else None,
            ),
            network=NetworkConfig(
                hidden_units=[int(k) for k in raw["network"]["hidden_units"]],
                initializer=str(raw["network"]["initializer"]),
                sigmoid_temperature=float(raw["network"]["sigmoid_temperature"]),
            ),
            training=TrainingConfig(
                learning_rate=float(raw["training"]["learning_rate"]),
                epochs=int(raw["training"]["epochs"]),
                batch_size=int(raw["training"]["batch_size"]),
                max_loss=float(raw["training"]["max_loss"]),
                lr_decay=float(raw["training"]["lr_decay"]),
                lr_decay_every=int(raw["training"]["lr_decay_every"]),
            ),
            loss=str(raw["loss"]),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"config.yaml nema ocekavanou strukturu (chybi {exc})") from exc

    validate_config(cfg)
    return cfg


def validate_config(cfg: ExperimentConfig) -> None:
    """Zkontroluje rozsahy hodnot v konfiguraci.

    Pri poruseni nektere podminky vyhodi ``ValueError`` se srozumitelnou
    ceskou hlaskou obsahujici zadanou hodnotu. Kontroluji se:

    - ``0 < data.test_size < 1``
    - ``network.hidden_units`` -- kazda polozka ``>= 1``
    - ``network.initializer`` z ``ALLOWED_INITIALIZERS``
    - ``network.sigmoid_temperature > 0``
    - ``training.learning_rate > 0``, ``training.epochs >= 1``,
      ``training.batch_size >= 1``, ``training.max_loss >= 0``,
      ``0 < training.lr_decay <= 1``, ``training.lr_decay_every >= 1``
    - ``loss`` z ``ALLOWED_LOSSES``

    Navratova hodnota je ``None`` -- funkce pouze validuje.
    """
    if not 0.0 < cfg.data.test_size < 1.0:
        raise ValueError(
            f"data.test_size musi lezet v intervalu (0, 1), zadano: {cfg.data.test_size}"
            )

    if any(k < 1 for k in cfg.network.hidden_units):
        raise ValueError(
            f"network.hidden_units musi obsahovat jen cela cisla >= 1, zadano: {
                cfg.network.hidden_units
                }"
        )
    if cfg.network.initializer not in ALLOWED_INITIALIZERS:
        raise ValueError(
            f"network.initializer musi byt jedno z {ALLOWED_INITIALIZERS}, "
            f"zadano: {cfg.network.initializer!r}"
        )
    if cfg.network.sigmoid_temperature <= 0:
        raise ValueError(
            f"network.sigmoid_temperature musi byt > 0, zadano: {cfg.network.sigmoid_temperature}"
        )

    tr = cfg.training
    if tr.learning_rate <= 0:
        raise ValueError(f"training.learning_rate musi byt > 0, zadano: {tr.learning_rate}")
    if tr.epochs < 1:
        raise ValueError(f"training.epochs musi byt >= 1, zadano: {tr.epochs}")
    if tr.batch_size < 1:
        raise ValueError(f"training.batch_size musi byt >= 1, zadano: {tr.batch_size}")
    if tr.max_loss < 0:
        raise ValueError(f"training.max_loss musi byt >= 0, zadano: {tr.max_loss}")
    if not 0.0 < tr.lr_decay <= 1.0:
        raise ValueError(f"training.lr_decay musi lezet v intervalu (0, 1], zadano: {tr.lr_decay}")
    if tr.lr_decay_every < 1:
        raise ValueError(f"training.lr_decay_every musi byt >= 1, zadano: {tr.lr_decay_every}")

    if cfg.loss not in ALLOWED_LOSSES:
        raise ValueError(f"loss musi byt jedno z {ALLOWED_LOSSES}, zadano: {cfg.loss!r}")
