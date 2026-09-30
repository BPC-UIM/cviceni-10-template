"""Nacitani datasetu Breast Cancer Wisconsin pro binarni klasifikaci (cviceni 10).

Na rozdil od Cviceni 08 (jediny priznak, 1D logisticka regrese) se zde
pouziva **vsech 30 priznaku** -- navaznost na Cviceni 05-07. Data se rozdeli
na trenovaci a testovaci cast a standardizuji.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def load_breast_cancer_data(
    random_state: int | None = 42,
    test_size: float = 0.2,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Nacte dataset, rozdeli ho na trenovaci/testovaci cast a standardizuje.

    Parametry
    ---------
    random_state:
        Seed rozdeleni dat (reprodukovatelnost).
    test_size:
        Podil testovacich vzorku, vychozi ``0.2``.

    Navratova hodnota
    -----------------
    x_train, y_train, x_test, y_test:
        ``x_*`` tvaru ``(n, 30)`` typu ``float64`` (standardizovane),
        ``y_*`` tvaru ``(n,)`` typu ``int64`` s hodnotami ``{0, 1}``.

    Kodovani cilove promenne
    ------------------------
    ``sklearn`` koduje ``target`` jako 0 = malignant, 1 = benign. Kurz
    pracuje s konvenci **1 = maligni (zhoubny)**, 0 = benigni (shoda
    s Cvicenimi 05-08), proto se stitky prohazuji: ``y = 1 - target``.
    Vystup site (sigmoida) je tak odhadem pravdepodobnosti malignity.

    Standardizace
    -------------
    Kazdy priznak se prevede na nulovy prumer a jednotkovou smerodatnou
    odchylku. Prumer a odchylka se pocitaji **jen z trenovacich dat**
    a stejne se pouziji na testovaci data -- jinak by informace z testovaci
    casti "unikla" do uceni. Bez standardizace by priznaky s velkymi
    hodnotami (napr. ``mean area`` ~ 10^3) posunuly ``z`` daleko od nuly
    a sigmoida by saturovala uz v prvni vrstve.

    Rozdeleni je **stratifikovane** -- pomer trid je v obou castech stejny.
    """
    dataset = load_breast_cancer()
    x = np.asarray(dataset.data, dtype=np.float64)
    y = 1 - np.asarray(dataset.target, dtype=np.int64)   # 1 = maligni

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    return x_train, y_train, x_test, y_test
