"""Vyhodnoceni binarniho klasifikatoru: matice zamen 2x2 a odvozene metriky.

Predvyplnena infrastruktura -- metriky si studenti implementovali
v Cviceni 06, zde se jen pouzivaji. Konvence: trida 1 = pozitivni
(maligni), trida 0 = negativni (benigni).
"""

from __future__ import annotations

import numpy as np


def confusion_matrix_2x2(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Vrati matici zamen 2x2 ve stejnem usporadani jako ``sklearn.metrics.confusion_matrix``.

    Parametry
    ---------
    y_true:
        Skutecne tridy ``{0, 1}``, tvar ``(n,)``.
    y_pred:
        Predikovane tridy ``{0, 1}``, tvar ``(n,)``.

    Navratova hodnota
    -----------------
    ``np.ndarray`` tvaru ``(2, 2)``: radek = skutecna trida, sloupec =
    predikovana trida, tedy ``[[TN, FP], [FN, TP]]``.
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    y_pred = np.asarray(y_pred).ravel().astype(int)
    matrix = np.zeros((2, 2), dtype=int)
    for true_class in (0, 1):
        for pred_class in (0, 1):
            matrix[true_class, pred_class] = int(
                np.sum((y_true == true_class) & (y_pred == pred_class))
                )
    return matrix


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Vrati presnost (accuracy), preciznost (precision) a senzitivitu (recall).

    Definice
    --------
    accuracy  = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)      -- kolik z oznacenych jako maligni je skutecne maligni
    recall    = TP / (TP + FN)      -- kolik skutecne malignich model zachyti (senzitivita)

    Je-li jmenovatel nulovy, vraci se ``0.0`` (shoda se ``sklearn``
    s ``zero_division=0``).

    Parametry
    ---------
    y_true, y_pred:
        Skutecne a predikovane tridy ``{0, 1}``.

    Navratova hodnota
    -----------------
    ``dict`` s klici ``accuracy``, ``precision``, ``recall``.
    """
    (tn, fp), (fn, tp) = confusion_matrix_2x2(y_true, y_pred)
    total = tn + fp + fn + tp
    return {
        "accuracy": float((tp + tn) / total) if total else 0.0,
        "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
    }
