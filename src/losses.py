"""Ztratove funkce (vzor Strategy): hodnota chyby a jeji gradient.

Ztratova funkce `L(prediction, target)` meri, jak moc se vystup site lisi
od cile. Uceni site je hledani vah, ktere `L` minimalizuji. Kazda ztrata
zde poskytuje dve veci:

- `forward` - hodnotu chyby (skalar), kterou sledujeme behem uceni,
- `gradient` - derivaci chyby podle vystupu site `dL/d(prediction)`;
  **tim startuje zpetne sireni chyby** (`Sequential.backward`).

Konvence (zafixovana pro cele cviceni): ztrata je **prumer** pres vsechny
prvky pole `prediction` (tj. pres vsechny vzorky davky). Gradient je presna
derivace tohoto prumeru, a proto obsahuje faktor `1 / N`, kde
`N = prediction.size`.

Dalsi vyskyt vzoru "abstraktni baze + zamenitelne implementace" v kurzu
(po `Distance`, `Initializer`, `Validator`, `Kernel`, `Activation`;
v tomto cviceni soucasne s `WeightsInitializer`).
Konkretni ztratu vybira `config.yaml` (klic `loss`) pres tovarni funkci
`make_loss` (Strategy + Factory, kap. 8c konfiguratoru).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Loss(ABC):
    """Abstraktni zaklad ztratovych funkci (plocha ABC hierarchie).

    Potomci sdileji jen rozhrani `forward` a `gradient`, ne kod. Instance
    se injektuje do `Trainer` (dependency injection) - `Trainer` nevi, kterou
    ztratu optimalizuje, jen vola jeji metody.
    """

    @abstractmethod
    def forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        """Vrati hodnotu ztraty (prumer pres vsechny prvky).

        Parametry
        ---------
        prediction : np.ndarray
            Vystup site, tvar `(n_vzorku, n_vystupu)`.
        target : np.ndarray
            Cilove hodnoty, stejny tvar jako `prediction`.

        Navratova hodnota
        -----------------
        float
            Nezaporna hodnota ztraty.
        """

    @abstractmethod
    def gradient(self, prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Vrati gradient ztraty podle vystupu site `dL/d(prediction)`.

        Parametry
        ---------
        prediction : np.ndarray
            Vystup site, tvar `(n_vzorku, n_vystupu)`.
        target : np.ndarray
            Cilove hodnoty, stejny tvar jako `prediction`.

        Navratova hodnota
        -----------------
        np.ndarray
            Pole stejneho tvaru jako `prediction`; vstupuje do
            `Sequential.backward`.
        """

    def __call__(self, prediction: np.ndarray, target: np.ndarray) -> float:
        """Zkratka pro `forward` - umoznuje volat instanci jako funkci.

        Parametry
        ---------
        prediction : np.ndarray
            Vystup site.
        target : np.ndarray
            Cilove hodnoty.

        Navratova hodnota
        -----------------
        float
            Vysledek `self.forward(prediction, target)`.
        """
        return self.forward(prediction, target)


class MSE(Loss):
    """Stredni kvadraticka chyba (Mean Squared Error).

    Intuitivni ztrata "prumerny ctverec odchylky". U klasifikace se
    sigmoidovym vystupem ma nevyhodu: jeji gradient podle `z` obsahuje
    derivaci sigmoidy, ktera pri saturaci (vystup blizko 0 nebo 1) mizi -
    i hrube chybna, ale "presvedcena" predikce se pak uci jen velmi pomalu.
    """

    def forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        """Vypocte stredni kvadratickou chybu.

        Definice
        --------
        L = (1 / N) * sum_i (target_i - prediction_i)^2,   N = prediction.size

        Parametry
        ---------
        prediction : np.ndarray
            Vystup site, tvar `(n_vzorku, n_vystupu)`.
        target : np.ndarray
            Cilove hodnoty, stejny tvar jako `prediction`.

        Navratova hodnota
        -----------------
        float
            Hodnota MSE (python `float`).
        """
        # assert  Ověřte, že prediction a target maji stejny tvar
        raise NotImplementedError(
            "Úkol: vratte float(np.mean((target - prediction) ** 2))."
        )

    def gradient(self, prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Vypocte gradient MSE podle vystupu site.

        Definice
        --------
        dL/d(prediction_i) = (2 / N) * (prediction_i - target_i),   N = prediction.size

        Znamenko: je-li predikce vetsi nez cil, gradient je kladny (zvyseni
        predikce by chybu zvetsilo) - gradientni sestup ji proto snizi.

        Parametry
        ---------
        prediction : np.ndarray
            Vystup site, tvar `(n_vzorku, n_vystupu)`.
        target : np.ndarray
            Cilove hodnoty, stejny tvar jako `prediction`.

        Navratova hodnota
        -----------------
        np.ndarray
            Pole stejneho tvaru jako `prediction`.
        """
        # assert  Ověřte, že prediction a target maji stejny tvar
        raise NotImplementedError(
            "Úkol: vratte 2.0 * (prediction - target) / prediction.size."
        )


class BCE(Loss):
    """Binarni krizova entropie (Binary Cross-Entropy).

    Prirozena ztrata pro binarni klasifikaci s pravdepodobnostnim vystupem
    v intervalu (0, 1). Za "presvedcene spatnou" predikci (napr. 0.001 pro
    cil 1) tresta velmi silne (`-log(0.001) ~ 6.9`). Ve spojeni se sigmoidou
    se derivace sigmoidy v gradientu podle `z` vykrati, takze uceni se pri
    saturaci nezastavi (viz README, Teoreticky zaklad).

    Atributy
    --------
    eps : float
        Mez pro oriznuti predikci do intervalu `[eps, 1 - eps]`, aby nikdy
        nevznikl `log(0)` ani deleni nulou.
    """

    eps: float = 1e-12

    def forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        """Vypocte binarni krizovou entropii.

        Definice
        --------
        p = clip(prediction, eps, 1 - eps)
        L = -(1 / N) * sum_i [ target_i * log(p_i) + (1 - target_i) * log(1 - p_i) ]

        Parametry
        ---------
        prediction : np.ndarray
            Vystup site (pravdepodobnost tridy 1), tvar `(n_vzorku, n_vystupu)`.
        target : np.ndarray
            Cilove tridy 0/1, stejny tvar jako `prediction`.

        Navratova hodnota
        -----------------
        float
            Hodnota BCE (python `float`), vzdy konecna diky oriznuti.
        """
        # assert  Ověřte, že prediction a target maji stejny tvar
        # assert  Ověřte, že hodnoty target lezi v intervalu [0, 1]
        raise NotImplementedError(
            "Úkol: oriznete p = np.clip(prediction, self.eps, 1 - self.eps) a vratte "
            "float(-np.mean(target * np.log(p) + (1 - target) * np.log(1 - p)))."
        )

    def gradient(self, prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Vypocte gradient BCE podle vystupu site.

        Definice
        --------
        p = clip(prediction, eps, 1 - eps)
        dL/d(prediction_i) = (1 / N) * (p_i - target_i) / (p_i * (1 - p_i))

        Jmenovatel `p * (1 - p)` je (az na faktor 1/T) prave derivace
        sigmoidy - pri zpetnem pruchodu se s ni vykrati.

        Parametry
        ---------
        prediction : np.ndarray
            Vystup site, tvar `(n_vzorku, n_vystupu)`.
        target : np.ndarray
            Cilove tridy 0/1, stejny tvar jako `prediction`.

        Navratova hodnota
        -----------------
        np.ndarray
            Pole stejneho tvaru jako `prediction`.
        """
        # assert  Ověřte, že prediction a target maji stejny tvar
        raise NotImplementedError(
            "Úkol: oriznete p = np.clip(prediction, self.eps, 1 - self.eps) a vratte "
            "(p - target) / (p * (1 - p)) / prediction.size."
        )


# Registr jmen ztrat (klic `loss` v config.yaml) na tridy.
_LOSS_REGISTRY: dict[str, type[Loss]] = {
    "mse": MSE,
    "bce": BCE,
}


def make_loss(name: str) -> Loss:
    """Tovarni funkce: vytvori ztratu podle jmena z `config.yaml` (PREDVYPLNENO).

    Parametry
    ---------
    name : str
        Jmeno ztraty, jedno z `"mse"`, `"bce"`.

    Navratova hodnota
    -----------------
    Loss
        Nova instance prislusne tridy.

    Vyjimky
    -------
    ``ValueError``:
        Pokud `name` neni v `_LOSS_REGISTRY`.
    """
    if name not in _LOSS_REGISTRY:
        povolene = ", ".join(sorted(_LOSS_REGISTRY))
        raise ValueError(f"Neznama ztratova funkce: {name!r}. Povolene hodnoty jsou: {povolene}.")
    return _LOSS_REGISTRY[name]()
