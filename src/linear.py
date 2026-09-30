"""Afinni transformace (vazeny soucet vstupu plus volitelny bias) bez aktivace.

BRANA+ (continuity gate s rozsirenim): Do metody `forward` vlozte sve reseni
`src/linear.py` z Cviceni 09 (signatura je totozna). NEPREPISUJTE cely soubor
najednou - v Cviceni 10 pribyla metoda `update`, ktera je novym ukolem.

`Linear` pocita `z = x @ weights + bias` a nic vic - je to samostatna, na
aktivaci nezavisla vrstva (stejny motiv jako `nn.Linear` v PyTorch).
`Neuron` (modul `src.neuron`) `Linear` sklada s injektovanou `Activation`.

Konvence tvaru (shodna s Cvicenim 09): **matice vah** tvaru
`(n_vstupu, n_jednotek)` - sloupec `j` jsou vahy `j`-teho neuronu - a
**vektor biasu** tvaru `(n_jednotek,)`.

NOVE v Cviceni 10: vahy prestavaji byt konstantou navrzenou rucne a stavaji
se **menitelnym stavem** vrstvy. Metoda `update` provede jeden krok
gradientniho sestupu `W <- W - learning_rate * dL/dW` (a obdobne pro bias).
Gradient sama nepocita - dostane ho hotovy od `Sequential.backward`.
"""

from __future__ import annotations

import numpy as np


class Linear:
    """Afinni vrstva: `z = x @ weights + bias`, bez aktivace.

    Bias je samostatny, volitelny, PyTorch-style parametr (NE stara "-1
    augmentace" vstupniho vektoru): `bias=None` znamena bez biasu (jako
    PyTorch `bias=False`), pole (nebo float) jej zapina.

    Atributy
    --------
    weights : np.ndarray
        Matice vah `(n_vstupu, n_jednotek)`; v Cviceni 10 se meni metodou `update`.
    bias : np.ndarray | float | None
        Vektor biasu `(n_jednotek,)`, nebo `None` (vrstva bez biasu).
    """

    def __init__(self, weights: np.ndarray, bias: np.ndarray | float | None = None) -> None:
        """Ulozi vahy a volitelny bias.

        Parametry
        ---------
        weights : np.ndarray
            Matice vah tvaru `(n_vstupu, n_jednotek)`.
        bias : np.ndarray | float | None, optional
            Vektor biasu tvaru `(n_jednotek,)` (pripadne skalar pro jediny
            neuron). `None` (vychozi) znamena bez biasu.
        """
        self.weights = weights
        self.bias = bias

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Provede dopredny pruchod afinni vrstvou pres davku vstupu.

        Definice
        --------
        z = x @ self.weights + (self.bias, pokud self.bias is not None, jinak 0.0)

        Parametry
        ---------
        x : np.ndarray
            Matice vstupnich vzorku tvaru `(n_vzorku, n_vstupu)`, kde
            `n_vstupu` odpovida prvnimu rozmeru `self.weights`.

        Navratova hodnota
        -----------------
        np.ndarray
            Pole tvaru `(n_vzorku, n_jednotek)`.
        """
        # assert  Ověřte, že x je 2D pole (ma atribut ndim == 2)
        # assert  Ověřte, že x.shape[1] odpovida prvnimu rozmeru self.weights
        raise NotImplementedError(
            "Úkol: (brana z Cviceni 09) spocitejte z = x @ self.weights + "
            "(self.bias if self.bias is not None else 0.0) a vratte ho - zadnou "
            "aktivaci zde neaplikujte."
        )

    def update(
        self,
        weight_gradient: np.ndarray,
        bias_gradient: np.ndarray | float | None,
        learning_rate: float,
        ) -> None:
        """Provede jeden krok gradientniho sestupu na vahach a biasu (NOVE v Cviceni 10).

        Definice
        --------
        W <- W - learning_rate * weight_gradient
        b <- b - learning_rate * bias_gradient      (jen pokud self.bias is not None)

        Znamenkova konvence (zafixovana pro cele cviceni): `weight_gradient`
        je **skutecny gradient** ztraty `dL/dW` (smer nejstrmejsiho RUSTU
        chyby), proto se odecita - krok vede proti gradientu, tedy k mensi
        chybe. Stejnou konvenci dodrzuje `Sequential.backward`, ktery
        gradienty pocita.

        Metoda nic nevraci - meni stav vrstvy (`self.weights`, `self.bias`).
        Vrstva bez biasu (`self.bias is None`) bias nema a `bias_gradient`
        je v tom pripade `None`.

        Parametry
        ---------
        weight_gradient : np.ndarray
            Gradient `dL/dW`, stejny tvar jako `self.weights`.
        bias_gradient : np.ndarray | float | None
            Gradient `dL/db`, stejny tvar jako `self.bias`; `None`, pokud
            vrstva bias nema.
        learning_rate : float
            Kladna delka kroku gradientniho sestupu.
        """
        # assert  Ověřte, že weight_gradient ma stejny tvar jako self.weights
        # assert  Ověřte, že learning_rate je kladne cislo
        raise NotImplementedError(
            "Úkol: aktualizujte self.weights = self.weights - learning_rate * "
            "weight_gradient; pokud self.bias is not None, obdobne "
            "self.bias = self.bias - learning_rate * bias_gradient."
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Zkratka pro `forward` - umoznuje volat instanci jako funkci.

        Parametry
        ---------
        x : np.ndarray
            Matice vstupnich vzorku predana primo do `forward`.

        Navratova hodnota
        -----------------
        np.ndarray
            Vysledek `self.forward(x)`.
        """
        return self.forward(x)
