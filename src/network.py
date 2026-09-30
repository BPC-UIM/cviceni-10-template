"""Vicevrstva sit jako retezeni vrstev - s doprednym i zpetnym pruchodem.

BRANA+ (continuity gate s rozsirenim): Metodu `forward` znate z Cviceni 09;
v Cviceni 10 ji **rozsirte** o ukladani predaktivacnich hodnot `z_`.
Metody `backward` a `update` jsou novym ukolem - jadro celeho cviceni
(zpetne sireni chyby, backpropagation).

`Sequential` drzi seznam vrstev (kompozice, ne dedicnost). Vrstvou je
`Neuron` (modul `src.neuron`) se slozkami `linear` (matice vah
`(n_vstupu, n_jednotek)`, vektor biasu `(n_jednotek,)`) a `activation`.

Konvence (zafixovane pro cele cviceni, vrstvy cislovane l = 1, ..., L;
v Pythonu `layers[l-1]`, `z_[l-1]`):

    io_[0] = x                                   (n_vzorku, n_vstupu)
    z_l    = io_[l-1] @ W_l + b_l                (n_vzorku, n_jednotek_l)
    io_[l] = f_l(z_l)                            (n_vzorku, n_jednotek_l)

    delta_L = dL/d(io_[L]) * f_L'(z_L)           ... * je nasobeni prvek po prvku
    delta_l = (delta_{l+1} @ W_{l+1}.T) * f_l'(z_l)
    dL/dW_l = io_[l-1].T @ delta_l               (n_vstupu_l, n_jednotek_l) = tvar W_l
    dL/db_l = soucet delta_l pres vzorky         (n_jednotek_l,)            = tvar b_l

`delta_l = dL/dz_l` je gradient chyby vzhledem k predaktivacni hodnote
vrstvy `l`. Gradienty jsou **skutecne gradienty** ztraty (smer rustu chyby);
`Linear.update` je proto odecita.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class Sequential:
    """Retezec vrstev `f_L( ... f_2(f_1(x)) ... )` s doprednym i zpetnym pruchodem.

    Obdoba `nn.Sequential` v PyTorch. Vrstvy jsou injektovane zvenku
    (dependency injection). Kazda vrstva musi mit atributy `linear`
    (instance `Linear`) a `activation` (instance `Activation`) - to splnuje
    `Neuron`.

    Model umi **jeden krok uceni**: `forward` (predikce + zaznam mezistavu),
    `backward` (gradienty vsech vah) a `update` (posun vah proti gradientu).
    Skladani kroku do procesu uceni (epochy, davky, michani) je ukolem
    tridy `Trainer` (modul `src.trainer`).

    Atributy
    --------
    layers : list
        Vrstvy v poradi pruchodu (typicky instance `Neuron`).
    io_ : list[np.ndarray] | None
        Stopa posledniho doprednego pruchodu `[x, a_1, ..., a_L]` (vstup a
        vystupy vrstev, jako v Cviceni 09); pred prvnim `forward` je `None`.
    z_ : list[np.ndarray] | None
        NOVE: predaktivacni hodnoty `[z_1, ..., z_L]` posledniho pruchodu
        (vystupy `Linear` jednotlivych vrstev); pred prvnim `forward` je `None`.
    """

    def __init__(self, layers: list) -> None:
        """Ulozi vrstvy site; mezistavy `io_` a `z_` zatim neexistuji.

        Parametry
        ---------
        layers : list
            Seznam vrstev (typicky `Neuron`) v poradi, v jakem jimi maji data
            projit.
        """
        self.layers: list[Any] = layers
        self.io_: list[np.ndarray] | None = None
        self.z_: list[np.ndarray] | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Provede dopredny pruchod vsemi vrstvami a zaznamena mezistavy.

        Definice
        --------
        io_[0] = x
        z      = layers[k].linear(io_[k])         pro k = 0, ..., L-1
        io_[k+1] = layers[k].activation(z)
        vystup = io_[L]

        Oproti Cviceni 09 se vrstva nevola jako celek (`layer(x)`), ale po
        slozkach: nejdriv `layer.linear`, pak `layer.activation`. Jen tak lze
        zachytit predaktivacni hodnotu `z`, kterou `Neuron.forward` nevraci
        a kterou `backward` potrebuje pro derivaci aktivace.

        Po skonceni drzi `self.io_` seznam delky `1 + L` a `self.z_` seznam
        delky `L`.

        Parametry
        ---------
        x : np.ndarray
            Matice vstupnich vzorku tvaru `(n_vzorku, n_vstupu)`.

        Navratova hodnota
        -----------------
        np.ndarray
            Vystup posledni vrstvy, tvaru `(n_vzorku, n_jednotek_posledni_vrstvy)`.
        """
        # assert  Ověřte, že seznam self.layers neni prazdny
        # assert  Ověřte, že x je 2D pole (n_vzorku, n_vstupu)
        raise NotImplementedError(
            "Úkol: (brana z Cviceni 09 + rozsireni) nastavte self.io_ = [x] a "
            "self.z_ = []; pro kazdou vrstvu spocitejte z = layer.linear(self.io_[-1]), "
            "pripojte z do self.z_ a layer.activation(z) do self.io_; vratte self.io_[-1]."
        )

    def backward(self, loss_gradient: np.ndarray) -> list[tuple[np.ndarray, np.ndarray | None]]:
        """Zpetne sireni chyby: spocita gradienty vah a biasu vsech vrstev (NOVE).

        Predpoklada, ze tesne predtim probehl `forward` na teze davce - pouziva
        jeho zaznam `self.io_` a `self.z_`.

        Algoritmus (retezove pravidlo, od posledni vrstvy k prvni)
        -----------------------------------------------------------
        1. `grad = loss_gradient` (tj. dL/d(io_[L]), gradient ztraty na vystupu).
        2. Pro vrstvy `k = L-1, ..., 0` (v obracenem poradi):
             delta = grad * layers[k].activation.derivative(z_[k])
             dW    = io_[k].T @ delta
             db    = delta.sum(axis=0)          (None, pokud vrstva nema bias)
             grad  = delta @ layers[k].linear.weights.T    (dL/d(io_[k]) pro vrstvu pred ni)
        3. Vratte seznam dvojic `(dW, db)` v **doprednem** poradi vrstev
           (prvni prvek patri `layers[0]`).

        `grad` pro predchozi vrstvu se pocita z vah **pred** aktualizaci -
        `backward` proto zadne vahy nemeni (to dela az `update`).

        Parametry
        ---------
        loss_gradient : np.ndarray
            Gradient ztraty vzhledem k vystupu site `dL/d(io_[L])`, tvar
            `(n_vzorku, n_jednotek_posledni_vrstvy)` - vraci ho `Loss.gradient`.

        Navratova hodnota
        -----------------
        list[tuple[np.ndarray, np.ndarray | None]]
            Seznam delky `L`; prvek `k` je `(dL/dW, dL/db)` vrstvy `layers[k]`
            ve stejnem tvaru jako jeji `weights` a `bias`.
        """
        # assert  Ověřte, že self.io_ a self.z_ nejsou None (nejdriv musi probehnout forward)
        # assert  Ověřte, že loss_gradient ma stejny tvar jako vystup site self.io_[-1]
        raise NotImplementedError(
            "Úkol: implementujte zpetne sireni chyby - od posledni vrstvy k prvni "
            "spocitejte delta = grad * activation.derivative(z), dW = io_[k].T @ delta, "
            "db = delta.sum(axis=0) a grad = delta @ W.T; vratte seznam (dW, db) "
            "v doprednem poradi vrstev."
        )

    def update(
        self,
        gradients: list[tuple[np.ndarray, np.ndarray | None]],
        learning_rate: float,
        ) -> None:
        """Aplikuje gradientni krok na vsechny vrstvy (NOVE v Cviceni 10).

        Sama zadnou vahu nemeni: pro kazdou vrstvu deleguje na
        `layer.linear.update(dW, db, learning_rate)`. Stav vah zije v `Linear`.

        Parametry
        ---------
        gradients : list[tuple[np.ndarray, np.ndarray | None]]
            Vystup `backward` - dvojice `(dW, db)` v doprednem poradi vrstev.
        learning_rate : float
            Kladna delka kroku gradientniho sestupu.
        """
        # assert  Ověřte, že pocet gradientu odpovida poctu vrstev
        raise NotImplementedError(
            "Úkol: pro kazdou dvojici (layer, (dW, db)) ze zip(self.layers, gradients) "
            "zavolejte layer.linear.update(dW, db, learning_rate)."
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
