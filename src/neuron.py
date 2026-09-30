"""Neuron jako slozeni nezavisle afinni vrstvy a injektovane aktivace.

BRANA (continuity gate): Zkopirujte sem sve reseni `src/neuron.py` z Cviceni 09
(vcetne `save`/`load`). Soubor je s Cvicenim 09 totozny, staci doplnit tela
metod. `save`/`load` v Cviceni 10 NENI novy ukol - zustava zde jako spojovaci
nit mezi cvicenimi.

`Neuron` sam nepocita vazeny soucet - to dela `Linear` (modul `src.linear`) -
ani zadnou aktivaci sam nezna: obe casti dostava hotove zvenku a jen je
sesklada (`forward = activation.forward(linear.forward(X))`), stejne jako se
v PyTorch sklada `nn.Linear` s `nn.ReLU()`.

Jeden `Neuron` predstavuje celou **vrstvu**: jeho `Linear` nese matici vah
`(n_priznaku, n_jednotek)` a vektor biasu `(n_jednotek,)`. Vrstvy se retezi
tridou `Sequential` (modul `src.network`). V Cviceni 10 se `Neuron` nemeni:
`Sequential` pri uceni pristupuje primo k jeho slozkam `linear` a
`activation` (potrebuje predaktivacni hodnotu `z`, kterou `Neuron.forward`
nevraci), vahy upravuje `Linear.update`.
"""

from __future__ import annotations

import numpy as np

from src.activations import Activation, Sigmoid, make_activation
from src.linear import Linear


class Neuron:
    """Slozeni `Linear` (afinni transformace) a injektovane `Activation`.

    `Linear` a `Activation` jsou nezavisle, samostatne testovatelne vrstvy;
    `Neuron` je jen tenky kontejner, ktery je za sebou zavola (kompozice,
    ne dedicnost). Aktivace je injektovana zvenku (dependency injection,
    kap. 9 konfiguratoru - paty vyskyt vzoru po `Distance`/`Initializer`/
    `Validator`/`Kernel`) - `Neuron` si zadnou konkretni `Activation` sam
    nevytvari, jen ji ulozi (uvnitr `Linear` totez plati pro `weights`/`bias`).

    Vlastnosti `weights`/`bias` jsou pouhe prostupne (passthrough) zkratky
    na `self.linear.weights`/`self.linear.bias`, aby kod, ktery uz na
    `neuron.weights`/`neuron.bias` spoleha (napr. `DecisionBoundaryPlotter`),
    fungoval beze zmeny.
    """

    def __init__(
        self,
        linear: Linear,
        activation: Activation,
        ) -> None:
        """Ulozi slozenou afinni vrstvu a injektovanou aktivaci.

        Parametry
        ---------
        linear : Linear
            Instance `Linear` (viz `src.linear`) - nese `weights`/`bias`,
            injektovana zvenku.
        activation : Activation
            Instance aktivacni funkce (viz `src.activations`), injektovana
            zvenku - `Neuron` ji sam nevytvari.
        """
        self.linear = linear
        self.activation = activation

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Provede dopredny pruchod: afinni transformace, pak aktivace.

        Definice
        --------
        z = self.linear.forward(x)
        vystup = self.activation.forward(z)

        `Neuron` sam nepocita ani vazeny soucet, ani aktivaci - obe deleguje
        na sve dve nezavisle slozky.

        Parametry
        ---------
        x : np.ndarray
            Matice vstupnich vzorku tvaru `(n_vzorku, n_priznaku)`.

        Navratova hodnota
        -----------------
        np.ndarray
            Vysledek `self.activation.forward` aplikovany na vystup
            `self.linear.forward(x)`.
        """
        z = self.linear.forward(x)
        return self.activation.forward(z)

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

    @property
    def weights(self) -> np.ndarray:
        """Prostupna (passthrough) zkratka na `self.linear.weights`."""
        return self.linear.weights

    @property
    def bias(self) -> np.ndarray | float | None:
        """Prostupna (passthrough) zkratka na `self.linear.bias`."""
        return self.linear.bias

    def save(self, path: str) -> None:
        """Ulozi model (vahy, bias, jmeno aktivace) do .npz archivu.

        Co presne se uklada
        --------------------
        - `weights` - `self.linear.weights` (numpy pole),
        - `bias` - `self.linear.bias`, POKUD neni `None` (klic se do archivu
          vubec nezapise, pokud bias vypnuty - nepritomnost klice je signal
          `bias=None`, zadny umely sentinel jako NaN),
        - `activation_name` - `type(self.activation).__name__` (retezec,
          napr. `"Sigmoid"`), aby `load` vedel, kterou tridu rekonstruovat,
        - `temperature` - POUZE pokud je aktivace `Sigmoid` (jinak se
          neuklada, protoze ostatni aktivace parametr nemaji).

        Format .npz (ne JSON, ne pickle) je v souladu s konfiguratorem
        kap. 9b: stav modelu jsou numpy pole, `pickle` je cerna skrinka
        a bezpecnostni riziko.

        Parametry
        ---------
        path : str
            Cesta k vystupnimu .npz souboru.
        """
        # assert  Ověřte, že self.linear.weights je typu np.ndarray
        raise NotImplementedError(
            "Úkol: sestavte slovnik klicu {'weights':..., 'activation_name':...} "
            "(pripadne 'bias' pokud self.linear.bias is not None a 'temperature' "
            "pokud je self.activation instance Sigmoid) a ulozte ho pomoci "
            "np.savez(path, **slovnik)."
        )

    @classmethod
    def load(cls, path: str) -> Neuron:
        """Nacte model ulozeny metodou `save` a vrati novou instanci `Neuron`.

        Presna inverze k `save`: `np.load(path)` vrati archiv, `weights`
        se precte primo, `bias` je `None` pokud klic v archivu chybi, jinak
        `archiv["bias"]` (pro vrstvu pole tvaru `(n_jednotek,)`; prevod
        `float(...)` z Cviceni 08 funguje jen pro jediny neuron se skalarnim
        biasem); `activation_name` a (je-li pritomna)
        `temperature` se predaji tovarni funkci `make_activation`
        (viz `src.activations`), ktera vrati spravnou instanci `Activation`.
        Z ziskanych `weights`/`bias` se sestavi novy `Linear`.

        Parametry
        ---------
        path : str
            Cesta k .npz souboru vytvorenemu metodou `save`.

        Navratova hodnota
        -----------------
        Neuron
            Nova instance se stejnymi vahami, biasem a aktivaci jako pri
            ulozeni - `load(p).forward(X)` musi davat stejne vysledky jako
            puvodni neuron pred ulozenim.
        """
        # assert  Ověřte, že soubor path existuje (napr. os.path.exists)
        raise NotImplementedError(
            "Úkol: nactete archiv pres np.load(path), precte 'weights' a "
            "volitelne 'bias' (None pokud klic chybi), sestavte Linear(weights, bias), "
            "pomoci make_activation('activation_name'[, temperature=...]) sestavte "
            "aktivaci a vratte cls(linear, activation)."
        )
