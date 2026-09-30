"""Inicializace vah vrstev (vzor Strategy + Factory).

Pred ucenim je treba vahy nejak zvolit. Nulove vahy nefunguji (vsechny
neurony vrstvy by pocitaly totez a dostavaly stejny gradient - symetrie by se
nikdy nerozbila), proto se voli nahodne. **Meritko** nahodnych vah ale
rozhoduje o ucitelnosti: prilis velke vahy posunou `z` daleko od nuly,
sigmoida saturuje a gradient mizi; prilis male vahy zase utlumi signal.

Inicializator vraci matici vah tvaru `(n_vstupu, n_jednotek)`, kterou
primo prijima `Linear` - trida vrstvy se kvuli inicializaci nemeni (vahy
jsou do ni stale vkladany zvenku, jako v Cviceni 09 rucne navrzene).

Primy sourozenec tridy `Initializer` z Cviceni 03 (inicializace centroidu
k-means): stejna sprava generatoru nahodnych cisel, stejny vzor Strategy
+ Factory (vyber retezcem z `config.yaml`, klic `network.initializer`).

Obsah modulu:
    WeightsInitializer   - abstraktni zaklad (PREDVYPLNENO)
    RandomUniformInit    - rovnomerne rozdeleni v pevnem intervalu (PREDVYPLNENO, vzor)
    XavierInit           - Glorot/Xavier, meritko podle n_vstupu + n_jednotek (UKOL)
    HeInit               - He/Kaiming, meritko podle n_vstupu (UKOL, BONUS)
    PyTorchDefaultInit   - vychozi inicializace vah `nn.Linear` v PyTorch (PREDVYPLNENO, pro srovnani)
    make_weights_initializer - tovarni funkce (PREDVYPLNENO)

Vsechny inicializatory zde pouzivaji **rovnomerne (uniformni) rozdeleni**
`U(-a, a)`: kazda hodnota z intervalu `[-a, a]` je stejne pravdepodobna,
rozptyl je `a^2 / 3`. Neni to normalni rozdeleni - to maji varianty
`xavier_normal_` a `kaiming_normal_` v PyTorch se stejnym rozptylem.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class WeightsInitializer(ABC):
    """Abstraktni zaklad inicializatoru vah (PREDVYPLNENO).

    Spravuje generator nahodnych cisel: kazda instance ma vlastni
    `np.random.Generator`, takze se stejnym `random_state` dava
    reprodukovatelne vahy a neovlivnuje globalni stav `np.random`.
    """

    def __init__(self, random_state: int | None = None) -> None:
        """Vytvori vlastni generator nahodnych cisel.

        Parametry
        ---------
        random_state : int | None, optional
            Seed generatoru; `None` znamena nereprodukovatelne vahy.
        """
        self.random_state = random_state
        self.rng: np.random.Generator = np.random.default_rng(random_state)

    @abstractmethod
    def initialize(self, n_inputs: int, n_units: int) -> np.ndarray:
        """Vrati nahodnou matici vah jedne vrstvy.

        Parametry
        ---------
        n_inputs : int
            Pocet vstupu vrstvy (pocet radku matice vah, "fan-in").
        n_units : int
            Pocet jednotek (neuronu) vrstvy (pocet sloupcu, "fan-out").

        Navratova hodnota
        -----------------
        np.ndarray
            Matice tvaru `(n_inputs, n_units)` typu float64.
        """


class RandomUniformInit(WeightsInitializer):
    """Rovnomerne rozdeleni v pevnem intervalu `[low, high]` (PREDVYPLNENO, vzor).

    Nejjednodussi volba, obdoba `np.random.uniform(low, high, size)`.
    Interval nezavisi na velikosti vrstvy - u vrstvy s mnoha vstupy proto
    vznikne velky soucet `z` a sigmoida snadno saturuje. Prave to resi
    `XavierInit`.
    """

    def __init__(self, low: float = -1.0, high: float = 1.0,
                 random_state: int | None = None) -> None:
        """Ulozi meze intervalu a vytvori generator.

        Parametry
        ---------
        low : float, optional
            Dolni mez intervalu, vychozi -1.0.
        high : float, optional
            Horni mez intervalu, vychozi 1.0.
        random_state : int | None, optional
            Seed generatoru.
        """
        super().__init__(random_state=random_state)
        self.low = low
        self.high = high

    def initialize(self, n_inputs: int, n_units: int) -> np.ndarray:
        """Vrati matici vah z rovnomerneho rozdeleni `U(low, high)`.

        Parametry
        ---------
        n_inputs : int
            Pocet vstupu vrstvy.
        n_units : int
            Pocet jednotek vrstvy.

        Navratova hodnota
        -----------------
        np.ndarray
            Matice tvaru `(n_inputs, n_units)`.
        """
        assert n_inputs >= 1 and n_units >= 1, "n_inputs i n_units musi byt >= 1"
        return self.rng.uniform(self.low, self.high, size=(n_inputs, n_units))


class XavierInit(WeightsInitializer):
    """Inicializace Xavier/Glorot (Glorot a Bengio, 2010) - vhodna pro sigmoidu a tanh.

    Voli meritko tak, aby rozptyl signalu zustaval pri pruchodu vrstvami
    (dopredu i nazpet) priblizne stejny. Pouziva rovnomerne rozdeleni
    `U(-limit, limit)` s

        limit = sqrt(6 / (n_inputs + n_units)),

    cemuz odpovida rozptyl vah `Var(W) = limit^2 / 3 = 2 / (n_inputs + n_units)`.
    """

    def initialize(self, n_inputs: int, n_units: int) -> np.ndarray:
        """Vrati matici vah z rozdeleni `U(-limit, limit)`, `limit = sqrt(6 / (n_in + n_out))`.

        Parametry
        ---------
        n_inputs : int
            Pocet vstupu vrstvy (fan-in).
        n_units : int
            Pocet jednotek vrstvy (fan-out).

        Navratova hodnota
        -----------------
        np.ndarray
            Matice tvaru `(n_inputs, n_units)`.
        """
        # assert  Ověřte, že n_inputs i n_units jsou alespon 1
        raise NotImplementedError(
            "Úkol: spocitejte limit = np.sqrt(6.0 / (n_inputs + n_units)) a vratte "
            "self.rng.uniform(-limit, limit, size=(n_inputs, n_units))."
        )


class HeInit(WeightsInitializer):
    """Inicializace He/Kaiming (He a kol., 2015) - vhodna pro ReLU (BONUS).

    ReLU propousti jen kladnou polovinu signalu, proto potrebuje priblizne
    dvojnasobny rozptyl nez Xavier a meritko zavisi jen na poctu vstupu:

        limit = sqrt(6 / n_inputs),   Var(W) = 2 / n_inputs.
    """

    def initialize(self, n_inputs: int, n_units: int) -> np.ndarray:
        """Vrati matici vah z rozdeleni `U(-limit, limit)`, `limit = sqrt(6 / n_in)`.

        Parametry
        ---------
        n_inputs : int
            Pocet vstupu vrstvy (fan-in).
        n_units : int
            Pocet jednotek vrstvy.

        Navratova hodnota
        -----------------
        np.ndarray
            Matice tvaru `(n_inputs, n_units)`.
        """
        # assert  Ověřte, že n_inputs i n_units jsou alespon 1
        raise NotImplementedError(
            "Úkol: (BONUS) spocitejte limit = np.sqrt(6.0 / n_inputs) a vratte "
            "self.rng.uniform(-limit, limit, size=(n_inputs, n_units))."
        )


class PyTorchDefaultInit(WeightsInitializer):
    """Vychozi inicializace vah vrstvy `torch.nn.Linear` (PREDVYPLNENO, pro srovnani).

    PyTorch v `nn.Linear.reset_parameters` vola
    `init.kaiming_uniform_(weight, a=sqrt(5))`, coz po dosazeni vede na

        limit = 1 / sqrt(n_inputs),   Var(W) = 1 / (3 * n_inputs).

    Nejde tedy o Xavier (ten zavisi i na `n_units`) ani o He (`Var = 2 / n_in`,
    tedy 6x vetsi rozptyl): meritko zavisi jen na poctu vstupu, jako u He, ale
    je zamerne male. PyTorch stejnym rozdelenim `U(-limit, limit)` inicializuje
    i bias; v tomto cviceni zacinaji biasy vsech inicializatoru na nule
    (sestavuje je pipeline), inicializator vraci jen matici vah.
    """

    def initialize(self, n_inputs: int, n_units: int) -> np.ndarray:
        """Vrati matici vah z rozdeleni `U(-limit, limit)`, `limit = 1 / sqrt(n_in)`.

        Parametry
        ---------
        n_inputs : int
            Pocet vstupu vrstvy (fan-in).
        n_units : int
            Pocet jednotek vrstvy.

        Navratova hodnota
        -----------------
        np.ndarray
            Matice tvaru `(n_inputs, n_units)`.
        """
        assert n_inputs >= 1 and n_units >= 1, "n_inputs i n_units musi byt >= 1"
        limit = 1.0 / np.sqrt(n_inputs)
        return self.rng.uniform(-limit, limit, size=(n_inputs, n_units))


# Registr jmen inicializatoru (klic `network.initializer` v config.yaml) na tridy.
_INITIALIZER_REGISTRY: dict[str, type[WeightsInitializer]] = {
    "random_uniform": RandomUniformInit,
    "xavier": XavierInit,
    "he": HeInit,
    "pytorch_default": PyTorchDefaultInit,
}


def make_weights_initializer(name: str, random_state: int | None = None) -> WeightsInitializer:
    """Tovarni funkce: vytvori inicializator podle jmena z `config.yaml` (PREDVYPLNENO).

    Parametry
    ---------
    name : str
        Jmeno inicializatoru, jedno z `"random_uniform"`, `"xavier"`, `"he"`,
        `"pytorch_default"`.
    random_state : int | None, optional
        Seed generatoru predany konstruktoru.

    Navratova hodnota
    -----------------
    WeightsInitializer
        Nova instance prislusne tridy.

    Vyjimky
    -------
    ``ValueError``:
        Pokud `name` neni v `_INITIALIZER_REGISTRY`.
    """
    if name not in _INITIALIZER_REGISTRY:
        povolene = ", ".join(sorted(_INITIALIZER_REGISTRY))
        raise ValueError(f"Neznamy inicializator: {name!r}. Povolene hodnoty jsou: {povolene}.")
    return _INITIALIZER_REGISTRY[name](random_state=random_state)
