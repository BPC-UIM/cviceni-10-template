"""Rodina aktivacnich funkci (vzor Strategy) pro neuron a vrstvy site.

BRANA+ (continuity gate s rozsirenim): Do metod `forward` a `output_range`
vlozte sve reseni `src/activations.py` z Cviceni 09 (signatury jsou totozne).
NEPREPISUJTE cely soubor najednou - v Cviceni 10 pribyla v kazde tride nova
metoda `derivative`, ktera je novym ukolem tohoto cviceni.

Trida `Activation` definuje spolecne rozhrani (`forward`, `derivative`,
`output_range`); konkretni potomci (`Step`, `Sigmoid`, `Tanh`, `ReLU`,
`ReLU6`) sdileji jen toto rozhrani, ne kod. Instance `Activation` se
injektuje do `Neuron` (dependency injection, kap. 9 konfiguratoru).

NOVE v Cviceni 10: `derivative(x)` vraci derivaci aktivace `f'(x)` prvek po
prvku. Argumentem je **predaktivacni** hodnota `x` (tj. `z = x @ W + b`
dane vrstvy), nikoli vystup aktivace - presne tu si `Sequential.forward`
uklada do `z_` a `Sequential.backward` ji sem pri zpetnem pruchodu predava.

Modul navic definuje tovarni funkci `make_activation` (Strategy + Factory,
kap. 8c konfiguratoru): z ulozeneho jmena aktivace (retezec) vytvori
odpovidajici instanci. Pouziva ji `Neuron.load` pri rekonstrukci ulozeneho
modelu.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Activation(ABC):
    """Abstraktni zaklad rodiny aktivacnich funkci (plocha ABC hierarchie).

    Jde o vzor Strategy (konfigurator kap. 8a): jednotlive aktivace jsou
    navzajem nezavisle a sdileji pouze spolecne rozhrani `forward`,
    `derivative` a `output_range`, ne implementaci. Diky tomu lze aktivaci,
    kterou pouziva `Neuron`, zamenit bez zasahu do teto tridy - staci
    injektovat jinou instanci potomka (dependency injection, kap. 9
    konfiguratoru).
    """

    @property
    @abstractmethod
    def output_range(self) -> tuple[float, float]:
        """Vrati rozsah hodnot, ktere muze `forward` vratit.

        Navratova hodnota
        -----------------
        tuple[float, float]
            Dvojice `(dolni_mez, horni_mez)` moznych vystupnich hodnot.
        """

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Provede dopredny pruchod aktivacni funkci.

        Parametry
        ---------
        x : np.ndarray
            Vstupni pole (typicky vysledek vazeneho souctu z `Linear.forward`).

        Navratova hodnota
        -----------------
        np.ndarray
            Pole stejneho tvaru jako `x` po aplikaci aktivacni funkce.
        """

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Vrati derivaci aktivacni funkce `f'(x)` prvek po prvku (NOVE v Cviceni 10).

        Derivace se pocita v bode `x`, ktery je **predaktivacni** hodnotou
        (vystupem `Linear.forward`, v `Sequential` ulozenym v `z_`). Pri
        zpetnem pruchodu se ji nasobi (prvek po prvku) gradient chyby
        prichazejici do vrstvy - retezove pravidlo `dL/dz = dL/da * f'(z)`.

        Parametry
        ---------
        x : np.ndarray
            Predaktivacni hodnoty libovolneho tvaru.

        Navratova hodnota
        -----------------
        np.ndarray
            Pole stejneho tvaru jako `x` s hodnotami `f'(x)`.
        """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Zkratka pro `forward` - umoznuje volat instanci jako funkci.

        Parametry
        ---------
        x : np.ndarray
            Vstupni pole predane primo do `forward`.

        Navratova hodnota
        -----------------
        np.ndarray
            Vysledek `self.forward(x)`.
        """
        return self.forward(x)


class Step(Activation):
    """Tvrda skokova aktivace (Heaviside step).

    Vystupem je ostre binarni rozhodnuti bez pravdepodobnostni interpretace:
    1, pokud je vstup nezaporny, jinak 0. Pro uceni gradientnim sestupem je
    nepouzitelna - jeji derivace je vsude (krome bodu 0) nulova, takze
    gradient chyby jejim prostrednictvim neprochazi.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Vypocte skokovou aktivaci.

        Definice
        --------
        f(x) = 1, pokud x >= 0
        f(x) = 0, pokud x < 0

        Parametry
        ---------
        x : np.ndarray
            Vstupni pole.

        Navratova hodnota
        -----------------
        np.ndarray
            Pole stejneho tvaru jako `x` s hodnotami z mnoziny {0.0, 1.0}.
        """
        # assert  Ověřte, že x je typu np.ndarray
        # assert  Ověřte, že vystup obsahuje jen hodnoty 0.0 a 1.0
        raise NotImplementedError(
            "Úkol: (brana z Cviceni 09) implementujte f(x) = 1 pro x >= 0, "
            "jinak 0, vektorizovane pomoci np.where."
        )

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Vrati derivaci skokove aktivace (NOVE v Cviceni 10).

        Definice
        --------
        f'(x) = 0   pro x != 0
        V bode x = 0 derivace neexistuje (skok); konvencne vracime take 0.

        Nulova derivace znamena, ze pres `Step` zadny gradient neprojde -
        proto se site se skokovou aktivaci gradientnim sestupem ucit nedaji
        (duvod, proc v Cviceni 10 pouzivame `Sigmoid`).

        Parametry
        ---------
        x : np.ndarray
            Predaktivacni hodnoty.

        Navratova hodnota
        -----------------
        np.ndarray
            Nulove pole stejneho tvaru jako `x` (typ float).
        """
        # assert  Ověřte, že x je typu np.ndarray
        raise NotImplementedError(
            "Úkol: vratte nulove pole stejneho tvaru jako x (np.zeros_like "
            "s dtype float) - derivace skoku je vsude nulova, v bode 0 "
            "konvencne take 0."
        )

    @property
    def output_range(self) -> tuple[float, float]:
        """Vrati rozsah hodnot skokove aktivace.

        Navratova hodnota
        -----------------
        tuple[float, float]
            Konstantni dvojice `(0.0, 1.0)`.
        """
        # assert  Ověřte, že vracite tuple dvou float hodnot, ne list ani int
        raise NotImplementedError(
            "Úkol: (brana z Cviceni 09) vratte konstantni tuple (0.0, 1.0) "
            "jako obor hodnot Step aktivace."
        )


class Sigmoid(Activation):
    """Hladka (meka) aktivace s nastavitelnou teplotou.

    Logisticka funkce parametrizovana `temperature`, ktera rika, jak ostry
    je prechod mezi 0 a 1. Cim mensi `temperature`, tim strmejsi je prechod
    (a tim vetsi je derivace v okoli nuly, ale tim rychleji sigmoida
    saturuje); cim vetsi `temperature`, tim je prechod plossi.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        """Ulozi teplotu sigmoidy.

        Parametry
        ---------
        temperature : float
            Kladne cislo ridici strmost prechodu; vychozi hodnota 1.0.
        """
        self.temperature = temperature

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Vypocte sigmoidu s teplotou.

        Definice
        --------
        f(x) = 1 / (1 + exp(-x / temperature))

        Parametry
        ---------
        x : np.ndarray
            Vstupni pole.

        Navratova hodnota
        -----------------
        np.ndarray
            Pole stejneho tvaru jako `x` s hodnotami v otevrenem intervalu (0.0, 1.0).
        """
        # assert  Ověřte, že x je typu np.ndarray
        # assert  Ověřte, že self.temperature je kladne cislo
        raise NotImplementedError(
            "Úkol: (brana z Cviceni 09) implementujte f(x) = 1 / (1 + exp(-x / "
            "self.temperature)) pomoci np.exp, vektorizovane pres cele pole x."
        )

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Vrati derivaci sigmoidy s teplotou (NOVE v Cviceni 10).

        Definice
        --------
        s     = f(x) = 1 / (1 + exp(-x / T))
        f'(x) = (1 / T) * s * (1 - s)

        Derivaci neni treba odvozovat znovu: staci spocitat hodnotu sigmoidy
        `s = self.forward(x)` a vyuzit identitu `sigma' = sigma * (1 - sigma)`.
        Faktor `1 / T` pochazi z vnitrni funkce `x / T` (retezove pravidlo).
        Maximum `1 / (4T)` nastava v `x = 0`; pro `|x| >> T` je `s` blizko
        0 nebo 1 a derivace je temer nulova - sigmoida **saturuje**.

        Parametry
        ---------
        x : np.ndarray
            Predaktivacni hodnoty (NE vystup sigmoidy).

        Navratova hodnota
        -----------------
        np.ndarray
            Pole stejneho tvaru jako `x` s hodnotami v intervalu (0, 1/(4T)].
        """
        # assert  Ověřte, že x je typu np.ndarray
        # assert  Ověřte, že self.temperature je kladne cislo
        raise NotImplementedError(
            "Úkol: spocitejte s = self.forward(x) a vratte "
            "(1.0 / self.temperature) * s * (1.0 - s)."
        )

    @property
    def output_range(self) -> tuple[float, float]:
        """Vrati rozsah hodnot sigmoidy.

        Navratova hodnota
        -----------------
        tuple[float, float]
            Konstantni dvojice `(0.0, 1.0)`.
        """
        # assert  Ověřte, že vracite tuple dvou float hodnot, ne list ani int
        raise NotImplementedError(
            "Úkol: (brana z Cviceni 09) vratte konstantni tuple (0.0, 1.0) "
            "jako obor hodnot Sigmoid aktivace."
        )


class Tanh(Activation):
    """Hyperbolicky tangens jako symetricka meka aktivace.

    Na rozdil od `Sigmoid` je vystup symetricky kolem nuly a nabyva zapornych
    i kladnych hodnot. Stejne jako sigmoida pro velka `|x|` saturuje.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Vypocte hyperbolicky tangens.

        Definice
        --------
        f(x) = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

        Parametry
        ---------
        x : np.ndarray
            Vstupni pole.

        Navratova hodnota
        -----------------
        np.ndarray
            Pole stejneho tvaru jako `x` s hodnotami v otevrenem intervalu (-1.0, 1.0).
        """
        # assert  Ověřte, že x je typu np.ndarray
        raise NotImplementedError(
            "Úkol: (brana z Cviceni 09) implementujte f(x) = tanh(x) pomoci "
            "np.tanh, vektorizovane pres cele pole x."
        )

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Vrati derivaci hyperbolickeho tangens (NOVE v Cviceni 10).

        Definice
        --------
        f'(x) = 1 - tanh(x)^2

        Obdoba identity u sigmoidy: derivace se vyjadri pomoci hodnoty
        funkce samotne. Maximum 1 v `x = 0`, pro velka `|x|` jde k nule.

        Parametry
        ---------
        x : np.ndarray
            Predaktivacni hodnoty.

        Navratova hodnota
        -----------------
        np.ndarray
            Pole stejneho tvaru jako `x` s hodnotami v intervalu (0.0, 1.0].
        """
        # assert  Ověřte, že x je typu np.ndarray
        raise NotImplementedError(
            "Úkol: spocitejte t = self.forward(x) a vratte 1.0 - t**2."
        )

    @property
    def output_range(self) -> tuple[float, float]:
        """Vrati rozsah hodnot tanh aktivace.

        Navratova hodnota
        -----------------
        tuple[float, float]
            Konstantni dvojice `(-1.0, 1.0)`.
        """
        # assert  Ověřte, že vracite tuple dvou float hodnot, ne list ani int
        raise NotImplementedError(
            "Úkol: (brana z Cviceni 09) vratte konstantni tuple (-1.0, 1.0) "
            "jako obor hodnot Tanh aktivace."
        )


class ReLU(Activation):
    """Rectified Linear Unit - useknuti zapornych hodnot na nule.

    Pro zaporny vstup vraci 0, pro nezaporny vstup vraci vstup samotny.
    Pro kladne vstupy nesaturuje (derivace je 1), proto se v hlubokych
    sitich uci rychleji nez sigmoida.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Vypocte ReLU.

        Definice
        --------
        f(x) = max(0, x)

        Parametry
        ---------
        x : np.ndarray
            Vstupni pole.

        Navratova hodnota
        -----------------
        np.ndarray
            Pole stejneho tvaru jako `x` s hodnotami v intervalu [0.0, +inf).
        """
        # assert  Ověřte, že x je typu np.ndarray
        raise NotImplementedError(
            "Úkol: (brana z Cviceni 09) implementujte f(x) = max(0, x) pomoci "
            "np.maximum, vektorizovane pres cele pole x."
        )

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Vrati derivaci ReLU (NOVE v Cviceni 10).

        Definice
        --------
        f'(x) = 1   pro x > 0
        f'(x) = 0   pro x < 0
        V bode x = 0 derivace neexistuje (zlom); konvencne vracime 0
        (shodne s PyTorch).

        Parametry
        ---------
        x : np.ndarray
            Predaktivacni hodnoty.

        Navratova hodnota
        -----------------
        np.ndarray
            Pole stejneho tvaru jako `x` s hodnotami z mnoziny {0.0, 1.0}.
        """
        # assert  Ověřte, že x je typu np.ndarray
        raise NotImplementedError(
            "Úkol: vratte 1.0 tam, kde x > 0, jinak 0.0 (napr. "
            "(x > 0).astype(float)); v bode x = 0 konvencne 0."
        )

    @property
    def output_range(self) -> tuple[float, float]:
        """Vrati rozsah hodnot ReLU aktivace.

        Navratova hodnota
        -----------------
        tuple[float, float]
            Konstantni dvojice `(0.0, np.inf)`.
        """
        # assert  Ověřte, že horni mez je +nekonecno (np.inf), ne konecne cislo
        raise NotImplementedError(
            "Úkol: (brana z Cviceni 09) vratte konstantni tuple (0.0, np.inf) "
            "jako obor hodnot ReLU aktivace."
        )


class ReLU6(Activation):
    """ReLU oriznuta shora na hodnotu 6.

    Stejne jako `ReLU` useka zaporne hodnoty na nule, navic ale shora orizne
    vystup na 6, cimz ziska (na rozdil od `ReLU`) konecny obor hodnot.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Vypocte ReLU6.

        Definice
        --------
        f(x) = min(max(0, x), 6)

        Parametry
        ---------
        x : np.ndarray
            Vstupni pole.

        Navratova hodnota
        -----------------
        np.ndarray
            Pole stejneho tvaru jako `x` s hodnotami v intervalu [0.0, 6.0].
        """
        # assert  Ověřte, že x je typu np.ndarray
        raise NotImplementedError(
            "Úkol: (brana z Cviceni 09) implementujte f(x) = min(max(0, x), 6) "
            "pomoci np.clip, vektorizovane pres cele pole x."
        )

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Vrati derivaci ReLU6 (NOVE v Cviceni 10).

        Definice
        --------
        f'(x) = 1   pro 0 < x < 6
        f'(x) = 0   pro x < 0 nebo x > 6
        V bodech zlomu x = 0 a x = 6 konvencne vracime 0.

        Parametry
        ---------
        x : np.ndarray
            Predaktivacni hodnoty.

        Navratova hodnota
        -----------------
        np.ndarray
            Pole stejneho tvaru jako `x` s hodnotami z mnoziny {0.0, 1.0}.
        """
        # assert  Ověřte, že x je typu np.ndarray
        raise NotImplementedError(
            "Úkol: vratte 1.0 tam, kde 0 < x < 6, jinak 0.0 (napr. "
            "((x > 0) & (x < 6)).astype(float))."
        )

    @property
    def output_range(self) -> tuple[float, float]:
        """Vrati rozsah hodnot ReLU6 aktivace.

        Navratova hodnota
        -----------------
        tuple[float, float]
            Konstantni dvojice `(0.0, 6.0)`.
        """
        # assert  Ověřte, že vracite tuple dvou float hodnot, ne list ani int
        raise NotImplementedError(
            "Úkol: (brana z Cviceni 09) vratte konstantni tuple (0.0, 6.0) "
            "jako obor hodnot ReLU6 aktivace."
        )


# Registr jmen aktivaci na tridy - podklad pro `make_activation` nize.
_ACTIVATION_REGISTRY: dict[str, type[Activation]] = {
    "Step": Step,
    "Sigmoid": Sigmoid,
    "Tanh": Tanh,
    "ReLU": ReLU,
    "ReLU6": ReLU6,
}


def make_activation(name: str, *, temperature: float = 1.0) -> Activation:
    """Tovarni funkce: vytvori instanci `Activation` podle jmena (Strategy + Factory).

    Predvyplnena infrastruktura pro `Neuron.load` (modul `src.neuron`) - ulozeny
    .npz archiv nese jen jmeno aktivace jako textova metadata (`type(x).__name__`),
    ne cely objekt, takze pri nacitani je potreba jmeno zpetne prevest na instanci.

    Parametry
    ---------
    name : str
        Jmeno tridy aktivace, jedno z `"Step"`, `"Sigmoid"`, `"Tanh"`, `"ReLU"`,
        `"ReLU6"` (presna shoda velikosti pismen, odpovida `type(x).__name__`).
    temperature : float, optional
        Teplota pro `Sigmoid` (viz `Sigmoid.__init__`); pro ostatni aktivace
        se ignoruje. Vychozi `1.0`.

    Navratova hodnota
    -----------------
    Activation
        Nova instance prislusne tridy (`Sigmoid` dostane `temperature`,
        ostatni aktivace nemaji zadne parametry konstruktoru).

    Vyjimky
    -------
    ``ValueError``:
        Pokud `name` neni v `_ACTIVATION_REGISTRY`.
    """
    if name not in _ACTIVATION_REGISTRY:
        povolene = ", ".join(sorted(_ACTIVATION_REGISTRY))
        raise ValueError(f"Neznama aktivace: {name!r}. Povolene hodnoty jsou: {povolene}.")

    cls = _ACTIVATION_REGISTRY[name]
    if cls is Sigmoid:
        return Sigmoid(temperature=temperature)
    return cls()
