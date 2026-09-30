# -*- coding: utf-8 -*-

"""
Created on 29. 09. 2026 at 10:20:00

Author: Richard Redina
Email: 195715@vut.cz
Affiliation:
         International Clinical Research Center, Brno
         Brno University of Technology, Brno
GitHub: RicRedi

(._.)
 <|>
_/|_

Description:
    Testy pro cviceni 10 -- uceni site gradientnim sestupem (backpropagation).

    Spousteni:  pytest -v

    Ve stavu stubu se sada NACTE a testy, ktere volaji nedokoncene ukoly,
    se oznaci jako xfail (ocekavane selhani s NotImplementedError) -- sada
    nikdy neskonci holym tracebackem. Po dokonceni ukolu z nich budou
    prochazejici testy.

    Pateri sady jsou kontroly gradientu: analyticky gradient (derivace
    aktivaci, gradient ztrat, Sequential.backward) se porovnava s numerickou
    derivaci (centralni diference). Chybne znamenko, chybejici transpozice
    nebo spatna derivace se tak projevi okamzite.

    Testy Sequential pouzivaji DummyLayer -- minimalni vrstvu (Linear +
    sigmoida s derivaci) v cistem numpy, NEZAVISLOU na brane src/ z Cviceni
    09. Spravnost zpetneho pruchodu se tak overi i tehdy, kdyz brana jeste
    neni vlozena (kap. 10/15 konfiguratoru).

    Trinact trid testu:
      TestDerivaceAktivaci   -- derivative() vs. numericka derivace, Step, tvar
      TestLinearUpdate       -- krok gradientniho sestupu na vahach a biasu
      TestSequentialForward  -- zaznam io_ a z_ (DummyLayer)
      TestSequentialBackward -- gradienty vs. numericka derivace (DummyLayer)
      TestSequentialUpdate   -- delegovani na linear.update, pokles chyby (DummyLayer)
      TestSitZNeuronu        -- kontrola gradientu site ze skutecnych Neuronu
      TestMSE / TestBCE      -- hodnota ztraty a gradient vs. numericka derivace
      TestInicializatory     -- tvar, meze a rozptyl vah (Xavier, He, uniform, PyTorch)
      TestTrainer            -- mechanika trenovaciho wrapperu (DummyModel)
      TestUceniEndToEnd      -- kratke uceni na separovatelnych datech snizi chybu
      TestMetriky            -- matice zamen a metriky (predvyplnene dataio)
      TestDataAKonfigurace   -- loader a load_config / validate_config
================================================================================
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest

from dataio import (
    TrainingConfig,
    classification_metrics,
    confusion_matrix_2x2,
    load_breast_cancer_data,
    load_config,
    validate_config,
)
from src import (
    BCE,
    MSE,
    HeInit,
    Linear,
    Neuron,
    PyTorchDefaultInit,
    RandomUniformInit,
    ReLU,
    ReLU6,
    Sequential,
    Sigmoid,
    Step,
    Tanh,
    Trainer,
    XavierInit,
    make_loss,
    make_weights_initializer,
)

STUB = pytest.mark.xfail(raises=NotImplementedError, strict=False,
                         reason="studentsky ukol jeste neni dokoncen")

EPS = 1e-6          # krok centralni diference
TOL = 1e-6          # tolerance shody analytickeho a numerickeho gradientu


# --------------------------------------------------------------------------- #
#  Pomocne funkce a Dummy tridy (nezavisle na src/)                          #
# --------------------------------------------------------------------------- #
def _numericka_derivace_prvkove(f: Callable[[np.ndarray], np.ndarray], x: np.ndarray) -> np.ndarray:
    """Centralni diference prvkove funkce f v bodech x."""
    return (f(x + EPS) - f(x - EPS)) / (2.0 * EPS)


def _numericky_gradient(skalarni_funkce: Callable[[], float], parametr: np.ndarray) -> np.ndarray:
    """Centralni diference skalarni funkce podle kazdeho prvku parametru (meni ho na miste)."""
    gradient = np.zeros_like(parametr, dtype=np.float64)
    for index in np.ndindex(parametr.shape):
        puvodni = parametr[index]
        parametr[index] = puvodni + EPS
        l_plus = skalarni_funkce()
        parametr[index] = puvodni - EPS
        l_minus = skalarni_funkce()
        parametr[index] = puvodni
        gradient[index] = (l_plus - l_minus) / (2.0 * EPS)
    return gradient


def _mse(p: np.ndarray, y: np.ndarray) -> float:
    """Referencni MSE (nezavisla na src/losses.py)."""
    return float(np.mean((p - y) ** 2))


def _mse_gradient(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Referencni gradient MSE podle predikce."""
    return 2.0 * (p - y) / p.size


class DummyLinear:
    """Minimalni ``Linear`` v cistem numpy (``x @ W + b`` a ``update``)."""

    def __init__(self, weights, bias=None) -> None:
        """Ulozi vahy a bias jako float pole; pripravi zaznam volani ``update``."""
        self.weights = np.asarray(weights, dtype=np.float64)
        self.bias = None if bias is None else np.asarray(bias, dtype=np.float64)
        self.update_calls: list[tuple] = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Afinni transformace ``x @ W + b`` (bez biasu, je-li ``None``)."""
        return x @ self.weights + (self.bias if self.bias is not None else 0.0)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Zkratka pro ``forward``."""
        return self.forward(x)

    def update(self, weight_gradient, bias_gradient, learning_rate) -> None:
        """Zaznamena volani a provede krok ``W <- W - lr * dW`` (a obdobne bias)."""
        self.update_calls.append((weight_gradient, bias_gradient, learning_rate))
        self.weights = self.weights - learning_rate * weight_gradient
        if self.bias is not None:
            self.bias = self.bias - learning_rate * bias_gradient


class DummySigmoid:
    """Minimalni sigmoida s derivaci podle predaktivacni hodnoty."""

    def forward(self, z: np.ndarray) -> np.ndarray:
        """Sigmoida ``1 / (1 + exp(-z))``."""
        return 1.0 / (1.0 + np.exp(-z))

    def __call__(self, z: np.ndarray) -> np.ndarray:
        """Zkratka pro ``forward``."""
        return self.forward(z)

    def derivative(self, z: np.ndarray) -> np.ndarray:
        """Derivace ``s * (1 - s)`` v predaktivacnim bodu ``z`` (ne ve vystupu)."""
        s = self.forward(z)
        return s * (1.0 - s)


class DummyLayer:
    """Vrstva se slozkami ``linear`` a ``activation`` (rozhrani ``Neuron``), nezavisla na src/."""

    def __init__(self, weights, bias=None) -> None:
        """Slozi ``DummyLinear`` s danymi vahami a ``DummySigmoid``."""
        self.linear = DummyLinear(weights, bias)
        self.activation = DummySigmoid()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Dopredny pruchod vrstvou: aktivace po afinni transformaci."""
        return self.activation(self.linear(x))


def _dummy_sit(bias: bool = True) -> tuple[Sequential, np.ndarray, np.ndarray]:
    """Sit 3-4-2-1 z DummyLayer s nahodnymi vahami, 5 vzorku a cile 0/1."""
    rng = np.random.default_rng(7)
    sizes = [3, 4, 2, 1]
    vrstvy = [DummyLayer(rng.normal(size=(a, b)), rng.normal(size=b) if bias else None)
              for a, b in zip(sizes[:-1], sizes[1:])]
    x = rng.normal(size=(5, 3))
    y = rng.integers(0, 2, size=(5, 1)).astype(np.float64)
    return Sequential(vrstvy), x, y


def _config(**zmeny) -> TrainingConfig:
    """TrainingConfig s rozumnymi vychozimi hodnotami pro testy."""
    hodnoty = dict(learning_rate=0.1, epochs=5, batch_size=1, max_loss=0.0,
                   lr_decay=1.0, lr_decay_every=1000)
    hodnoty.update(zmeny)
    return TrainingConfig(**hodnoty)


# --------------------------------------------------------------------------- #
#  Derivace aktivaci (activations.py)                                        #
# --------------------------------------------------------------------------- #
# Body mimo zlomy ReLU (0) a ReLU6 (0, 6), aby byla numericka derivace definovana.
_BODY = np.array([[-7.3, -2.0, -0.4], [0.3, 1.7, 5.2], [6.8, 9.1, -0.05]])


class TestDerivaceAktivaci:
    """derivative(x) bere predaktivacni x a shoduje se s numerickou derivaci forward."""

    @STUB
    @pytest.mark.parametrize("teplota", [0.5, 1.0, 2.0])
    def test_sigmoid_derivace_vs_numericka(self, teplota: float) -> None:
        """Derivace sigmoidy (vcetne faktoru 1/T) odpovida centralni diferenci."""
        aktivace = Sigmoid(temperature=teplota)
        numericka = _numericka_derivace_prvkove(aktivace.forward, _BODY)
        assert np.allclose(aktivace.derivative(_BODY), numericka, atol=TOL)

    @STUB
    @pytest.mark.parametrize("trida", [Tanh, ReLU, ReLU6])
    def test_derivace_vs_numericka(self, trida) -> None:
        """Derivace tanh, ReLU a ReLU6 odpovida centralni diferenci mimo zlomy."""
        aktivace = trida()
        numericka = _numericka_derivace_prvkove(aktivace.forward, _BODY)
        assert np.allclose(aktivace.derivative(_BODY), numericka, atol=TOL)

    @STUB
    def test_sigmoid_maximum_derivace_v_nule(self) -> None:
        """sigma'(0) = 1/(4T): derivace se pocita z predaktivacni hodnoty, ne z vystupu."""
        assert Sigmoid(temperature=0.5).derivative(np.array([[0.0]]))[0, 0] == pytest.approx(0.5)
        assert Sigmoid(temperature=1.0).derivative(np.array([[0.0]]))[0, 0] == pytest.approx(0.25)

    @STUB
    def test_sigmoid_saturace_derivace_mizi(self) -> None:
        """Pro |z| >> T je derivace sigmoidy prakticky nulova (saturace)."""
        derivace = Sigmoid(temperature=1.0).derivative(np.array([[-30.0, 30.0]]))
        assert np.all(derivace < 1e-10)

    @STUB
    def test_step_derivace_je_nulova(self) -> None:
        """Derivace Step je vsude nulova a zachovava tvar vstupu."""
        derivace = Step().derivative(_BODY)
        assert derivace.shape == _BODY.shape
        assert np.all(derivace == 0.0)

    @STUB
    def test_relu_derivace_v_nule_je_nula(self) -> None:
        """Konvence v bode zlomu: ReLU'(0) = 0 (shodne s PyTorch)."""
        assert ReLU().derivative(np.array([[0.0]]))[0, 0] == 0.0

    @STUB
    def test_relu6_derivace_v_bodech_zlomu_je_nula(self) -> None:
        """Konvence v bodech zlomu: ReLU6'(0) = ReLU6'(6) = 0."""
        assert np.array_equal(ReLU6().derivative(np.array([[0.0, 6.0]])), np.array([[0.0, 0.0]]))

    @STUB
    @pytest.mark.parametrize("aktivace", [Sigmoid(), Tanh(), ReLU(), ReLU6(), Step()],
                             ids=["Sigmoid", "Tanh", "ReLU", "ReLU6", "Step"])
    def test_derivace_zachova_tvar(self, aktivace) -> None:
        """Derivace vraci pole stejneho tvaru jako vstup (prvek po prvku)."""
        assert np.asarray(aktivace.derivative(np.zeros((4, 3)))).shape == (4, 3)


# --------------------------------------------------------------------------- #
#  Linear.update                                                             #
# --------------------------------------------------------------------------- #
class TestLinearUpdate:
    """update provede W <- W - lr * dW (a obdobne bias); gradient vrstva nepocita."""

    @STUB
    def test_vahy_se_posunou_proti_gradientu(self) -> None:
        """Vahy i bias se posunou o -lr * gradient (znamenkova konvence)."""
        linear = Linear(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([0.5, -0.5]))
        linear.update(np.array([[1.0, 0.0], [0.0, -2.0]]), np.array([1.0, 1.0]), 0.1)
        assert np.allclose(linear.weights, np.array([[0.9, 2.0], [3.0, 4.2]]))
        assert np.allclose(linear.bias, np.array([0.4, -0.6]))

    @STUB
    def test_bez_biasu_zustane_none(self) -> None:
        """Vrstva bez biasu dostane bias_gradient None a bias zustane None."""
        linear = Linear(np.ones((2, 1)), None)
        linear.update(np.ones((2, 1)), None, 0.5)
        assert linear.bias is None
        assert np.allclose(linear.weights, 0.5)

    @STUB
    def test_nulovy_gradient_nic_nemeni(self) -> None:
        """Nulovy gradient ponecha parametry beze zmeny i pri velkem kroku."""
        linear = Linear(np.array([[1.0], [-1.0]]), np.array([0.3]))
        linear.update(np.zeros((2, 1)), np.zeros(1), 10.0)
        assert np.allclose(linear.weights, np.array([[1.0], [-1.0]]))
        assert np.allclose(linear.bias, np.array([0.3]))

    @STUB
    def test_forward_po_update_pouziva_nove_vahy(self) -> None:
        """Vahy jsou menitelny stav: forward po update pocita s novymi vahami."""
        linear = Linear(np.array([[1.0], [1.0]]), np.array([0.0]))
        linear.update(np.array([[1.0], [1.0]]), np.array([0.0]), 1.0)
        assert np.allclose(linear(np.array([[2.0, 3.0]])), np.array([[0.0]]))


# --------------------------------------------------------------------------- #
#  Sequential.forward (rozsireni o z_)                                        #
# --------------------------------------------------------------------------- #
class TestSequentialForward:
    """forward uklada io_ = [x, a_1, ..., a_L] i z_ = [z_1, ..., z_L]."""

    def test_init_nastavi_mezistavy_na_none(self) -> None:
        """__init__ je PREDVYPLNENY, neni STUB: io_ i z_ jsou pred pruchodem None."""
        sit, _, _ = _dummy_sit()
        assert sit.io_ is None and sit.z_ is None

    @STUB
    def test_delky_io_a_z(self) -> None:
        """io_ ma delku 1 + L, z_ delku L."""
        sit, x, _ = _dummy_sit()
        sit.forward(x)
        assert len(sit.io_) == 1 + len(sit.layers)
        assert len(sit.z_) == len(sit.layers)

    @STUB
    def test_z_je_vystup_linear_a_io_je_aktivace(self) -> None:
        """z_[k] je vystup linear vrstvy k a io_[k+1] je jeho aktivace."""
        sit, x, _ = _dummy_sit()
        vystup = sit.forward(x)
        assert np.array_equal(sit.io_[0], x)
        for k, vrstva in enumerate(sit.layers):
            assert np.allclose(sit.z_[k], vrstva.linear(sit.io_[k]))
            assert np.allclose(sit.io_[k + 1], vrstva.activation(sit.z_[k]))
        assert np.allclose(vystup, sit.io_[-1])

    @STUB
    def test_vystup_odpovida_primemu_vypoctu(self) -> None:
        """Vystup site se shoduje s primym vypoctem vzorcu v numpy."""
        sit, x, _ = _dummy_sit()
        a = x
        for vrstva in sit.layers:
            a = 1.0 / (1.0 + np.exp(-(a @ vrstva.linear.weights + vrstva.linear.bias)))
        assert np.allclose(sit(x), a)

    @STUB
    def test_novy_pruchod_prepise_mezistavy(self) -> None:
        """Novy pruchod zaklada io_ i z_ znovu (neprodluzuje stare seznamy)."""
        sit, x, _ = _dummy_sit()
        sit(x)
        sit(x[:2])
        assert sit.io_[0].shape == (2, 3)
        assert len(sit.z_) == len(sit.layers)
        assert sit.z_[0].shape == (2, 4)


# --------------------------------------------------------------------------- #
#  Sequential.backward -- jadro cviceni                                       #
# --------------------------------------------------------------------------- #
class TestSequentialBackward:
    """backward vraci [(dW, db), ...] v doprednem poradi; shoda s numerickou derivaci."""

    @STUB
    def test_vraci_dvojici_pro_kazdou_vrstvu_ve_spravnem_tvaru(self) -> None:
        """Pro kazdou vrstvu dvojice (dW, db) ve tvaru jejich parametru."""
        sit, x, y = _dummy_sit()
        p = sit.forward(x)
        gradienty = sit.backward(_mse_gradient(p, y))
        assert len(gradienty) == len(sit.layers)
        for vrstva, (d_w, d_b) in zip(sit.layers, gradienty):
            assert np.shape(d_w) == vrstva.linear.weights.shape
            assert np.shape(d_b) == vrstva.linear.bias.shape

    @STUB
    def test_jednovrstva_sit_analyticky(self) -> None:
        """Jedna vrstva: dW = x.T @ (g * f'(z)), db = soucet pres vzorky."""
        vrstva = DummyLayer(np.array([[0.5], [-1.0]]), np.array([0.25]))
        sit = Sequential([vrstva])
        x = np.array([[1.0, 2.0], [0.0, -1.0], [3.0, 1.0]])
        g = np.array([[0.1], [-0.2], [0.3]])
        sit.forward(x)
        (d_w, d_b), = sit.backward(g)
        delta = g * vrstva.activation.derivative(x @ vrstva.linear.weights + vrstva.linear.bias)
        assert np.allclose(d_w, x.T @ delta)
        assert np.allclose(d_b, delta.sum(axis=0))

    @STUB
    def test_kontrola_gradientu_vah(self) -> None:
        """Klicovy test: dW kazde vrstvy = numericka derivace MSE podle W."""
        sit, x, y = _dummy_sit()
        gradienty = sit.backward(_mse_gradient(sit.forward(x), y))
        for vrstva, (d_w, _) in zip(sit.layers, gradienty):
            numericky = _numericky_gradient(lambda: _mse(sit.forward(x), y), vrstva.linear.weights)
            assert np.allclose(d_w, numericky, atol=TOL)

    @STUB
    def test_kontrola_gradientu_biasu(self) -> None:
        """db kazde vrstvy = numericka derivace MSE podle biasu."""
        sit, x, y = _dummy_sit()
        gradienty = sit.backward(_mse_gradient(sit.forward(x), y))
        for vrstva, (_, d_b) in zip(sit.layers, gradienty):
            numericky = _numericky_gradient(lambda: _mse(sit.forward(x), y), vrstva.linear.bias)
            assert np.allclose(d_b, numericky, atol=TOL)

    @STUB
    def test_kontrola_gradientu_s_bce(self) -> None:
        """Totez s gradientem BCE ze src/losses.py (jako faze 2 pipeline)."""
        sit, x, y = _dummy_sit()
        ztrata = BCE()
        gradienty = sit.backward(ztrata.gradient(sit.forward(x), y))
        for vrstva, (d_w, d_b) in zip(sit.layers, gradienty):
            assert np.allclose(d_w, _numericky_gradient(lambda: ztrata.forward(sit.forward(x), y),
                                                        vrstva.linear.weights), atol=TOL)
            assert np.allclose(d_b, _numericky_gradient(lambda: ztrata.forward(sit.forward(x), y),
                                                        vrstva.linear.bias), atol=TOL)

    @STUB
    def test_vrstva_bez_biasu_vraci_none(self) -> None:
        """Vrstvy bez biasu vraci db = None, dW zustava spravne."""
        sit, x, y = _dummy_sit(bias=False)
        gradienty = sit.backward(_mse_gradient(sit.forward(x), y))
        assert all(d_b is None for _, d_b in gradienty)
        for vrstva, (d_w, _) in zip(sit.layers, gradienty):
            numericky = _numericky_gradient(lambda: _mse(sit.forward(x), y), vrstva.linear.weights)
            assert np.allclose(d_w, numericky, atol=TOL)

    @STUB
    def test_backward_nemeni_vahy(self) -> None:
        """backward vahy nemeni a nevola update -- to dela az Sequential.update."""
        sit, x, y = _dummy_sit()
        pred = [vrstva.linear.weights.copy() for vrstva in sit.layers]
        sit.backward(_mse_gradient(sit.forward(x), y))
        for vrstva, puvodni in zip(sit.layers, pred):
            assert np.array_equal(vrstva.linear.weights, puvodni)
            assert vrstva.linear.update_calls == []


# --------------------------------------------------------------------------- #
#  Sequential.update                                                         #
# --------------------------------------------------------------------------- #
class TestSequentialUpdate:
    """update deleguje na linear.update kazde vrstvy; krok proti gradientu snizi chybu."""

    @STUB
    def test_deleguje_na_linear_update(self) -> None:
        """Kazda vrstva dostane prave jedno volani update se svym (dW, db) a lr."""
        sit, x, y = _dummy_sit()
        gradienty = sit.backward(_mse_gradient(sit.forward(x), y))
        sit.update(gradienty, 0.05)
        for vrstva, (d_w, d_b) in zip(sit.layers, gradienty):
            assert len(vrstva.linear.update_calls) == 1
            g_w, g_b, lr = vrstva.linear.update_calls[0]
            assert np.array_equal(g_w, d_w) and np.array_equal(g_b, d_b) and lr == 0.05

    @STUB
    def test_maly_krok_snizi_chybu(self) -> None:
        """Jeden maly krok proti gradientu snizi chybu na stejnych datech."""
        sit, x, y = _dummy_sit()
        pred = _mse(sit.forward(x), y)
        sit.update(sit.backward(_mse_gradient(sit.io_[-1], y)), 0.1)
        assert _mse(sit.forward(x), y) < pred


# --------------------------------------------------------------------------- #
#  Sit ze skutecnych Neuronu (vyzaduje branu a derivace)                      #
# --------------------------------------------------------------------------- #
class TestSitZNeuronu:
    """Kontrola gradientu site slozene z Neuron(Linear, Sigmoid) -- cela cesta src/."""

    @STUB
    def test_kontrola_gradientu_se_skutecnymi_vrstvami(self) -> None:
        """Sit 4-3-1 ze skutecnych Neuronu: dW i db odpovidaji numericke derivaci."""
        rng = np.random.default_rng(3)
        sizes = [4, 3, 1]
        sit = Sequential([Neuron(Linear(rng.normal(size=(a, b)), rng.normal(size=b)),
                                 Sigmoid(temperature=0.5))
                          for a, b in zip(sizes[:-1], sizes[1:])])
        x = rng.normal(size=(6, 4))
        y = rng.integers(0, 2, size=(6, 1)).astype(np.float64)
        gradienty = sit.backward(_mse_gradient(sit.forward(x), y))
        for vrstva, (d_w, d_b) in zip(sit.layers, gradienty):
            assert np.allclose(d_w, _numericky_gradient(lambda: _mse(sit.forward(x), y),
                                                        vrstva.linear.weights), atol=TOL)
            assert np.allclose(d_b, _numericky_gradient(lambda: _mse(sit.forward(x), y),
                                                        vrstva.linear.bias), atol=TOL)


# --------------------------------------------------------------------------- #
#  Ztratove funkce                                                           #
# --------------------------------------------------------------------------- #
_P = np.array([[0.9], [0.2], [0.6], [0.35]])
_Y = np.array([[1.0], [0.0], [0.0], [1.0]])
# Dvousloupcovy vystup: odhali deleni poctem radku misto poctem vsech prvku.
_P2 = np.array([[0.9, 0.2], [0.6, 0.35]])
_Y2 = np.array([[1.0, 0.0], [0.0, 1.0]])


class TestMSE:
    """MSE: prumer ctvercu odchylek a jeho gradient (vcetne faktoru 1/N)."""

    @STUB
    def test_hodnota(self) -> None:
        """Hodnota MSE na znamych cislech."""
        # (0.01 + 0.04 + 0.36 + 0.4225) / 4
        assert MSE().forward(_P, _Y) == pytest.approx(0.208125)

    @STUB
    def test_gradient_vs_numericky(self) -> None:
        """Gradient MSE odpovida centralni diferenci."""
        p = _P.copy()
        numericky = _numericky_gradient(lambda: MSE().forward(p, _Y), p)
        assert np.allclose(MSE().gradient(p, _Y), numericky, atol=TOL)

    @STUB
    def test_gradient_vicevystupovy(self) -> None:
        """N = prediction.size (vsechny prvky), ne pocet radku."""
        p = _P2.copy()
        numericky = _numericky_gradient(lambda: MSE().forward(p, _Y2), p)
        assert np.allclose(MSE().gradient(p, _Y2), numericky, atol=TOL)

    @STUB
    def test_nulova_chyba_pri_shode(self) -> None:
        """Pri shode predikce s cilem je ztrata i gradient nulova."""
        assert MSE().forward(_Y, _Y) == pytest.approx(0.0)
        assert np.allclose(MSE().gradient(_Y, _Y), 0.0)


class TestBCE:
    """BCE: krizova entropie, jeji gradient a numericka stabilita (oriznuti)."""

    @STUB
    def test_hodnota_pri_polovicni_jistote(self) -> None:
        """Predikce 0.5 dava BCE = ln 2 bez ohledu na cil."""
        p = np.full((3, 1), 0.5)
        assert BCE().forward(p, np.array([[0.0], [1.0], [1.0]])) == pytest.approx(np.log(2.0))

    @STUB
    def test_hodnota(self) -> None:
        """Hodnota BCE odpovida definici na znamych cislech."""
        ocekavano = -np.mean(_Y * np.log(_P) + (1 - _Y) * np.log(1 - _P))
        assert BCE().forward(_P, _Y) == pytest.approx(ocekavano)

    @STUB
    def test_gradient_vs_numericky(self) -> None:
        """Gradient BCE odpovida centralni diferenci."""
        p = _P.copy()
        numericky = _numericky_gradient(lambda: BCE().forward(p, _Y), p)
        assert np.allclose(BCE().gradient(p, _Y), numericky, rtol=1e-5, atol=TOL)

    @STUB
    def test_gradient_vicevystupovy(self) -> None:
        """N = prediction.size (vsechny prvky), ne pocet radku."""
        p = _P2.copy()
        numericky = _numericky_gradient(lambda: BCE().forward(p, _Y2), p)
        assert np.allclose(BCE().gradient(p, _Y2), numericky, rtol=1e-5, atol=TOL)

    @STUB
    def test_krajni_predikce_nedaji_nekonecno(self) -> None:
        """Diky oriznuti je ztrata i gradient konecny i pro predikce 0 a 1."""
        p = np.array([[0.0], [1.0]])
        y = np.array([[1.0], [0.0]])
        assert np.isfinite(BCE().forward(p, y))
        assert np.all(np.isfinite(BCE().gradient(p, y)))

    @STUB
    def test_bce_se_sigmoidou_dava_p_minus_y(self) -> None:
        """Pro sigmoidu (T=1) je dL/dz = (p - y) / N -- derivace sigmoidy se vykrati."""
        z = np.array([[2.0], [-1.0], [0.5]])
        y = np.array([[1.0], [1.0], [0.0]])
        aktivace = Sigmoid(temperature=1.0)
        p = aktivace.forward(z)
        dl_dz = BCE().gradient(p, y) * aktivace.derivative(z)
        assert np.allclose(dl_dz, (p - y) / 3.0)


def test_make_loss_zna_obe_ztraty_a_odmitne_nezname() -> None:
    """Tovarni funkce je PREDVYPLNENA, neni STUB."""
    assert isinstance(make_loss("mse"), MSE)
    assert isinstance(make_loss("bce"), BCE)
    with pytest.raises(ValueError):
        make_loss("hinge")


# --------------------------------------------------------------------------- #
#  Inicializatory                                                            #
# --------------------------------------------------------------------------- #
class TestInicializatory:
    """Vahy tvaru (n_inputs, n_units), meze a rozptyl podle zvolene strategie."""

    def test_random_uniform_tvar_meze_seed(self) -> None:
        """RandomUniformInit je PREDVYPLNENY vzor, neni STUB."""
        w = RandomUniformInit(-0.5, 0.5, random_state=1).initialize(30, 9)
        assert w.shape == (30, 9)
        assert w.min() >= -0.5 and w.max() <= 0.5
        assert np.array_equal(w, RandomUniformInit(-0.5, 0.5, random_state=1).initialize(30, 9))

    def test_pytorch_default_meze_a_rozptyl(self) -> None:
        """PyTorchDefaultInit je PREDVYPLNENY: U(+-1/sqrt(n_in)), Var = 1/(3 n_in)."""
        w = PyTorchDefaultInit(random_state=0).initialize(300, 400)
        assert w.shape == (300, 400)
        assert np.abs(w).max() <= 1.0 / np.sqrt(300)
        assert w.var() == pytest.approx(1.0 / (3 * 300), rel=0.05)

    @STUB
    def test_xavier_tvar_a_meze(self) -> None:
        """Xavier vraci matici (n_inputs, n_units) v mezich +-sqrt(6/(n_in+n_out))."""
        w = XavierInit(random_state=0).initialize(30, 9)
        assert w.shape == (30, 9)
        assert np.abs(w).max() <= np.sqrt(6.0 / (30 + 9))

    @STUB
    def test_xavier_rozptyl(self) -> None:
        """Rozptyl vah Xavier je 2/(n_in + n_out)."""
        w = XavierInit(random_state=0).initialize(200, 300)
        assert w.var() == pytest.approx(2.0 / (200 + 300), rel=0.05)

    @STUB
    def test_xavier_reprodukovatelny_seedem(self) -> None:
        """Stejny random_state dava stejne vahy (pouziva se self.rng)."""
        assert np.array_equal(XavierInit(random_state=5).initialize(4, 3),
                              XavierInit(random_state=5).initialize(4, 3))

    @STUB
    def test_he_tvar_a_rozptyl(self) -> None:
        """BONUS: He inicializace ma tvar (n_in, n_out) a rozptyl 2/n_in."""
        w = HeInit(random_state=0).initialize(400, 300)
        assert w.shape == (400, 300)
        assert w.var() == pytest.approx(2.0 / 400, rel=0.05)

    def test_tovarni_funkce(self) -> None:
        """make_weights_initializer je PREDVYPLNENA, neni STUB."""
        assert isinstance(make_weights_initializer("xavier", 0), XavierInit)
        assert isinstance(make_weights_initializer("he", 0), HeInit)
        assert isinstance(make_weights_initializer("random_uniform", 0), RandomUniformInit)
        assert isinstance(make_weights_initializer("pytorch_default", 0), PyTorchDefaultInit)
        with pytest.raises(ValueError):
            make_weights_initializer("zeros", 0)


# --------------------------------------------------------------------------- #
#  Trainer (predvyplneny) -- mechanika na DummyModel                         #
# --------------------------------------------------------------------------- #
class DummyModel:
    """Model s rozhranim forward/backward/update, ktery jen pocita volani."""

    def __init__(self) -> None:
        """Jedna DummyLayer (kvuli logovani tvaru) a prazdne zaznamy volani."""
        self.layers = [DummyLayer(np.zeros((2, 1)), np.zeros(1))]
        self.update_learning_rates: list[float] = []
        self.batch_sizes: list[int] = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Konstantni vystup 0.5 tvaru (n, 1)."""
        return np.full((x.shape[0], 1), 0.5)

    def backward(self, loss_gradient: np.ndarray) -> list:
        """Zaznamena velikost davky a vrati nulove gradienty."""
        self.batch_sizes.append(loss_gradient.shape[0])
        return [(np.zeros((2, 1)), np.zeros(1))]

    def update(self, gradients: list, learning_rate: float) -> None:
        """Zaznamena pouzity krok uceni."""
        self.update_learning_rates.append(learning_rate)


class DummyLoss:
    """Ztrata s konstantni hodnotou (nastavitelnou) a nulovym gradientem."""

    def __init__(self, hodnota: float = 1.0) -> None:
        """Ulozi konstantni hodnotu ztraty."""
        self.hodnota = hodnota

    def forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        """Vrati konstantni hodnotu ztraty."""
        return self.hodnota

    def gradient(self, prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Vrati nulovy gradient tvaru predikce."""
        return np.zeros_like(prediction)


_X_MALA = np.arange(20, dtype=np.float64).reshape(10, 2)
_Y_MALA = np.array([0, 1] * 5)


class TestTrainer:
    """Stav train/eval, pocet kroku, davky, predcasne ukonceni, snizovani kroku uceni."""

    def test_step_v_eval_stavu_je_chyba(self) -> None:
        """Ve vyhodnocovacim stavu nesmi step menit vahy -- vyvola RuntimeError."""
        trainer = Trainer(DummyModel(), DummyLoss(), _config(), log_dir=None)
        trainer.eval()
        with pytest.raises(RuntimeError):
            trainer.step(_X_MALA[:1], _Y_MALA[:1].reshape(-1, 1))

    def test_evaluate_nemeni_vahy(self) -> None:
        """evaluate prepne do stavu eval a nevola model.update."""
        model = DummyModel()
        trainer = Trainer(model, DummyLoss(), _config(), log_dir=None)
        trainer.evaluate(_X_MALA, _Y_MALA)
        assert trainer.training is False
        assert model.update_learning_rates == []

    def test_pocet_kroku_odpovida_epocham_a_davkam(self) -> None:
        """Pocet kroku = epochy x ceil(N / batch_size), posledni davka je kratsi."""
        model = DummyModel()
        Trainer(model, DummyLoss(), _config(epochs=3, batch_size=4), log_dir=None).run(
            _X_MALA, _Y_MALA, _X_MALA, _Y_MALA)
        assert len(model.update_learning_rates) == 3 * 3          # ceil(10 / 4) = 3 davky
        assert model.batch_sizes[:3] == [4, 4, 2]

    def test_historie_ma_hodnotu_za_kazdou_epochu(self) -> None:
        """Historie ma pro kazdou epochu jednu hodnotu v kazdem klici."""
        historie = Trainer(DummyModel(), DummyLoss(), _config(epochs=4), log_dir=None).run(
            _X_MALA, _Y_MALA, _X_MALA, _Y_MALA)
        assert historie["epoch"] == [1, 2, 3, 4]
        for klic in ("train_loss", "test_loss", "learning_rate"):
            assert len(historie[klic]) == 4

    def test_predcasne_ukonceni_pod_max_loss(self) -> None:
        """Uceni skonci, jakmile trenovaci ztrata klesne pod max_loss."""
        historie = Trainer(DummyModel(), DummyLoss(0.001), _config(epochs=50, max_loss=0.01),
                           log_dir=None).run(_X_MALA, _Y_MALA, _X_MALA, _Y_MALA)
        assert historie["epoch"] == [1]

    def test_snizovani_kroku_uceni(self) -> None:
        """Krok uceni se kazdych lr_decay_every epoch vynasobi lr_decay."""
        model = DummyModel()
        historie = Trainer(model, DummyLoss(), _config(epochs=5, learning_rate=1.0, lr_decay=0.5,
                                                        lr_decay_every=2), log_dir=None).run(
            _X_MALA, _Y_MALA, _X_MALA, _Y_MALA)
        assert historie["learning_rate"] == pytest.approx([1.0, 1.0, 0.5, 0.5, 0.25])
        assert model.update_learning_rates[-1] == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
#  Uceni end-to-end                                                          #
# --------------------------------------------------------------------------- #
class TestUceniEndToEnd:
    """Kratke uceni site 2-3-1 na dvou oddelenych shlucich chybu vyrazne snizi."""

    @STUB
    def test_uceni_snizi_chybu_a_klasifikuje_spravne(self) -> None:
        """Po 60 epochach klesne ztrata alespon na polovinu a vsechny body jsou spravne."""
        rng = np.random.default_rng(0)
        x = np.vstack([rng.normal(-1.0, 0.3, size=(20, 2)), rng.normal(1.0, 0.3, size=(20, 2))])
        y = np.array([0] * 20 + [1] * 20)
        init = RandomUniformInit(-1.0, 1.0, random_state=0)
        sit = Sequential([Neuron(Linear(init.initialize(2, 3), np.zeros(3)), Sigmoid()),
                          Neuron(Linear(init.initialize(3, 1), np.zeros(1)), Sigmoid())])
        historie = Trainer(sit, BCE(), _config(learning_rate=0.5, epochs=60),
                           random_state=0, log_dir=None).run(x, y, x, y)
        assert historie["train_loss"][-1] < 0.5 * historie["train_loss"][0]
        predikce = (sit.forward(x).ravel() >= 0.5).astype(int)
        assert np.array_equal(predikce, y)


# --------------------------------------------------------------------------- #
#  Metriky (predvyplnene dataio)                                             #
# --------------------------------------------------------------------------- #
class TestMetriky:
    """Matice zamen ve tvaru [[TN, FP], [FN, TP]] a z ni odvozene metriky."""

    Y_TRUE = np.array([1, 1, 1, 0, 0, 0, 0, 1])
    Y_PRED = np.array([1, 0, 1, 0, 1, 0, 0, 1])

    def test_matice_zamen(self) -> None:
        """Matice zamen ma usporadani [[TN, FP], [FN, TP]]."""
        assert np.array_equal(confusion_matrix_2x2(self.Y_TRUE, self.Y_PRED),
                              np.array([[3, 1], [1, 3]]))

    def test_metriky(self) -> None:
        """Accuracy, precision a recall na znamem prikladu."""
        metriky = classification_metrics(self.Y_TRUE, self.Y_PRED)
        assert metriky["accuracy"] == pytest.approx(0.75)
        assert metriky["precision"] == pytest.approx(0.75)
        assert metriky["recall"] == pytest.approx(0.75)

    def test_nulovy_jmenovatel(self) -> None:
        """Pri nulovem jmenovateli vraci metriky 0.0 (jako sklearn, zero_division=0)."""
        metriky = classification_metrics(np.array([0, 0]), np.array([0, 0]))
        assert metriky["precision"] == 0.0 and metriky["recall"] == 0.0


# --------------------------------------------------------------------------- #
#  Data a konfigurace                                                        #
# --------------------------------------------------------------------------- #
class TestDataAKonfigurace:
    """Loader (tvary, standardizace, kodovani trid) a validace konfigurace."""

    def test_loader_tvary_a_standardizace(self) -> None:
        """Tvary rozdeleni, standardizace z trenovacich dat a kodovani 1 = maligni."""
        x_train, y_train, x_test, y_test = load_breast_cancer_data(random_state=42, test_size=0.2)
        assert x_train.shape == (455, 30) and x_test.shape == (114, 30)
        assert np.allclose(x_train.mean(axis=0), 0.0, atol=1e-10)
        assert np.allclose(x_train.std(axis=0), 1.0)
        assert set(np.unique(y_train)) == {0, 1}
        assert int(y_train.sum() + y_test.sum()) == 212      # 212 malignich = trida 1

    def test_vychozi_konfigurace_se_nacte(self) -> None:
        """Vychozi config.yaml se nacte a projde validaci."""
        cfg = load_config()
        assert cfg.training.learning_rate > 0
        assert cfg.loss in ("mse", "bce")
        assert all(k >= 1 for k in cfg.network.hidden_units)

    @pytest.mark.parametrize("sekce, klic, hodnota", [
        ("training", "learning_rate", 0.0),
        ("training", "batch_size", 0),
        ("training", "epochs", 0),
        ("network", "initializer", "zeros"),
        ("network", "sigmoid_temperature", -1.0),
        ("data", "test_size", 1.0),
    ])
    def test_neplatne_hodnoty_jsou_chyba(self, sekce: str, klic: str, hodnota) -> None:
        """Hodnota mimo povoleny rozsah vyvola ve validate_config ValueError."""
        cfg = load_config()
        setattr(getattr(cfg, sekce), klic, hodnota)
        with pytest.raises(ValueError):
            validate_config(cfg)

    def test_neznama_ztrata_je_chyba(self) -> None:
        """Neznamy nazev ztraty vyvola ve validate_config ValueError."""
        cfg = load_config()
        cfg.loss = "hinge"
        with pytest.raises(ValueError):
            validate_config(cfg)
