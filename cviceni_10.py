# -*- coding: utf-8 -*-

"""
Created on 29. 09. 2026 at 10:00:00

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
    Vstupni bod cviceni 10 (uceni site gradientnim sestupem -- backpropagation).
    Pipeline projde sest fazi:

      1. data -- Breast Cancer Wisconsin, 30 priznaku, standardizace,
         rozdeleni na trenovaci a testovaci cast,
      2. kontrola gradientu -- gradienty z Sequential.backward se porovnaji
         s numerickou derivaci (centralni diference) na male nahodne siti,
      3. sestaveni site -- vrstvy Neuron(Linear, Sigmoid), vahy z inicializatoru
         zvoleneho v config.yaml,
      4. jeden krok uceni -- forward, ztrata, backward a update na jedinem
         vzorku (na kopii site); vypise, jak se zmenila chyba,
      5. uceni -- Trainer.run (epochy, michani, davky, logovani do logs/),
      6. vyhodnoceni -- matice zamen 2x2, metriky a krivka uceni do graphs/.

    Cela pipeline je predvyplnena. Studentske ukoly jsou v src/ (derivace
    aktivaci, Linear.update, Sequential.forward/backward/update, ztraty,
    XavierInit) -- viz README, Pokyny k vypracovani.

    Repozitar bezi v kazdem stavu. Dokud nejsou ukoly hotove, faze se
    zastavi jen hlaskou [NENI HOTOVO] Úkol: ... a pipeline pokracuje dal --
    nikdy nezpracovanym tracebackem. Neni-li hotovy XavierInit, faze 3
    pouzije predvyplneny RandomUniformInit, aby slo ucit uz pred jeho
    dokoncenim.
================================================================================
"""

from __future__ import annotations

import copy
import sys
import time

import numpy as np

# --- Import guard: srozumitelna hlaska misto holeho ImportError ----------------
try:
    from dataio import (
        ExperimentConfig,
        load_breast_cancer_data,
        load_config,
        plot_confusion_matrix,
        plot_loss_curve,
    )
    from src import (
        Linear,
        Loss,
        Neuron,
        RandomUniformInit,
        Sequential,
        Sigmoid,
        Trainer,
        WeightsInitializer,
        make_loss,
        make_weights_initializer,
    )
except ImportError as exc:  # pragma: no cover - jen ochranna hlaska
    print(f"[CHYBA IMPORTU] Nepodarilo se nacist moduly projektu: {exc}")
    print("Zkontrolujte, ze spoustite skript z korene repozitare a mate "
          "nainstalovane zavislosti (pip install -r requirements.txt).")
    sys.exit(1)

GRAPHS_DIR = "graphs"          # vystupni grafy (.png)
LOGS_DIR = "logs"              # soubor logu uceni (trainer.log)

# Kontrola gradientu: relativni chyba pod touto mezi = backward je spravne.
PRAH_KONTROLY_GRADIENTU = 1e-5


# ============================================================================ #
#  Pomocne funkce (predvyplneno)                                               #
# ============================================================================ #
def _banner(text: str) -> None:
    """Vypise oddelovaci nadpis faze pipeline."""
    print("\n" + "=" * 78)
    print(f"  {text}")
    print("=" * 78)


def _faze_neni_hotova(exc: NotImplementedError) -> None:
    """Vypise pratelskou hlasku, kdyz faze narazi na nedokonceny ukol."""
    print(f"  [NENI HOTOVO] {exc}")
    print("  -> Tuto cast dokoncite v ramci ukolu; pipeline pokracuje dal.")


def _chyba_implementace(exc: Exception) -> None:
    """Vypise hlasku, kdyz dokoncena cast selze (assert, spatny tvar pole)."""
    print(f"  [CHYBA IMPLEMENTACE] {type(exc).__name__}: {exc}")
    print("  -> Zkontrolujte tvary poli a sve asserty (README, Pokyny k vypracovani); "
          "pipeline pokracuje dal.")


def _sestav_sit(
    n_inputs: int,
    cfg: ExperimentConfig,
    initializer: WeightsInitializer
    ) -> Sequential:
    """Sestavi sit ``n_inputs - hidden_units... - 1`` ze sigmoidovych vrstev.

    Kazda vrstva je ``Neuron(Linear(W, b), Sigmoid(T))``; ``W`` tvaru
    ``(n_vstupu, n_jednotek)`` dodava inicializator, bias zacina na nule.

    Parametry
    ---------
    n_inputs : int
        Pocet vstupnich priznaku.
    cfg : ExperimentConfig
        Konfigurace (``network.hidden_units``, ``network.sigmoid_temperature``).
    initializer : WeightsInitializer
        Inicializator vah (injektovany zvenku).

    Navratova hodnota
    -----------------
    Sequential
        Neucena sit.
    """
    sizes = [n_inputs] + list(cfg.network.hidden_units) + [1]
    layers = []
    for n_in, n_out in zip(sizes[:-1], sizes[1:]):
        weights = np.asarray(initializer.initialize(n_in, n_out), dtype=np.float64)
        linear = Linear(weights, np.zeros(n_out))
        layers.append(Neuron(linear, Sigmoid(temperature=cfg.network.sigmoid_temperature)))
    return Sequential(layers)


def _predikce(vystup: np.ndarray) -> np.ndarray:
    """Prevede vystup site ``(n, 1)`` na tridy 0/1 prahem 0.5."""
    return (np.asarray(vystup, dtype=np.float64).reshape(-1) >= 0.5).astype(int)


def _numericky_gradient(sit: Sequential, loss: Loss, x: np.ndarray, y: np.ndarray,
                        parametr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Centralni diference ``(L(p + eps) - L(p - eps)) / (2 eps)`` pro kazdy prvek parametru.

    Parametr (matice vah nebo vektor biasu) se docasne meni **na miste**
    a po kazdem vypoctu se vraci na puvodni hodnotu.
    """
    gradient = np.zeros_like(parametr)
    for index in np.ndindex(parametr.shape):
        puvodni = parametr[index]
        parametr[index] = puvodni + eps
        l_plus = loss.forward(sit.forward(x), y)
        parametr[index] = puvodni - eps
        l_minus = loss.forward(sit.forward(x), y)
        parametr[index] = puvodni
        gradient[index] = (l_plus - l_minus) / (2.0 * eps)
    return gradient


def _relativni_chyba(a: np.ndarray, b: np.ndarray) -> float:
    """Maximalni relativni rozdil ``|a - b| / max(|a| + |b|, 1e-8)`` pres vsechny prvky."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(a) + np.abs(b), 1e-8)))


# ============================================================================ #
#  Faze pipeline (predvyplneno)                                                #
# ============================================================================ #
def faze_data(cfg: ExperimentConfig) -> tuple[np.ndarray, ...]:
    """Faze 1 -- nacteni a priprava dat."""
    _banner("Faze 1: Data -- Breast Cancer Wisconsin (30 priznaku)")
    x_train, y_train, x_test, y_test = load_breast_cancer_data(
        random_state=cfg.data.random_state, test_size=cfg.data.test_size,
    )
    print(f"  trenovaci data: x {x_train.shape}, y {y_train.shape}, "
          f"maligni {int(y_train.sum())} / {len(y_train)}")
    print(f"  testovaci data: x {x_test.shape}, y {y_test.shape}, "
          f"maligni {int(y_test.sum())} / {len(y_test)}")
    print(f"  standardizace (z trenovacich dat): prumer priznaku ~ "
          f"{abs(x_train.mean()):.1e}, smerodatna odchylka ~ {x_train.std():.3f}")
    return x_train, y_train, x_test, y_test


def faze_kontrola_gradientu(cfg: ExperimentConfig) -> None:
    """Faze 2 -- porovnani ``Sequential.backward`` s numerickou derivaci.

    Mala sit 3-4-2-1 s nahodnymi vahami a 6 nahodnych vzorku. Pro kazdy
    parametr kazde vrstvy se spocita centralni diference ztraty a porovna
    s gradientem z ``backward``. Shoda na ~1e-7 znamena, ze zpetny pruchod
    (vcetne derivaci aktivaci a gradientu ztraty) je spravne.
    """
    _banner("Faze 2: Kontrola gradientu -- backward vs. numericka derivace")
    rng = np.random.default_rng(0)
    sizes = [3, 4, 2, 1]
    sit = Sequential([
        Neuron(Linear(rng.normal(size=(n_in, n_out)), rng.normal(size=n_out)),
               Sigmoid(temperature=cfg.network.sigmoid_temperature))
        for n_in, n_out in zip(sizes[:-1], sizes[1:])
    ])
    x = rng.normal(size=(6, 3))
    y = rng.integers(0, 2, size=(6, 1)).astype(np.float64)
    loss = make_loss(cfg.loss)

    try:
        vystup = sit.forward(x)
        gradienty = sit.backward(loss.gradient(vystup, y))
        if len(gradienty) != len(sit.layers):
            print(
                f"  [CHYBA] backward vratil {len(gradienty)} dvojic, ocekavano {len(sit.layers)}."
                )
            return

        nejhorsi = 0.0
        for k, (vrstva, (d_w, d_b)) in enumerate(zip(sit.layers, gradienty), start=1):
            if np.shape(d_w) != vrstva.linear.weights.shape:
                print(f"  [CHYBA] vrstva {k}: dW ma tvar {np.shape(d_w)}, "
                      f"ocekavan {vrstva.linear.weights.shape}.")
                return
            chyba_w = _relativni_chyba(
                d_w,
                _numericky_gradient(sit, loss, x, y, vrstva.linear.weights)
                )
            chyba_b = _relativni_chyba(
                d_b,
                _numericky_gradient(sit, loss, x, y, vrstva.linear.bias)
                )
            nejhorsi = max(nejhorsi, chyba_w, chyba_b)
            print(f"  vrstva {k}: W {vrstva.linear.weights.shape}  rel. chyba dW {chyba_w:.2e}, "
                  f"db {chyba_b:.2e}")

        if nejhorsi < PRAH_KONTROLY_GRADIENTU:
            print(f"  [OK] backward odpovida numericke derivaci (max. rel. chyba {nejhorsi:.2e}).")
        else:
            print(f"  [CHYBA] max. rel. chyba {nejhorsi:.2e} >= {PRAH_KONTROLY_GRADIENTU:.0e} -- "
                  "zkontrolujte znamenka, transpozice a derivace aktivaci.")
    except NotImplementedError as exc:
        _faze_neni_hotova(exc)
    except (AssertionError, ValueError) as exc:
        _chyba_implementace(exc)


def faze_sestaveni(cfg: ExperimentConfig, n_inputs: int) -> Sequential:
    """Faze 3 -- sestaveni site s vahami z inicializatoru."""
    _banner("Faze 3: Sestaveni site -- inicializace vah")
    architektura = "-".join(str(k) for k in [n_inputs, *cfg.network.hidden_units, 1])
    print(f"  architektura {architektura}, aktivace Sigmoid(T={cfg.network.sigmoid_temperature}), "
          f"inicializator '{cfg.network.initializer}'")

    initializer = make_weights_initializer(
        cfg.network.initializer,
        random_state=cfg.data.random_state
        )
    try:
        sit = _sestav_sit(n_inputs, cfg, initializer)
    except (NotImplementedError, AssertionError, ValueError) as exc:
        if isinstance(exc, NotImplementedError):
            _faze_neni_hotova(exc)
        else:
            _chyba_implementace(exc)
        print("  -> Nahradne pouzivam predvyplneny RandomUniformInit(-1, 1).")
        sit = _sestav_sit(n_inputs, cfg, RandomUniformInit(random_state=cfg.data.random_state))

    for k, vrstva in enumerate(sit.layers, start=1):
        w = vrstva.linear.weights
        print(f"  vrstva {k}: W {w.shape}, b {vrstva.linear.bias.shape}, "
              f"smerodatna odchylka vah {w.std():.3f}")
    return sit


def faze_jeden_krok(cfg: ExperimentConfig, sit: Sequential,
                    x_train: np.ndarray, y_train: np.ndarray) -> None:
    """Faze 4 -- jeden krok gradientniho sestupu na jedinem vzorku (na kopii site)."""
    _banner("Faze 4: Jeden krok uceni na jednom vzorku (stochasticky gradientni sestup)")
    kopie = copy.deepcopy(sit)          # skutecna sit zustane neucena
    loss = make_loss(cfg.loss)
    x = x_train[:1]
    y = np.asarray(y_train[:1], dtype=np.float64).reshape(1, 1)
    lr = cfg.training.learning_rate

    try:
        vystup_pred = kopie.forward(x)
        chyba_pred = loss.forward(vystup_pred, y)
        gradienty = kopie.backward(loss.gradient(vystup_pred, y))
        vahy_pred = [vrstva.linear.weights.copy() for vrstva in kopie.layers]
        kopie.update(gradienty, lr)
        vystup_po = kopie.forward(x)
        chyba_po = loss.forward(vystup_po, y)

        print(f"  vzorek 0, cil y = {int(y[0, 0])}, learning_rate = {lr}")
        print(f"  pred krokem: vystup {float(vystup_pred[0, 0]):.6f}, ztrata {chyba_pred:.6f}")
        print(f"  po kroku:    vystup {float(vystup_po[0, 0]):.6f}, ztrata {chyba_po:.6f}")
        for k, (vrstva, puvodni) in enumerate(zip(kopie.layers, vahy_pred), start=1):
            zmena = vrstva.linear.weights - puvodni
            print(f"  vrstva {k}: max |zmena vahy| {np.abs(zmena).max():.2e}")
        if chyba_po < chyba_pred:
            print("  [OK] Krok proti gradientu chybu na tomto vzorku snizil.")
        else:
            print("  [POZOR] Chyba se nesnizila -- zkontrolujte znamenko v Linear.update "
                  "(krok ma jit PROTI gradientu), pripadne zmensete learning_rate.")
    except NotImplementedError as exc:
        _faze_neni_hotova(exc)
    except (AssertionError, ValueError) as exc:
        _chyba_implementace(exc)


def faze_uceni(cfg: ExperimentConfig, sit: Sequential,
               data: tuple[np.ndarray, ...]) -> dict[str, list[float]] | None:
    """Faze 5 -- uceni site tridou Trainer (logovani do konzole a do logs/)."""
    _banner("Faze 5: Uceni -- Trainer.run")
    trainer = Trainer(sit, make_loss(cfg.loss), cfg.training,
                      random_state=cfg.data.random_state, log_dir=LOGS_DIR)
    try:
        start = time.perf_counter()
        history = trainer.run(*data)
        print(f"  doba uceni: {time.perf_counter() - start:.1f} s; "
              f"uplny zaznam kazde epochy: {LOGS_DIR}/trainer.log")
        return history
    except NotImplementedError as exc:
        _faze_neni_hotova(exc)
    except (AssertionError, ValueError) as exc:
        _chyba_implementace(exc)
        return None


def faze_vyhodnoceni(cfg: ExperimentConfig, sit: Sequential, history: dict[str, list[float]] | None,
                     x_test: np.ndarray, y_test: np.ndarray) -> None:
    """Faze 6 -- matice zamen, metriky a krivka uceni."""
    _banner("Faze 6: Vyhodnoceni na testovacich datech")
    if history is None:
        print("  [PRESKOCENO] Sit nebyla naucena (faze 5 neprobehla).")
        return

    y_pred = _predikce(sit.forward(x_test))
    metriky = plot_confusion_matrix(y_test, y_pred, save_path=f"{GRAPHS_DIR}/matice_zamen.png")
    plot_loss_curve(history, save_path=f"{GRAPHS_DIR}/krivka_uceni.png", loss_name=cfg.loss.upper())
    print(f"  presnost (accuracy)      {metriky['accuracy']:.3f}")
    print(f"  preciznost (precision)   {metriky['precision']:.3f}")
    print(f"  senzitivita (recall)     {metriky['recall']:.3f}")
    print(f"  grafy ulozeny: {GRAPHS_DIR}/matice_zamen.png, {GRAPHS_DIR}/krivka_uceni.png")


def main() -> None:
    """Spusti celou pipeline cviceni 10 s ochrannymi bloky u kazde faze."""
    _banner("CVICENI 10 -- Uceni site gradientnim sestupem (backpropagation) -- start")

    # --- Config guard -----------------------------------------------------------
    try:
        cfg = load_config()
    except (ValueError, FileNotFoundError) as exc:
        print(f"[CHYBA KONFIGURACE] {exc}")
        sys.exit(1)

    x_train, y_train, x_test, y_test = faze_data(cfg)
    faze_kontrola_gradientu(cfg)
    sit = faze_sestaveni(cfg, n_inputs=x_train.shape[1])
    faze_jeden_krok(cfg, sit, x_train, y_train)
    history = faze_uceni(cfg, sit, (x_train, y_train, x_test, y_test))
    faze_vyhodnoceni(cfg, sit, history, x_test, y_test)

    _banner("CVICENI 10 -- konec")


if __name__ == "__main__":
    main()
