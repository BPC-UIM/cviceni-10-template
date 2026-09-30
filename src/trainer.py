"""Trenovaci wrapper: sklada jednotlive kroky uceni do celeho procesu (PREDVYPLNENO).

Delba zodpovednosti (klicovy navrhovy princip cviceni):

- **Model** (`Sequential`) umi *jeden krok uceni*: `forward` (predikce a zaznam
  mezistavu), `backward` (gradienty) a `update` (posun vah). Vahy ziji
  v `Linear` a meni je jen `Linear.update`.
- **Trainer** sklada kroky do *procesu*: epochy, michani vzorku, davky,
  sledovani chyby, snizovani kroku uceni, predcasne ukonceni a logovani.
  Do vah nikdy primo nesaha.

Stav `train` / `eval` (obdoba `model.train()` / `model.eval()` v PyTorch):
v treninkovem stavu `step` aktualizuje vahy, ve vyhodnocovacim stavu se
provadi pouze dopredny pruchod (`evaluate`) a volani `step` je chyba.

Logovani (prvni cviceni s dlouhym iteracnim behem, kde dava smysl misto
`print`): pojmenovany logger `logging.getLogger(__name__)` s
`propagate = False` a dvema handlery - konzole (uroven INFO, milniky uceni)
a rotujici soubor `logs/trainer.log` (uroven DEBUG, zaznam kazde epochy).
Do logu se zapisuji jen skalary a statistiky (hodnoty chyby, normy vah),
nikdy cela pole ani jednotlive vzorky.
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dataio.config_manager import TrainingConfig
    from src.losses import Loss
    from src.network import Sequential

LOG_FILE_NAME = "trainer.log"
_LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)-7s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class _ConsoleHandler(logging.StreamHandler):
    """Konzolovy handler, ktery vzdy pise do aktualniho `sys.stdout`.

    Beznemu `StreamHandler` se proud preda jednou pri vytvoreni; pokud jej
    nekdo mezitim vymeni (napr. pytest pri zachytavani vystupu), handler by
    psal do uzavreneho proudu. Tato varianta si proud zjisti pri kazdem zapisu.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Nastavi aktualni `sys.stdout` a zaznam zapise."""
        self.stream = sys.stdout
        super().emit(record)


def get_logger(log_dir: str | None = "logs") -> logging.Logger:
    """Vrati logger modulu, pri prvnim volani ho nakonfiguruje.

    Konfigurace probehne **jen jednou** za beh programu (kontrola, zda uz
    logger ma handlery) - opakovane volani tak nezpusobi zdvojeny vystup.

    Parametry
    ---------
    log_dir : str | None, optional
        Adresar pro rotujici soubor logu; `None` znamena jen konzoli.

    Navratova hodnota
    -----------------
    logging.Logger
        Logger `src.trainer` s urovni DEBUG a `propagate = False`.
    """
    logger = logging.getLogger(__name__)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # zadne zdvojeni pres korenovy logger

    console = _ConsoleHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("  [%(levelname)s] %(message)s"))
    logger.addHandler(console)

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = RotatingFileHandler(
            os.path.join(log_dir, LOG_FILE_NAME),
            maxBytes=1_000_000, backupCount=3, encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(file_handler)

    return logger


def _as_column(y: np.ndarray) -> np.ndarray:
    """Prevede vektor cilu `(n,)` na sloupec `(n, 1)` typu float (tvar vystupu site)."""
    y = np.asarray(y, dtype=np.float64)
    return y.reshape(-1, 1) if y.ndim == 1 else y


class Trainer:
    """Ridi uceni modelu gradientnim sestupem (PREDVYPLNENO).

    Model, ztrata i konfigurace se injektuji zvenku (dependency injection).
    Trainer vola jen verejne rozhrani modelu (`forward`, `backward`,
    `update`) a ztraty (`forward`, `gradient`).

    Atributy
    --------
    model : Sequential
        Uceny model.
    loss : Loss
        Optimalizovana ztratova funkce.
    config : TrainingConfig
        Hyperparametry uceni (krok, epochy, davka, prah chyby, snizovani kroku).
    training : bool
        `True` v treninkovem stavu, `False` ve vyhodnocovacim.
    learning_rate_ : float
        Aktualni krok uceni (behem uceni se snizuje podle `lr_decay`).
    history_ : dict[str, list[float]] | None
        Prubeh uceni posledniho `run`; pred prvnim `run` je `None`.
    """

    def __init__(
        self,
        model: "Sequential",
        loss: "Loss",
        config: "TrainingConfig",
        random_state: int | None = None,
        log_dir: str | None = "logs",
        ) -> None:
        """Ulozi model, ztratu a konfiguraci; pripravi generator a logger.

        Parametry
        ---------
        model : Sequential
            Model s metodami `forward`, `backward`, `update`.
        loss : Loss
            Ztratova funkce s metodami `forward`, `gradient`.
        config : TrainingConfig
            Hyperparametry uceni (sekce `training` v `config.yaml`).
        random_state : int | None, optional
            Seed generatoru pro michani vzorku v kazde epose.
        log_dir : str | None, optional
            Adresar pro soubor logu (vychozi `logs`); `None` = jen konzole.
        """
        self.model = model
        self.loss = loss
        self.config = config
        self.rng = np.random.default_rng(random_state)
        self.logger = get_logger(log_dir)
        self.training: bool = True
        self.learning_rate_: float = config.learning_rate
        self.history_: dict[str, list[float]] | None = None

    # ------------------------------------------------------------------ #
    #  Stav train / eval                                                 #
    # ------------------------------------------------------------------ #
    def train(self) -> None:
        """Prepne do treninkoveho stavu: `step` smi aktualizovat vahy."""
        self.training = True

    def eval(self) -> None:
        """Prepne do vyhodnocovaciho stavu: jen dopredny pruchod, zadne zmeny vah."""
        self.training = False

    # ------------------------------------------------------------------ #
    #  Jeden krok a vyhodnoceni                                          #
    # ------------------------------------------------------------------ #
    def step(self, x_batch: np.ndarray, y_batch: np.ndarray) -> float:
        """Provede jeden krok uceni na jedne davce.

        forward -> loss.forward -> loss.gradient -> model.backward -> model.update

        Parametry
        ---------
        x_batch : np.ndarray
            Vstupy davky, tvar `(n_davka, n_vstupu)`.
        y_batch : np.ndarray
            Cile davky, tvar `(n_davka, n_vystupu)`.

        Navratova hodnota
        -----------------
        float
            Hodnota ztraty na davce **pred** aktualizaci vah.

        Vyjimky
        -------
        ``RuntimeError``:
            Pokud je trainer ve vyhodnocovacim stavu (`eval`).
        """
        if not self.training:
            raise RuntimeError("Trainer.step lze volat jen v treninkovem stavu (trainer.train()).")

        prediction = self.model.forward(x_batch)
        loss_value = self.loss.forward(prediction, y_batch)
        gradients = self.model.backward(self.loss.gradient(prediction, y_batch))
        self.model.update(gradients, self.learning_rate_)
        return loss_value

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        """Vrati ztratu modelu na datech; prepne do stavu `eval` (vahy se nemeni).

        Parametry
        ---------
        x : np.ndarray
            Vstupy, tvar `(n_vzorku, n_vstupu)`.
        y : np.ndarray
            Cile, tvar `(n_vzorku,)` nebo `(n_vzorku, n_vystupu)`.

        Navratova hodnota
        -----------------
        float
            Hodnota ztraty na celych datech.
        """
        self.eval()
        return float(self.loss.forward(self.model.forward(x), _as_column(y)))

    # ------------------------------------------------------------------ #
    #  Cely proces uceni                                                 #
    # ------------------------------------------------------------------ #
    def run(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        ) -> dict[str, list[float]]:
        """Provede cele uceni a vrati jeho prubeh.

        Prubeh jedne epochy
        -------------------
        1. `train()`; nahodna permutace poradi trenovacich vzorku.
        2. Pruchod davkami velikosti `batch_size` (vychozi 1 = stochasticky
           gradientni sestup): pro kazdou davku `step`.
        3. `eval()`; ztrata na celych trenovacich a testovacich datech.
        4. Ukonceni, klesne-li trenovaci ztrata pod `max_loss`.
        5. Kazdych `lr_decay_every` epoch se krok uceni vynasobi `lr_decay`.

        Parametry
        ---------
        x_train, y_train : np.ndarray
            Trenovaci vstupy `(n, n_vstupu)` a cile `(n,)`.
        x_test, y_test : np.ndarray
            Testovaci vstupy a cile (jen se sleduji, uceni je nepouziva).

        Navratova hodnota
        -----------------
        dict[str, list[float]]
            Klice `epoch`, `train_loss`, `test_loss`, `learning_rate` - jedna
            hodnota za kazdou probehlou epochu.
        """
        cfg = self.config
        y_train_col, y_test_col = _as_column(y_train), _as_column(y_test)
        n_samples = x_train.shape[0]
        log_every = max(1, cfg.epochs // 10)
        self.learning_rate_ = cfg.learning_rate
        history: dict[str, list[float]] = {
            "epoch": [], "train_loss": [], "test_loss": [], "learning_rate": [],
        }

        shapes = " -> ".join(
            [str(x_train.shape[1])] + [str(layer.linear.weights.shape[1])
                                       for layer in self.model.layers]
        )
        self.logger.info(
            "Start uceni: architektura %s, ztrata %s, %d trenovacich / %d testovacich vzorku",
            shapes, type(self.loss).__name__, n_samples, x_test.shape[0],
        )
        self.logger.info(
            "Hyperparametry: learning_rate=%g, epochs=%d, batch_size=%d, max_loss=%g, "
            "lr_decay=%g kazdych %d epoch",
            cfg.learning_rate, cfg.epochs, cfg.batch_size, cfg.max_loss,
            cfg.lr_decay, cfg.lr_decay_every,
        )

        for epoch in range(1, cfg.epochs + 1):
            self.train()
            order = self.rng.permutation(n_samples)
            batch_losses = []
            for start in range(0, n_samples, cfg.batch_size):
                idx = order[start:start + cfg.batch_size]
                batch_losses.append(self.step(x_train[idx], y_train_col[idx]))

            train_loss = self.evaluate(x_train, y_train_col)
            test_loss = self.evaluate(x_test, y_test_col)
            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["test_loss"].append(test_loss)
            history["learning_rate"].append(self.learning_rate_)

            self.logger.debug(
                "epocha %4d | lr %.5f | prumer davek %.5f | train %.5f | test %.5f | normy vah %s",
                epoch, self.learning_rate_, float(np.mean(batch_losses)), train_loss, test_loss,
                self._weight_norms(),
            )
            if epoch == 1 or epoch % log_every == 0:
                self.logger.info("epocha %4d/%d  train %.4f  test %.4f",
                                 epoch, cfg.epochs, train_loss, test_loss)

            if not np.isfinite(train_loss):
                self.logger.warning("Ztrata neni konecna (%s) - uceni diverguje, koncim. "
                                    "Zkuste mensi learning_rate.", train_loss)
                break
            if train_loss <= cfg.max_loss:
                self.logger.info(
                    "Trenovaci ztrata %.4f <= max_loss %g - uceni ukonceno v epose %d.",
                    train_loss,
                    cfg.max_loss,
                    epoch,
                    )
                break
            if epoch % cfg.lr_decay_every == 0:
                self.learning_rate_ *= cfg.lr_decay
                self.logger.debug("Krok uceni snizen na %.5f.", self.learning_rate_)

        self.logger.info("Konec uceni po %d epochach: train %.4f, test %.4f.",
                         history["epoch"][-1], history["train_loss"][-1], history["test_loss"][-1])
        self.eval()
        self.history_ = history
        return history

    def _weight_norms(self) -> str:
        """Frobeniovy normy matic vah vsech vrstev jako kratky retezec (jen statistika)."""
        return "[" + ", ".join(f"{np.linalg.norm(layer.linear.weights):.3f}"
                               for layer in self.model.layers) + "]"
