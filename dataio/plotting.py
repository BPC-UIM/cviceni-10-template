"""Vykreslovani vysledku cviceni 10 -- vse PREDVYPLNENE.

Dva grafy z pipeline:

* ``plot_loss_curve`` -- prubeh trenovaci a testovaci ztraty behem uceni
  (krivka uceni); na druhe ose krok uceni, aby bylo videt jeho snizovani.
* ``plot_confusion_matrix`` -- matice zamen 2x2 na testovacich datech
  s metrikami presnost / preciznost / senzitivita (navaznost na Cviceni 06).

Obrazek ztratove krajiny v README **neni** soucasti tohoto modulu ani
pipeline -- je to autorsky obrazek generovany skriptem v ``docs/code/``.

Vsechny funkce pouzivaji neinteraktivni backend ``Agg``: figuru sestavi,
volitelne ulozi do ``save_path`` (vcetne vytvoreni nadrazeneho adresare) a
vzdy ji zavrou. Funkce ``plt.show`` se nikdy nevola.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # neinteraktivni backend, vykreslujeme jen do souboru

import matplotlib.pyplot as plt  # noqa: E402  (musi az po matplotlib.use)
import numpy as np  # noqa: E402

from dataio.metrics import classification_metrics, confusion_matrix_2x2  # noqa: E402

# --- Paleta (shodna s Cvicenim 09) --------------------------------------------
_TRAIN_COLOR = "#2a78d6"     # modra
_TEST_COLOR = "#eb6834"      # oranzova
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_MUTED = "#898781"
_HAIRLINE = "#c3c2b7"
_CELL_CORRECT = "#d5e5f8"    # svetle modra -- spravne klasifikovane (diagonala)
_CELL_WRONG = "#fadccf"      # svetle oranzova -- chyby


def _save_and_close(fig: plt.Figure, save_path: str | None) -> None:
    """Pomocna funkce: ulozi figuru do ``save_path`` a zavre ji.

    Pokud je ``save_path`` ``None``, figura se pouze zavre. Nadrazeny
    adresar se v pripade potreby vytvori.
    """
    if save_path is not None:
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        fig.savefig(save_path, dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _style_axes(ax: plt.Axes) -> None:
    """Sjednoti vzhled os: tlumene ramecky a popisky."""
    for spine in ax.spines.values():
        spine.set_color(_HAIRLINE)
    ax.tick_params(colors=_INK_SECONDARY, labelsize=9)
    ax.xaxis.label.set_color(_INK_SECONDARY)
    ax.yaxis.label.set_color(_INK_SECONDARY)
    ax.title.set_color(_INK)


def plot_loss_curve(history: dict[str, list[float]], save_path: str | None = None,
                    loss_name: str = "ztrata") -> None:
    """Vykresli prubeh trenovaci a testovaci ztraty po epochach.

    Parametry
    ---------
    history:
        Slovnik z ``Trainer.run`` s klici ``epoch``, ``train_loss``,
        ``test_loss``, ``learning_rate``.
    save_path:
        Cesta k vystupnimu .png; ``None`` = neukladat.
    loss_name:
        Popisek osy y (napr. ``"BCE"``).
    """
    epochs = np.asarray(history["epoch"])
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)

    ax.plot(epochs, history["train_loss"], color=_TRAIN_COLOR, lw=2.0, label="trenovaci data")
    ax.plot(
        epochs,
        history["test_loss"],
        color=_TEST_COLOR,
        lw=2.0,
        ls="--",
        label="testovaci data",
        )
    ax.set_xlabel("epocha")
    ax.set_ylabel(loss_name)
    ax.set_title("Krivka uceni")
    ax.grid(True, color=_HAIRLINE, lw=0.5, alpha=0.6)
    _style_axes(ax)

    # Krok uceni na vedlejsi ose -- schodovity prubeh odpovida lr_decay.
    ax_lr = ax.twinx()
    ax_lr.plot(epochs, history["learning_rate"], color=_MUTED, lw=1.0, drawstyle="steps-post",
               label="krok uceni")
    ax_lr.set_ylim(0.0, 1.1 * max(history["learning_rate"]))
    ax_lr.set_ylabel("krok uceni", color=_MUTED)
    ax_lr.tick_params(colors=_MUTED, labelsize=8)
    for spine in ax_lr.spines.values():
        spine.set_color(_HAIRLINE)

    handles = ax.get_legend_handles_labels()[0] + ax_lr.get_legend_handles_labels()[0]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="center right")
    _save_and_close(fig, save_path)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          save_path: str | None = None) -> dict[str, float]:
    """Vykresli matici zamen 2x2 s metrikami a metriky vrati.

    Parametry
    ---------
    y_true:
        Skutecne tridy ``{0, 1}`` (1 = maligni).
    y_pred:
        Predikovane tridy ``{0, 1}``.
    save_path:
        Cesta k vystupnimu .png; ``None`` = neukladat.

    Navratova hodnota
    -----------------
    ``dict`` s klici ``accuracy``, ``precision``, ``recall``
    (viz ``dataio.metrics.classification_metrics``).
    """
    matrix = confusion_matrix_2x2(y_true, y_pred)
    metrics = classification_metrics(y_true, y_pred)
    names = ("benigni (0)", "maligni (1)")
    abbreviations = (("TN", "FP"), ("FN", "TP"))

    fig, ax = plt.subplots(figsize=(5.4, 4.9))
    fig.patch.set_facecolor(_SURFACE)
    for i in range(2):
        for j in range(2):
            color = _CELL_CORRECT if i == j else _CELL_WRONG
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor="white", lw=3))
            ax.text(j + 0.5, i + 0.45, str(matrix[i, j]), ha="center", va="center",
                    fontsize=22, color=_INK)
            ax.text(j + 0.5, i + 0.75, abbreviations[i][j], ha="center", va="center",
                    fontsize=9, color=_INK_SECONDARY)

    ax.set_xlim(0, 2)
    ax.set_ylim(2, 0)
    ax.set_xticks([0.5, 1.5], names)
    ax.set_yticks([0.5, 1.5], names)
    ax.set_xlabel("predikovana trida")
    ax.set_ylabel("skutecna trida")
    ax.set_title("Matice zamen (testovaci data)")
    ax.set_aspect("equal")
    _style_axes(ax)

    summary = (f"presnost {metrics['accuracy']:.3f}   "
               f"preciznost {metrics['precision']:.3f}   "
               f"senzitivita {metrics['recall']:.3f}")
    fig.subplots_adjust(bottom=0.2)
    fig.text(0.5, 0.02, summary, ha="center", va="bottom", fontsize=9, color=_INK_SECONDARY)
    _save_and_close(fig, save_path)
    return metrics
