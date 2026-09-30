"""Obrazek ztratove krajiny jednoho neuronu pro README cviceni 10 (autorsky skript).

Neni soucasti pipeline ani studentskych ukolu -- student vidi jen vysledny
obrazek `docs/img/loss_landscape.png` v README.

Levy panel: dve skupiny bodu (make_blobs, centra [0, 0] a [1, 1], mirny
prekryv) a rozhodovaci primky sigmoidoveho neuronu s pevnymi vahami
w = (W, W) pro sedm hodnot biasu b: optimum a po trech hodnotach na kazde
strane.
Pravy panel: ztrata L(b) (MSE) **jen v techto sedmi bodech** a v kazdem z nich
kratka tecna se sklonem dL/db (gradient). Skutecny prubeh L(b) je vykreslen
jen jako slaba carkovana krivka: v praxi ho nezname -- ucici algoritmus zna
v kazdem kroku jen hodnotu a gradient v jedinem bode. Se sigmoidou ma L(b)
tvar misky se zplostelymi konci: tam neuron saturuje a gradient je temer
nulovy.

Spusteni (z korene repozitare):  python docs/code/generate_loss_landscape.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.datasets import make_blobs  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "img"

SURFACE, INK, INK2, MUTED, HAIR = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
GRAD = "#b5493a"          # cihlova (shodna s diagramem backpropagation)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

W = 3.0                   # pevne vahy neuronu w = (W, W)
T = 1.0                   # teplota sigmoidy
B_OTHERS = (-12.0, -8.0, -5.0, -1.0, 2.0, 6.0)   # po trech hodnotach na kazde strane optima
B_CURVE = np.linspace(-15.0, 9.0, 600)            # skutecny (v praxi neznamy) prubeh
TANGENT_HALF_WIDTH = 1.1                          # polovicni sirka tecny v jednotkach b


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoida s teplotou T."""
    return 1.0 / (1.0 + np.exp(-z / T))


def mse_for_bias(x: np.ndarray, y: np.ndarray, b: float) -> float:
    """MSE neuronu sigmoid(x @ (W, W) + b) vuci stitkum y."""
    return float(np.mean((y - sigmoid(x @ np.array([W, W]) + b)) ** 2))


def mse_gradient_for_bias(x: np.ndarray, y: np.ndarray, b: float) -> float:
    """dL/db = (2/N) * sum (y_hat - y) * y_hat * (1 - y_hat) / T -- bodovy vypocet gradientu."""
    y_hat = sigmoid(x @ np.array([W, W]) + b)
    return float(np.mean(2.0 * (y_hat - y) * y_hat * (1.0 - y_hat) / T))


def main() -> None:
    """Vykresli dvoupanelovy obrazek a ulozi ho do docs/img/loss_landscape.png."""
    x, y = make_blobs(n_samples=120, centers=[[0.0, 0.0], [1.0, 1.0]], cluster_std=0.3,
                      random_state=3)
    losses_curve = np.array([mse_for_bias(x, y, b) for b in B_CURVE])
    b_opt = round(float(B_CURVE[np.argmin(losses_curve)]), 2)
    b_values = sorted(B_OTHERS + (b_opt,))
    losses = [mse_for_bias(x, y, b) for b in b_values]
    slopes = [mse_gradient_for_bias(x, y, b) for b in b_values]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.0, 5.2),
                                     gridspec_kw={"width_ratios": [1, 1.2]})
    fig.patch.set_facecolor(SURFACE)
    for ax in (ax_l, ax_r):
        ax.set_facecolor(SURFACE)
        for spine in ax.spines.values():
            spine.set_color(HAIR)
        ax.tick_params(colors=INK2, labelsize=9)

    # --- levy panel: data a sedm rozhodovacich primek ------------------------
    lim = (-2.5, 3.5)
    for cls, color in ((0, BLUE), (1, ORANGE)):
        ax_l.scatter(x[y == cls, 0], x[y == cls, 1], s=26, c=color, edgecolors="white",
                     linewidths=0.5, label=f"třída {cls}", zorder=3)
    xs = np.array(lim)
    for b in b_values:
        is_opt = b == b_opt
        # primka W*x1 + W*x2 + b = 0  ->  x1 + x2 = c,  c = -b / W
        c = -b / W
        ax_l.plot(xs, c - xs, color=INK if is_opt else MUTED, lw=2.0 if is_opt else 1.1,
                  ls="-" if is_opt else "--", zorder=2)
        # popisek na primce, posunuty podel ni mimo shluky dat
        lx, ly = c / 2 + 1.15, c / 2 - 1.15
        ax_l.text(lx, ly, f"b = {b:g}", rotation=-45, rotation_mode="anchor", ha="center",
                  va="bottom", fontsize=8.5, color=INK if is_opt else INK2,
                  fontweight="bold" if is_opt else "normal",
                  bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 0.6}, zorder=4)
    ax_l.set_xlim(*lim)
    ax_l.set_ylim(*lim)
    ax_l.set_aspect("equal")
    ax_l.set_xlabel("$x_1$", color=INK2)
    ax_l.set_ylabel("$x_2$", color=INK2)
    ax_l.set_title(f"Neuron $\\sigma(W x_1 + W x_2 + b)$, $W = {W:g}$: sedm hodnot biasu",
                   color=INK, fontsize=10.5)
    ax_l.legend(frameon=False, fontsize=8.5, loc="upper left")

    # --- pravy panel: jen sedm vypoctenych bodu a jejich gradient -----------
    ax_r.plot(B_CURVE, losses_curve, color=HAIR, lw=1.2, ls=(0, (4, 3)), zorder=1,
              label="skutečný průběh $L(b)$ (v praxi neznámý)")
    for i, (b, loss, slope) in enumerate(zip(b_values, losses, slopes)):
        tb = np.array([b - TANGENT_HALF_WIDTH, b + TANGENT_HALF_WIDTH])
        ax_r.plot(tb, loss + slope * (tb - b), color=GRAD, lw=2.0, solid_capstyle="round",
                  zorder=2, label="sklon $\\partial L/\\partial b$ v bodě (gradient)" if i == 0 else None)
    ax_r.scatter(b_values, losses, s=42, color=INK, edgecolors=SURFACE, linewidths=1.5, zorder=3,
                 label="vypočtená ztráta")
    ax_r.scatter([b_opt], [mse_for_bias(x, y, b_opt)], s=110, facecolors="none", edgecolors=ORANGE,
                 linewidths=2.0, zorder=4, label="minimum (cíl učení)")

    ax_r.set_xticks(b_values, [f"{b:g}" for b in b_values])
    ax_r.set_xlim(B_CURVE[0], B_CURVE[-1])
    ax_r.set_ylim(-0.02, 0.66)
    ax_r.grid(True, axis="x", color=HAIR, lw=0.5, alpha=0.6)
    ax_r.grid(True, axis="y", color=HAIR, lw=0.5, alpha=0.4)
    ax_r.set_xlabel("bias $b$ (značky = vypočtené body)", color=INK2)
    ax_r.set_ylabel("ztráta $L(b)$ (MSE)", color=INK2)
    ax_r.set_title("Ztráta a její gradient ve vypočtených bodech", color=INK, fontsize=10.5)
    ax_r.legend(frameon=False, fontsize=8.5, loc="upper center", ncol=2)

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "loss_landscape.png", dpi=130, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    for b, loss, slope in zip(b_values, losses, slopes):
        print(f"b = {b:7.2f}   L = {loss:.4f}   dL/db = {slope:+.5f}")
    print("ulozeno", OUT / "loss_landscape.png")


if __name__ == "__main__":
    main()
