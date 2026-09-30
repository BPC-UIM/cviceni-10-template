"""Schema architektury site 2-2-1 s backpropagation pro README cviceni 10.

Forward (nahore, sede): data tecou dopredu, vystup jedne vrstvy je vstupem dalsi.
Backward (dole, cihlove): gradient chyby tece nazpet, v kazde vrstve nasoben derivaci aktivace.
Rozsireni forward-only diagramu z cv09 o zpetnou cestu.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "img"

SURFACE, INK, INK2, MUTED, HAIR = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
BLUE_T, VIOLET_T, ORANGE_T = "#d5e5f8", "#e6e4f5", "#fadccf"
GRAD = "#b5493a"  # cihlova pro zpetnou cestu (gradient)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})


def architektura() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.9))
    fig.patch.set_facecolor(SURFACE)
    ax.set_xlim(-0.6, 10.6)
    ax.set_ylim(-2.75, 3.35)
    ax.axis("off")

    cols = {0: (0.6, [2.2, 0.8], ["x₁", "x₂"], BLUE_T),
            1: (5.0, [2.2, 0.8], ["y₁", "y₂"], VIOLET_T),
            2: (9.4, [1.5], ["ŷ"], ORANGE_T)}
    pos = {(c, n): (x, y)
           for c, (x, ys, names, _) in cols.items()
           for y, n in zip(ys, names)}

    for c1, c2 in ((0, 1), (1, 2)):
        for n1 in cols[c1][2]:
            for n2 in cols[c2][2]:
                (xa, ya), (xb, yb) = pos[(c1, n1)], pos[(c2, n2)]
                ax.plot([xa + 0.33, xb - 0.33], [ya, yb], color=HAIR, lw=1.3, zorder=1)

    for c, (x, ys, names, fc) in cols.items():
        for y, n in zip(ys, names):
            ax.add_patch(Circle((x, y), 0.33, facecolor=fc, edgecolor=INK2, lw=1.2, zorder=2))
            ax.text(x, y, n, ha="center", va="center", fontsize=14, color=INK, zorder=3)

    for xc, text in ((2.8, "vrstva 1\nNeuron(Linear(W₁, b₁), Sigmoid)\nW₁: (2, 2)   b₁: (2,)"),
                     (7.2, "vrstva 2\nNeuron(Linear(W₂, b₂), Sigmoid)\nW₂: (2, 1)   b₂: (1,)")):
        ax.add_patch(FancyBboxPatch((xc - 1.45, 2.72), 2.9, 0.58, boxstyle="round,pad=0.06",
                                    facecolor="white", edgecolor=HAIR, lw=1.0))
        ax.text(xc, 3.01, text, ha="center", va="center", fontsize=8.6, color=INK2,
                linespacing=1.25)

    for x, lab in ((0.6, "vstupní\nprostor"), (5.0, "skrytý\nprostor"), (9.4, "výstup")):
        ax.text(x, 3.05, lab, ha="center", va="center", fontsize=9, color=INK2)

    # --- FORWARD stopa (nahore) ---
    for x, lab in ((0.6, "io_[0] = x\n(n, 2)"), (5.0, "io_[1]\n(n, 2)"), (9.4, "io_[2]\n(n, 1)")):
        ax.text(x, -0.35, lab, ha="center", va="top", fontsize=10, color=INK,
                linespacing=1.3, family="DejaVu Sans Mono")
    ax.add_patch(FancyArrowPatch((0.2, -1.2), (9.9, -1.2), arrowstyle="-|>",
                                 mutation_scale=10, color=MUTED, linewidth=1.0))
    ax.text(5.0, -1.12, "Sequential.forward: výstup jedné vrstvy je vstupem další",
            ha="center", va="bottom", fontsize=9, color=INK2)

    # --- BACKWARD stopa (dole) ---
    for x, lab in ((0.6, "∇W₁ = xᵀ·δ₁\n(2, 2)"), (5.0, "δ₁\n(n, 2)"),
                   (9.4, "∂L/∂ŷ → δ₂\n(n, 1)")):
        ax.text(x, -1.62, lab, ha="center", va="top", fontsize=10, color=GRAD,
                linespacing=1.3, family="DejaVu Sans Mono")
    ax.add_patch(FancyArrowPatch((9.9, -2.5), (0.2, -2.5), arrowstyle="-|>",
                                 mutation_scale=10, color=GRAD, linewidth=1.0))
    ax.text(5.0, -2.42, "Sequential.backward: gradient chyby teče nazpět (× derivace aktivace)",
            ha="center", va="bottom", fontsize=9, color=GRAD)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "architektura_2_2_1_backprop.png", dpi=130, bbox_inches="tight",
                facecolor=SURFACE)
    plt.close(fig)
    print("ulozeno", OUT / "architektura_2_2_1_backprop.png")


if __name__ == "__main__":
    architektura()
