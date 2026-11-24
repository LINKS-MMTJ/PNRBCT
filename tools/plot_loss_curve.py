
import os
import sys
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "CARETR_Project")

SURFACE = "#fcfcfb"
INK = "#2a2a28"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

C_TRAIN = "#2a78d6"   # 蓝
C_VAL = "#eb6834"     # 橙


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    src = args[0] if args else os.path.join(ROOT, "results", "stage1",
                                            "caret_s0_448_bal_train_metrics.csv")
    dst = args[1] if len(args) > 1 else os.path.join(ROOT, "figures",
                                                     "fig_loss_448_bal.png")
    best = None
    if "--best" in sys.argv:
        best = int(sys.argv[sys.argv.index("--best") + 1])

    rows = list(csv.DictReader(open(src, encoding="utf-8")))
    ep = [int(r["epoch"]) for r in rows]
    tr = [float(r["train_loss"]) for r in rows]
    va = [float(r["val_loss"]) for r in rows]

    plt.rcParams.update({
        "font.family": "Segoe UI", "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "font.size": 11,
    })

    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    ax.plot(ep, tr, color=C_TRAIN, lw=2, label="Training loss")
    ax.plot(ep, va, color=C_VAL, lw=2, label="Validation loss")

    if best is not None:
        ax.axvline(best, color=MUTED, lw=1, ls=":", alpha=0.8)
        ax.annotate(f"best epoch {best}", xy=(best, 0.02),
                    xycoords=("data", "axes fraction"), xytext=(5, 0),
                    textcoords="offset points", fontsize=9, color=MUTED,
                    ha="left", va="bottom")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_xlim(0, ep[-1])
    ax.set_ylim(0, 1)
    ax.set_yticks([i / 10 for i in range(11)])
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right", ncol=2)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    fig.savefig(dst, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    print(f"[OK] {dst}")
    print(f"     epoch={ep[-1]}  train_loss {tr[0]:.4f} -> {tr[-1]:.4f}  "
          f"val_loss {va[0]:.4f} -> {va[-1]:.4f}")


if __name__ == "__main__":
    main()
