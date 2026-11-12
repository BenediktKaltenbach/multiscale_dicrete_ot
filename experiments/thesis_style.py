"""Shared look of the thesis result figures (conventions in docs/figures.md)."""
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TEXT_WIDTH = 345 / 72.27  # \textwidth of the article class at 10pt, in inches: figures are used at 100%
LATEX = shutil.which("pdflatex") is not None
ACCENT, MUTED, INK, FAINT = "#2b5d8c", "#6b7380", "#1d232b", "#c9ced6"
KINDS = ("uniform", "clustered", "grid")


def style():
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "legend.fontsize": 8, "axes.linewidth": 0.5,
                         "axes.edgecolor": MUTED, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                         "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": "black",
                         "ytick.labelcolor": "black", "axes.spines.top": False, "axes.spines.right": False})
    if LATEX:
        plt.rcParams.update({"pgf.texsystem": "pdflatex", "pgf.rcfonts": False,
                             "pgf.preamble": r"\usepackage{amsmath}\usepackage{amssymb}"})
    else:
        plt.rcParams.update({"mathtext.fontset": "cm", "font.serif": ["cmr10"], "axes.formatter.use_mathtext": True})


def save(fig, folder, name):
    """PDF for the thesis, PNG at 600 ppi for previews."""
    for suffix in ("pdf", "png"):
        fig.savefig(folder / f"{name}.{suffix}", bbox_inches="tight", pad_inches=0.02, dpi=600,
                    backend="pgf" if LATEX else None)
    plt.close(fig)


def strip_plot(axis, runs, variants, field, value, label):
    """Per point kind: one marker per seed for each variant (run[field] == key of variants), the mean as a short bar."""
    offsets = np.linspace(-0.17, 0.17, len(variants)) if len(variants) > 1 else [0.0]
    for position, kind in enumerate(KINDS):
        for offset, (name, look) in zip(offsets, variants.items()):
            values = np.array([value(run) for run in runs if run["kind"] == kind and run[field] == name])
            spread = np.linspace(-0.07, 0.07, len(values))
            axis.scatter(position + offset + spread, values, s=11, marker=look["marker"], lw=0.9,
                         facecolor=look["color"] if look["marker"] == "o" else "white", edgecolor=look["color"],
                         zorder=3)
            axis.plot([position + offset - 0.13, position + offset + 0.13], [values.mean()] * 2, color=look["color"],
                      lw=1.2, zorder=2)
    axis.set_xticks(range(len(KINDS)), KINDS)
    axis.tick_params(axis="x", length=0)
    axis.set_xlim(-0.6, len(KINDS) - 0.4)
    axis.set_ylabel(label)
    axis.set_ylim(0, 1.1 * max(value(run) for run in runs))


def strip_legend(fig, variants, with_mean=True):
    handles = [plt.Line2D([], [], color=look["color"], marker=look["marker"], lw=0, markersize=4,
                          markerfacecolor=look["color"] if look["marker"] == "o" else "white", label=look["label"])
               for look in variants.values()]
    if with_mean:
        handles.append(plt.Line2D([], [], color=MUTED, lw=1.2, label="mean of 3 seeds"))
    fig.legend(handles=handles, loc="outside lower center", ncols=len(handles), frameon=False, handletextpad=0.4,
               columnspacing=1.4)
