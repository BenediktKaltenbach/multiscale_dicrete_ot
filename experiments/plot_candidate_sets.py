"""Thesis figures for experiments/candidate_sets.py (conventions in docs/figures.md).

candidate_sets_runtime: total time per point kind, one marker per seed, mean as a bar.
candidate_sets_level0: solve_sparse iterations and |N| / |X| at level 0 (N the shielding neighbourhood of the
last round, the variables of the local problem), same layout.
candidate_sets_q: distribution of the cone constant q (5.6) at level 0, all seeds pooled, one panel per kind.

python experiments/plot_candidate_sets.py [results folder]
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from thesis_style import ACCENT, KINDS, MUTED, TEXT_WIDTH, save, strip_legend, strip_plot, style

VARIANTS = {"1-ring": dict(color=ACCENT, marker="o", label="Delaunay 1-ring", linestyle="-"),
            "8 sectors": dict(color=MUTED, marker="s", label="8 sector neighbours", linestyle="--")}


def runtime(runs, folder):
    fig, axis = plt.subplots(figsize=(TEXT_WIDTH, 0.42 * TEXT_WIDTH), layout="constrained")
    strip_plot(axis, runs, VARIANTS, "S", lambda run: run["total time"], "total time [s]")
    strip_legend(fig, VARIANTS)
    save(fig, folder, "candidate_sets_runtime")


def level0(runs, folder):
    fig, (left, right) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 0.42 * TEXT_WIDTH), layout="constrained")
    strip_plot(left, runs, VARIANTS, "S", lambda run: len(run["levels"][-1]["history"]), "iterations at level 0")
    left.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    strip_plot(right, runs, VARIANTS, "S", lambda run: run["levels"][-1]["history"][-1]["N"] / run["cells per level"][0],
               r"$|N| \,/\, |X|$ at level 0")
    strip_legend(fig, VARIANTS)
    save(fig, folder, "candidate_sets_level0")


def cone_constant(q, folder):
    fig, axes = plt.subplots(1, len(KINDS), figsize=(TEXT_WIDTH, 0.36 * TEXT_WIDTH), layout="constrained",
                             sharey=True)
    for axis, kind in zip(axes, KINDS):
        for name, look in VARIANTS.items():
            values = np.sort(np.concatenate([q[key] for key in q.files if key.startswith(f"{kind}/") and key.endswith(name)]))
            axis.plot(values, np.arange(1, len(values) + 1) / len(values), color=look["color"],
                      linestyle=look["linestyle"], lw=1.1)
        axis.set_title(kind, fontsize=9)
        axis.set_xlim(-0.2, 1)
        axis.set_xlabel(r"cone constant $q$")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("share of points")
    handles = [plt.Line2D([], [], color=look["color"], linestyle=look["linestyle"], lw=1.1, label=look["label"])
               for look in VARIANTS.values()]
    fig.legend(handles=handles, loc="outside lower center", ncols=2, frameon=False)
    save(fig, folder, "candidate_sets_q")


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "results" / "candidate_sets"
    runs = json.loads((folder / "runs.json").read_text(encoding="utf-8"))["runs"]
    style()
    runtime(runs, folder)
    level0(runs, folder)
    cone_constant(np.load(folder / "q_level0.npz"), folder)


if __name__ == "__main__":
    main()
