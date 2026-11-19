"""Thesis figures for experiments/candidate_sets.py (conventions in docs/figures.md).

Two versions of each figure: 1-ring vs. 8 sectors (candidate_sets_*), and with 6 sectors added
(candidate_sets_*_with_6_sectors, drawn only when those runs exist).
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

from thesis_style import ACCENT, INK, KINDS, MUTED, TEXT_WIDTH, save, strip_legend, strip_plot, style

LOOKS = {"1-ring": dict(color=ACCENT, marker="o", label="Delaunay 1-ring", linestyle="-"),
         "6 sectors": dict(color=INK, marker="^", label="6 sector neighbours", linestyle=":"),
         "8 sectors": dict(color=MUTED, marker="s", label="8 sector neighbours", linestyle="--")}
VERSIONS = {"": ("1-ring", "8 sectors"), "_with_6_sectors": ("1-ring", "6 sectors", "8 sectors")}


def runtime(runs, variants, folder, suffix):
    fig, axis = plt.subplots(figsize=(TEXT_WIDTH, 0.42 * TEXT_WIDTH), layout="constrained")
    strip_plot(axis, runs, variants, "S", lambda run: run["total time"], "total time [s]")
    strip_legend(fig, variants)
    save(fig, folder, f"candidate_sets_runtime{suffix}")


def level0(runs, variants, folder, suffix):
    fig, (left, right) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 0.42 * TEXT_WIDTH), layout="constrained")
    strip_plot(left, runs, variants, "S", lambda run: len(run["levels"][-1]["history"]), "iterations at level 0")
    left.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    strip_plot(right, runs, variants, "S", lambda run: run["levels"][-1]["history"][-1]["N"] / run["cells per level"][0],
               r"$|\mathcal{N}| \,/\, |X|$ at level 0")
    strip_legend(fig, variants)
    save(fig, folder, f"candidate_sets_level0{suffix}")


def cone_constant(q, variants, folder, suffix):
    fig, axes = plt.subplots(1, len(KINDS), figsize=(TEXT_WIDTH, 0.36 * TEXT_WIDTH), layout="constrained",
                             sharey=True)
    for axis, kind in zip(axes, KINDS):
        for name, look in variants.items():
            values = np.sort(np.concatenate([q[key] for key in q.files if key.startswith(f"{kind}/") and key.endswith(name)]))
            axis.plot(values, np.arange(1, len(values) + 1) / len(values), color=look["color"],
                      linestyle=look["linestyle"], lw=1.1)
        axis.set_title(kind, fontsize=9)
        axis.set_xlim(-0.2, 1)
        axis.set_xlabel(r"cone constant $q$")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("share of points")
    handles = [plt.Line2D([], [], color=look["color"], linestyle=look["linestyle"], lw=1.1, label=look["label"])
               for look in variants.values()]
    fig.legend(handles=handles, loc="outside lower center", ncols=len(handles), frameon=False)
    save(fig, folder, f"candidate_sets_q{suffix}")


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "results" / "candidate_sets"
    runs = json.loads((folder / "runs.json").read_text(encoding="utf-8"))["runs"]
    q = np.load(folder / "q_level0.npz")
    style()
    for suffix, names in VERSIONS.items():
        if not all(any(run["S"] == name for run in runs) for name in names):
            continue
        variants = {name: LOOKS[name] for name in names}
        if len(names) > 2:  # shorter labels, so that the legend stays one line at text width
            variants = {name: dict(look, label=look["label"].replace(" sector neighbours", " sectors")) for name, look in variants.items()}
        runtime(runs, variants, folder, suffix)
        level0(runs, variants, folder, suffix)
        cone_constant(q, variants, folder, suffix)


if __name__ == "__main__":
    main()
