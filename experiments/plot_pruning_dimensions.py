"""Thesis figures for experiments/pruning_dimensions.py (conventions in docs/figures.md).

pruning_dimensions_time: total time per point kind, one panel per dimension, without pruning, exact and closest.
pruning_dimensions_ratio: shield time with / without pruning per seed, one panel per dimension.
pruning_dimensions_dropped: share of the 1-ring that the exact face test drops (no facet), per level.

python experiments/plot_pruning_dimensions.py [results folder]
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from thesis_style import BLUE, GREEN, KIND_COLOURS, LATEX, MUTED, ORANGE, TEXT_WIDTH, save, strip_legend, strip_plot, style

KINDS = ("uniform", "clustered")
VARIANTS = {"off": dict(color=BLUE, marker="o", label="no pruning"),
            "exact": dict(color=ORANGE, marker="s", label="exact facets"),
            "closest": dict(color=GREEN, marker="^", label="closest 60 %")}
LINES = {(dimension, kind): dict(color=KIND_COLOURS[kind], linestyle=linestyle, marker=marker)
         for dimension, linestyle in ((4, "--"), (5, "-")) for kind, marker in (("uniform", "o"), ("clustered", "^"))}


def dimensions(runs):
    return sorted({run["dimension"] for run in runs})


def times(runs, folder):
    fig, axes = plt.subplots(1, len(dimensions(runs)), figsize=(TEXT_WIDTH, 0.42 * TEXT_WIDTH), layout="constrained")
    for axis, dimension in zip(np.atleast_1d(axes), dimensions(runs)):
        strip_plot(axis, [run for run in runs if run["dimension"] == dimension], VARIANTS, "pruning",
                   lambda run: run["total time"], "total time [s]", KINDS)
        axis.set_title(f"$d = {dimension}$", fontsize=9)
    strip_legend(fig, VARIANTS)
    save(fig, folder, "pruning_dimensions_time")


def ratio(runs, folder):
    shield = {(run["dimension"], run["kind"], run["seed"], run["pruning"]): run["shield time"] for run in runs}
    ratios = [{"dimension": d, "kind": kind, "seed": seed, "pruning": name, "ratio": value / shield[(d, kind, seed, "off")]}
              for (d, kind, seed, name), value in shield.items() if name != "off"]
    looks = {name: VARIANTS[name] for name in ("exact", "closest")}
    fig, axes = plt.subplots(1, len(dimensions(runs)), figsize=(TEXT_WIDTH, 0.4 * TEXT_WIDTH), layout="constrained",
                             sharey=True)
    for axis, dimension in zip(np.atleast_1d(axes), dimensions(runs)):
        axis.axhline(1, color=MUTED, lw=0.5, zorder=1)
        strip_plot(axis, [run for run in ratios if run["dimension"] == dimension], looks, "pruning",
                   lambda run: run["ratio"], "shield time / no pruning", KINDS)
        axis.set_title(f"$d = {dimension}$", fontsize=9)
    for axis in np.atleast_1d(axes)[1:]:
        axis.set_ylabel("")
    values = [run["ratio"] for run in ratios]
    np.atleast_1d(axes)[0].set_ylim(0, 1.1 * max(max(values), 1))
    strip_legend(fig, looks)
    save(fig, folder, "pruning_dimensions_ratio")


def dropped(runs, folder):
    fig, axis = plt.subplots(figsize=(TEXT_WIDTH, 0.45 * TEXT_WIDTH), layout="constrained")
    highest = 0
    for (dimension, kind), look in LINES.items():
        share_per_level = {}
        for run in runs:
            if (run["dimension"], run["kind"], run["pruning"]) != (dimension, kind, "exact"):
                continue
            level_of = {cells: k for k, cells in enumerate(run["cells per level"])}
            totals = {}
            for call in run["pruning calls"]:
                total = totals.setdefault(level_of[call["points"]], [0, 0])
                total[0] += call["candidates"] - call["kept"]
                total[1] += call["candidates"]
            for k, (drop, candidates) in totals.items():
                share_per_level.setdefault(k, []).append(drop / candidates)
        if not share_per_level:
            continue
        levels = sorted(share_per_level, reverse=True)
        shares = [100 * np.mean(share_per_level[k]) for k in levels]
        highest = max(highest, max(shares))
        axis.plot(levels, shares, color=look["color"], linestyle=look["linestyle"], marker=look["marker"],
                  markersize=3.5, lw=1.1, markerfacecolor=look["color"] if look["marker"] == "o" else "white",
                  label=f"$d = {dimension}$, {kind}")
        axis.set_xticks(levels)
    axis.invert_xaxis()
    axis.set_xlabel(r"level $k$ (0 = finest)")
    axis.set_ylabel(r"no facet [\%]" if LATEX else "no facet [%]")
    axis.set_ylim(0, 10 * np.ceil(1.1 * highest / 10))
    axis.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))
    fig.legend(loc="outside lower center", ncols=4, frameon=False, handlelength=2.2, columnspacing=1.2)
    save(fig, folder, "pruning_dimensions_dropped")


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "results" / "pruning_dimensions"
    runs = json.loads((folder / "runs.json").read_text(encoding="utf-8"))["runs"]
    style()
    times(runs, folder)
    ratio(runs, folder)
    dropped(runs, folder)


if __name__ == "__main__":
    main()
