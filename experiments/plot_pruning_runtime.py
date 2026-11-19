"""Thesis figures for experiments/pruning.py (conventions in docs/figures.md).

pruning_time: total and shield time per point kind, pruning off vs. on, one marker per seed, mean as a bar.
pruning_ratio: shield time with / without pruning per seed and point kind.
pruning_dropped: share of the candidates that pruning drops, per level, mean over the seeds.

python experiments/plot_pruning_runtime.py [results folder]
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from thesis_style import BLUE, KIND_COLOURS, KINDS, LATEX, MUTED, ORANGE, TEXT_WIDTH, save, strip_legend, strip_plot, style

VARIANTS = {False: dict(color=BLUE, marker="o", label="pruning off"),
            True: dict(color=ORANGE, marker="s", label="pruning on")}
KIND_LOOKS = {kind: dict(color=KIND_COLOURS[kind], marker=marker, linestyle="-")
              for kind, marker in (("grid", "s"), ("uniform", "o"), ("clustered", "^"))}


def times(runs, folder):
    fig, (left, right) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 0.42 * TEXT_WIDTH), layout="constrained")
    strip_plot(left, runs, VARIANTS, "prune", lambda run: run["total time"], "total time [s]")
    strip_plot(right, runs, VARIANTS, "prune", lambda run: run["shield time"], "shield time [s]")
    strip_legend(fig, VARIANTS)
    save(fig, folder, "pruning_time")


def ratio(runs, folder):
    pairs = {}
    for run in runs:
        pairs.setdefault((run["kind"], run["seed"]), {})[run["prune"]] = run["shield time"]
    ratios = [{"kind": kind, "seed": seed, "variant": "ratio", "ratio": shield[True] / shield[False]}
              for (kind, seed), shield in pairs.items()]
    look = {"ratio": dict(color=BLUE, marker="o", label="one seed")}
    fig, axis = plt.subplots(figsize=(TEXT_WIDTH, 0.36 * TEXT_WIDTH), layout="constrained")
    axis.axhline(1, color=MUTED, lw=0.5, zorder=1)
    strip_plot(axis, ratios, look, "variant", lambda run: run["ratio"], "shield time on / off")
    values = [run["ratio"] for run in ratios]
    margin = 0.25 * (max(values) - min(values)) + 0.02
    axis.set_ylim(min(min(values), 1) - margin, max(max(values), 1) + margin)
    strip_legend(fig, look)
    save(fig, folder, "pruning_ratio")


def dropped(runs, folder):
    fig, axis = plt.subplots(figsize=(TEXT_WIDTH, 0.4 * TEXT_WIDTH), layout="constrained")
    highest = 0
    for kind in KINDS:
        share_per_level = {}
        for run in runs:
            if run["kind"] != kind or not run["prune"]:
                continue
            level_of = {cells: k for k, cells in enumerate(run["cells per level"])}
            totals = {}
            for call in run["pruning calls"]:
                total = totals.setdefault(level_of[call["points"]], [0, 0])
                total[0] += call["candidates"] - call["kept"]
                total[1] += call["candidates"]
            for k, (drop, candidates) in totals.items():
                share_per_level.setdefault(k, []).append(drop / candidates)
        levels = sorted(share_per_level, reverse=True)
        look = KIND_LOOKS[kind]
        shares = [100 * np.mean(share_per_level[k]) for k in levels]
        highest = max(highest, max(shares))
        axis.set_xticks(levels)
        axis.plot(levels, shares, color=look["color"],
                  linestyle=look["linestyle"], marker=look["marker"], markersize=3.5, lw=1.1,
                  markerfacecolor=look["color"] if look["marker"] == "o" else "white", label=kind)
    axis.invert_xaxis()
    axis.set_xlabel(r"level $k$ (0 = finest)")
    axis.set_ylabel(r"dropped [\%]" if LATEX else "dropped [%]")
    axis.set_ylim(0, 5 * np.ceil(1.1 * highest / 5))
    axis.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
    fig.legend(loc="outside lower center", ncols=len(KINDS), frameon=False)
    save(fig, folder, "pruning_dropped")


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "results" / "pruning"
    runs = json.loads((folder / "runs.json").read_text(encoding="utf-8"))["runs"]
    style()
    times(runs, folder)
    ratio(runs, folder)
    dropped(runs, folder)


if __name__ == "__main__":
    main()
