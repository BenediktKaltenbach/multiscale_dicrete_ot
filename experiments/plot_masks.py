"""Thesis figures for experiments/masks.py (conventions in docs/figures.md).

masks_problem: the masked masses mu and nu of the grid problem with seed 0 (log scale), showing the masks.
masks_time: total time per point kind, unmasked vs. masked, one marker per seed, mean as a bar.
masks_level0: solve_sparse iterations and |N| / |X| at level 0, same layout.

python experiments/plot_masks.py [results folder]
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from masks import N, problems
from thesis_style import ACCENT, LATEX, MUTED, TEXT_WIDTH, save, strip_legend, strip_plot, style

VARIANTS = {"unmasked": dict(color=MUTED, marker="s", label="without masks"),
            "masked": dict(color=ACCENT, marker="o", label="with masks")}


def problem(folder):
    X, mu, Y, nu = problems("grid", 0)["masked"]
    side = int(round(np.sqrt(N)))
    images = [np.log10(mass / mass.mean()).reshape(side, side) for mass in (mu, nu)]
    low, high = min(image.min() for image in images), max(image.max() for image in images)
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 0.5 * TEXT_WIDTH), layout="constrained")
    for axis, image, name in zip(axes, images, (r"$\mu$", r"$\nu$")):
        shown = axis.imshow(image, origin="lower", extent=(0, 1, 0, 1), cmap="Blues", vmin=low, vmax=high,
                            interpolation="nearest")
        axis.set_title(name, fontsize=9)
        axis.set_xticks([0, 1])
        axis.set_yticks([0, 1])
    colour_bar = fig.colorbar(shown, ax=axes, shrink=0.85, pad=0.02)
    colour_bar.set_label(r"$\log_{10}$ (mass / mean mass)")
    colour_bar.outline.set_linewidth(0.5)
    save(fig, folder, "masks_problem")


def times(runs, folder):
    fig, axis = plt.subplots(figsize=(TEXT_WIDTH, 0.42 * TEXT_WIDTH), layout="constrained")
    strip_plot(axis, runs, VARIANTS, "problem", lambda run: run["total time"], "total time [s]")
    strip_legend(fig, VARIANTS)
    save(fig, folder, "masks_time")


def level0(runs, folder):
    fig, (left, right) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 0.42 * TEXT_WIDTH), layout="constrained")
    strip_plot(left, runs, VARIANTS, "problem", lambda run: len(run["levels"][-1]["history"]), "iterations at level 0")
    left.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    strip_plot(right, runs, VARIANTS, "problem",
               lambda run: run["levels"][-1]["history"][-1]["N"] / run["cells per level"][0], r"$|N| \,/\, |X|$ at level 0")
    strip_legend(fig, VARIANTS)
    save(fig, folder, "masks_level0")


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "results" / "masks"
    runs = json.loads((folder / "runs.json").read_text(encoding="utf-8"))["runs"]
    style()
    problem(folder)
    times(runs, folder)
    level0(runs, folder)


if __name__ == "__main__":
    main()
