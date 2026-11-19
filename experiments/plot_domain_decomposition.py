"""Figure dd_sweeps for docs/domain_decomposition.md (Stage A): relative gap (C - C*) / C* to the optimum on N1
(top) and the entries of N1 that violate the certificate (bottom) after each sweep (sweep 0: the north-west start),
one line per level l of the basic cells, one column per problem. Reads results/domain_decomposition/runs.json.

python experiments/plot_domain_decomposition.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from thesis_style import BLUE, GREEN, KINDS, ORANGE, TEXT_WIDTH, save, style

RESULTS = Path(__file__).parent / "results" / "domain_decomposition"
LEVEL_LOOKS = {"1": dict(color=BLUE, marker="o"), "2": dict(color=ORANGE, marker="s", mfc="white"),
               "3": dict(color=GREEN, marker="^", mfc="white")}
ROWS = (("gap", r"gap to $C^*$"), ("violated entries", "violated entries"))


def main():
    runs = json.loads((RESULTS / "runs.json").read_text(encoding="utf-8"))["runs"]
    runs = sorted(runs.values(), key=lambda run: KINDS.index(run["kind"]))
    style()
    fig, axes = plt.subplots(len(ROWS), len(runs), figsize=(TEXT_WIDTH, 0.6 * TEXT_WIDTH), layout="constrained",
                             sharex=True, sharey="row", squeeze=False)
    for column, run in enumerate(runs):
        for row, (field, label) in enumerate(ROWS):
            axis = axes[row, column]
            for l, level in run["levels"].items():
                values = np.array([sweep[field] for sweep in level["sweeps"]], dtype=float)
                sweeps = np.arange(1, len(values) + 1)
                if field == "gap":  # sweep 0: the north-west start
                    values, sweeps = np.r_[run["start gap"], values], np.r_[0, sweeps]
                shown = values > 0
                axis.plot(sweeps[shown], values[shown], lw=1.0, markersize=2.5, markeredgewidth=0.6,
                          label=f"$l = {l}$", **LEVEL_LOOKS[l])
            axis.set_yscale("log")
            if field == "gap":
                ticks = [1e-4, 3e-4, 1e-3, 3e-3]
                axis.set_yticks(ticks, [r"$10^{-4}$", r"$3 \cdot 10^{-4}$", r"$10^{-3}$", r"$3 \cdot 10^{-3}$"])
                axis.minorticks_off()
            if row == 0:
                axis.set_title(f"{run['kind']}, $n = {run['n']}$", fontsize=9)
            if column == 0:
                axis.set_ylabel(label)
            if row == len(ROWS) - 1:
                axis.set_xlabel("sweep")
    fig.align_ylabels(axes[:, 0])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels), frameon=False)
    save(fig, RESULTS, "dd_sweeps")


if __name__ == "__main__":
    main()
