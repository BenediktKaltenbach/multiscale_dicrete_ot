"""Thesis figures for experiments/scaling.py (conventions in docs/figures.md).

scaling_time: total time against n (log-log), one panel per kind, one marker per run, mean per n as a thin line;
multiscale and the dense references (POT, CPLEX on all of X x Y) on the same problems.
scaling_time_guides: the same with guide lines of slope 1 and 2.
scaling_sparsity: |N| / |X| and rounds of Alg. 4.1 at level 0 against n.
scaling_breakdown: shares of solve, shield, hierarchy and other time in the total, mean over seeds, one panel per kind.
Prints the fitted exponents b of time = a n^b.

python experiments/plot_scaling.py [results folder]
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.text import Text

from thesis_style import ACCENT, FAINT, INK, KINDS, MUTED, TEXT_WIDTH, save, style

KIND_LOOKS = {"grid": dict(marker="s", linestyle="--"), "uniform": dict(marker="o", linestyle="-"),
              "clustered": dict(marker="^", linestyle=":")}
DENSE_LOOKS = {"pot": dict(color=MUTED, marker="D", label="POT (dense)"),
               "cplex dense": dict(color=INK, marker="x", label="CPLEX (dense)")}
SHARES = {"solve time": (ACCENT, "local solver"), "shield time": (MUTED, "shield"),
          "hierarchy time": (INK, "hierarchy"), "other time": (FAINT, "other")}
SIZES_SHOWN = [2**10, 2**13, 2**16]


def series(runs, value):
    """Sizes and the mean of value over the runs of each size."""
    sizes = sorted({run["n"] for run in runs})
    return np.array(sizes), np.array([np.mean([value(run) for run in runs if run["n"] == n]) for n in sizes])


def exponent(sizes, times):
    return np.polyfit(np.log(sizes), np.log(times), 1)[0]


def scatter(axis, sizes, values, look, color):
    axis.scatter(sizes, values, s=9, marker=look["marker"], lw=0.8, color=color,
                 facecolor="white" if look["marker"] in "s^D" else color, zorder=3)


class LineLabel(Text):
    """Text parallel to a line given by two points in data coordinates; the angle is taken when drawn."""

    def __init__(self, line, *arguments, **keywords):
        super().__init__(*arguments, **keywords)
        self.line = line

    def draw(self, renderer):
        (x0, y0), (x1, y1) = self.axes.transData.transform(self.line)
        self.set_rotation(np.degrees(np.arctan2(y1 - y0, x1 - x0)))
        super().draw(renderer)


def time_figure(runs, folder, guides):
    fig, axes = plt.subplots(1, len(KINDS), figsize=(TEXT_WIDTH, 0.42 * TEXT_WIDTH), layout="constrained",
                             sharex=True, sharey=True)
    multiscale = dict(color=ACCENT, marker="o", label="multiscale")
    for axis, kind in zip(axes, KINDS):
        own = [run for run in runs if run["kind"] == kind]
        for name, look in (("total time", multiscale), *DENSE_LOOKS.items()):
            time = (lambda run: run["total time"]) if name == "total time" else (lambda run: run[name]["time"])
            measured = [run for run in own if name in run]
            scatter(axis, [run["n"] for run in measured], [time(run) for run in measured], look, look["color"])
            axis.plot(*series(measured, time), color=look["color"], lw=0.8)
        if guides:
            sizes = np.array([min(run["n"] for run in own), max(run["n"] for run in own)])
            anchor = np.mean([run["total time"] for run in own if run["n"] == sizes[0]])
            for slope, linestyle in ((1, (0, (1, 2))), (2, (0, (4, 3)))):
                line = np.column_stack([sizes, anchor * (sizes / sizes[0]) ** slope])
                axis.plot(*line.T, color=FAINT, lw=0.8, linestyle=linestyle, zorder=1)
                if axis is axes[0]:
                    x = sizes[1] / 2**0.6
                    y = anchor * (x / sizes[0]) ** slope
                    axis.add_artist(LineLabel(line, x, y * (1.4 if slope == 2 else 1 / 1.4), f"slope {slope}",
                                              color=MUTED, fontsize=8, ha="center", clip_on=False,
                                              va="bottom" if slope == 2 else "top", rotation_mode="anchor"))
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xticks(SIZES_SHOWN)
        axis.minorticks_off()
        axis.set_title(kind, fontsize=9)
        axis.set_xlabel(r"$n$")
    axes[0].set_ylabel("time [s]")
    handles = [plt.Line2D([], [], color=look["color"], lw=0.8, marker=look["marker"], markersize=3.5,
                          markerfacecolor="white" if look["marker"] == "D" else look["color"], label=look["label"])
               for look in (multiscale, *DENSE_LOOKS.values())]
    fig.legend(handles=handles, loc="outside lower center", ncols=len(handles), frameon=False)
    save(fig, folder, "scaling_time_guides" if guides else "scaling_time")


def sparsity_figure(runs, folder):
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 0.42 * TEXT_WIDTH), layout="constrained")
    values = ((lambda run: run["per level"][-1]["last N per cell"], r"$|N| \,/\, |X|$ at level 0"),
              (lambda run: run["per level"][-1]["rounds"], "rounds at level 0"))
    for axis, (value, label) in zip(axes, values):
        for kind in KINDS:
            own = [run for run in runs if run["kind"] == kind]
            look = KIND_LOOKS[kind]
            scatter(axis, [run["n"] for run in own], [value(run) for run in own], look, ACCENT)
            axis.plot(*series(own, value), color=ACCENT, lw=0.8, linestyle=look["linestyle"])
        axis.set_xscale("log", base=2)
        axis.set_xticks(sorted({run["n"] for run in runs})[::2])
        axis.minorticks_off()
        axis.set_ylim(0, 1.1 * max(value(run) for run in runs))
        axis.set_xlabel(r"$n$")
        axis.set_ylabel(label)
    handles = [plt.Line2D([], [], color=ACCENT, lw=0.8, linestyle=look["linestyle"], marker=look["marker"],
                          markersize=3.5, markerfacecolor="white" if look["marker"] in "s^" else ACCENT, label=kind)
               for kind, look in KIND_LOOKS.items()]
    fig.legend(handles=handles, loc="outside lower center", ncols=len(handles), frameon=False)
    save(fig, folder, "scaling_sparsity")


def breakdown_figure(runs, folder):
    fig, axes = plt.subplots(1, len(KINDS), figsize=(TEXT_WIDTH, 0.38 * TEXT_WIDTH), layout="constrained",
                             sharey=True)
    for axis, kind in zip(axes, KINDS):
        own = [run for run in runs if run["kind"] == kind]
        bottom = None
        for field, (color, _) in SHARES.items():
            sizes, share = series(own, lambda run: run[field] / (run["total time"] + run["hierarchy time"]))
            bottom = np.zeros_like(share) if bottom is None else bottom
            axis.fill_between(sizes, bottom, bottom + share, color=color, lw=0)
            bottom = bottom + share
        axis.set_xscale("log", base=2)
        axis.set_xlim(sizes[0], sizes[-1])
        axis.set_xticks(sizes[::3])
        axis.minorticks_off()
        axis.set_title(kind, fontsize=9)
        axis.set_xlabel(r"$n$")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("share of time")
    handles = [plt.Rectangle((0, 0), 1, 1, color=color, label=label) for color, label in SHARES.values()]
    fig.legend(handles=handles, loc="outside lower center", ncols=len(handles), frameon=False)
    save(fig, folder, "scaling_breakdown")


def report(runs):
    for kind in KINDS:
        own = [run for run in runs if run["kind"] == kind]
        line = f"{kind:9s} b: multiscale {exponent(*series(own, lambda run: run['total time'])):.2f}"
        for name in DENSE_LOOKS:
            measured = [run for run in own if name in run]
            if len({run["n"] for run in measured}) > 1:
                line += f", {name} {exponent(*series(measured, lambda run: run[name]['time'])):.2f}"
        print(line)
    worst = max(run["dual violation"] for run in runs)
    differences = [run[name]["relative cost difference"] for run in runs for name in DENSE_LOOKS if name in run]
    print(f"max dual violation {worst:.1e}, max relative cost difference {max(differences, default=0):.1e}")


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "results" / "scaling"
    runs = json.loads((folder / "runs.json").read_text(encoding="utf-8"))["runs"]
    style()
    for guides in (False, True):
        time_figure(runs, folder, guides)
    sparsity_figure(runs, folder)
    breakdown_figure(runs, folder)
    report(runs)


if __name__ == "__main__":
    main()
