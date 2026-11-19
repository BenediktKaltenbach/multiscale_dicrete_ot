"""Phase 1 of docs/analysis_rounds.md: how close to optimal pi is after each round of Alg. 4.1, from the stored runs.

Reads the scaling runs (no solving). Per run and level: rounds R, relative gap g_r = (C_r - C_R) / C_R per round,
r(eps) = first round with g_r <= eps, the time of the rounds after r(eps) (solve and shield of rounds r(eps)+1 .. R),
and |N_r| (first, maximum, last). Writes results/rounds/rounds.json and rounds_table.txt, and the figure rounds_gap.

python experiments/rounds_analysis.py [scaling results folder]
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from thesis_style import FAINT, KIND_COLOURS, KINDS, MUTED, TEXT_WIDTH, save, style

EPSILONS = (1e-3, 1e-6, 1e-9)
RESULTS = Path(__file__).parent / "results" / "rounds"


def level_rounds(run, level):
    history = level["history"]
    costs = np.array([step["C"] for step in history])
    gaps = (costs - costs[-1]) / costs[-1]
    times = np.array([step["time_solve"] + step["time_shield"] for step in history])
    record = {"k": level["k"], "R": len(history), "gaps": gaps.tolist(), "level time": times.sum(),
              "N first": history[0]["N"] / level["cells"][0], "N max": max(step["N"] for step in history) / level["cells"][0],
              "N last": history[-1]["N"] / level["cells"][0]}
    for epsilon in EPSILONS:
        reached = int(np.argmax(gaps <= epsilon)) + 1
        record[f"r({epsilon:.0e})"] = reached
        record[f"tail time({epsilon:.0e})"] = times[reached:].sum()
        record[f"tail share of level({epsilon:.0e})"] = times[reached:].sum() / times.sum()
        record[f"tail share of total({epsilon:.0e})"] = times[reached:].sum() / run["total time"]
    return record


def analyse(runs):
    return [{"n": run["n"], "kind": run["kind"], "seed": run["seed"], "total time": run["total time"],
             "levels": [level_rounds(run, level) for level in run["levels"]]} for run in runs]


def summary(values):
    return f"{np.mean(values):6.1f} ({min(values):g}-{max(values):g})" if len(values) > 1 else f"{values[0]:6.1f}"


def table(records):
    lines = ["level 0, mean (range) over seeds; tail = rounds after r(1e-6), share of the level-0 time and of the total",
             f"{'kind':9s} {'n':>6s}  {'R':>14s}  {'r(1e-6)':>14s}  {'r(1e-9)':>14s}  {'g_1':>8s}  "
             f"{'tail level':>10s}  {'tail total':>10s}  {'N first/max/last per point':>26s}"]
    for kind in KINDS:
        for n in sorted({record["n"] for record in records if record["kind"] == kind}):
            levels = [record["levels"][-1] for record in records if (record["kind"], record["n"]) == (kind, n)]
            lines.append(f"{kind:9s} {n:6d}  {summary([level['R'] for level in levels]):>14s}  "
                         f"{summary([level['r(1e-06)'] for level in levels]):>14s}  "
                         f"{summary([level['r(1e-09)'] for level in levels]):>14s}  "
                         f"{np.mean([level['gaps'][0] for level in levels]):8.1e}  "
                         f"{np.mean([level['tail share of level(1e-06)'] for level in levels]):10.2f}  "
                         f"{np.mean([level['tail share of total(1e-06)'] for level in levels]):10.2f}  "
                         f"{np.mean([level['N first'] for level in levels]):8.1f} "
                         f"{np.mean([level['N max'] for level in levels]):8.1f} "
                         f"{np.mean([level['N last'] for level in levels]):8.1f}")
    certifying = sum(record["levels"][-1]["gaps"][-2] == 0 for record in records if record["levels"][-1]["R"] > 1)
    lines += ["", f"g = 0 already in round R - 1 (the last round only certifies): {certifying} of {len(records)} runs"]
    lines += ["", "per run, level 0: R, r(1e-3), r(1e-6), r(1e-9), g_1, tail share of total after r(1e-6); rounds of the coarser levels"]
    for record in sorted(records, key=lambda record: (KINDS.index(record["kind"]), record["n"], record["seed"])):
        level = record["levels"][-1]
        coarser = ", ".join(f"k={other['k']}: {other['R']}" for other in record["levels"][:-1])
        lines.append(f"{record['kind']:9s} {record['n']:6d} seed {record['seed']}: R {level['R']:2d}, "
                     f"r {level['r(1e-03)']:2d} {level['r(1e-06)']:2d} {level['r(1e-09)']:2d}, g_1 {level['gaps'][0]:.1e}, "
                     f"tail {level['tail share of total(1e-06)']:.2f}; {coarser}")
    return "\n".join(lines)


def gap_figure(records, folder):
    sizes = (4096, 65536)
    fig, axes = plt.subplots(len(sizes), len(KINDS), figsize=(TEXT_WIDTH, 0.62 * TEXT_WIDTH), layout="constrained",
                             sharex=True, sharey=True)
    for row, n in zip(axes, sizes):
        for axis, kind in zip(row, KINDS):
            for epsilon in EPSILONS[1:]:
                axis.axhline(epsilon, color=FAINT, lw=0.6, zorder=1)
            for record in records:
                if (record["kind"], record["n"]) != (kind, n):
                    continue
                gaps = np.array(record["levels"][-1]["gaps"])
                rounds = np.flatnonzero(gaps > 0)  # gap 0: the optimum is reached (usually in round R - 1)
                axis.plot(rounds + 1, gaps[rounds], color=KIND_COLOURS[kind], lw=0.9, marker="o", markersize=2,
                          zorder=2)
            axis.set_yscale("log")
            axis.set_ylim(1e-16, 1e-1)
            axis.set_title(f"{kind}, $n = {n}$", fontsize=9)
    for axis in axes[-1]:
        axis.set_xlabel("round $r$ at level 0")
    for axis in axes[:, 0]:
        axis.set_ylabel("gap $g_r$")
    for epsilon in EPSILONS[1:]:
        axes[0, -1].annotate(f"$10^{{{int(np.log10(epsilon))}}}$", (1, epsilon), xycoords=("axes fraction", "data"),
                             xytext=(2, 0), textcoords="offset points", va="center", color=MUTED, fontsize=8)
    save(fig, folder, "rounds_gap")


def main():
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "results" / "scaling"
    runs = json.loads((source / "runs.json").read_text(encoding="utf-8"))["runs"]
    records = analyse(runs)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "rounds.json").write_text(json.dumps(records, indent=1), encoding="utf-8")
    text = table(records)
    (RESULTS / "rounds_table.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    style()
    gap_figure(records, RESULTS)


if __name__ == "__main__":
    main()
