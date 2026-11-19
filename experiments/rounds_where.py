"""Figure rounds_where for docs/analysis_rounds.md (Phase 2): the points of X coloured by the last round at level 0
in which t(x) changed (top row) and in which x had a global dual violation (bottom row), one column per problem.
Points without either stay light grey; later rounds are drawn larger, so isolated late points stay visible. Reads results/rounds/instrumented_*.npz from rounds_instrumented.py.

python experiments/rounds_where.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from thesis_style import FAINT, TEXT_WIDTH, save, style

RESULTS = Path(__file__).parent / "results" / "rounds"
ROWS = (("last_change", "last change of $t(x)$"), ("last_violation", "last dual violation"))


def main():
    stored = json.loads((RESULTS / "instrumented.json").read_text(encoding="utf-8"))
    style()
    fig, axes = plt.subplots(len(ROWS), len(stored), figsize=(TEXT_WIDTH, 0.72 * TEXT_WIDTH), layout="constrained",
                             squeeze=False)
    for column, (name, problem) in enumerate(stored.items()):
        data = np.load(RESULTS / f"instrumented_{name}.npz")
        rounds = problem["rounds at level 0"]
        for row, (field, label) in enumerate(ROWS):
            axis = axes[row, column]
            last = data[field]
            order = np.argsort(last, kind="stable")
            quiet = last[order] == 0
            axis.scatter(*data["X"][order][quiet].T, s=0.15, color=FAINT, lw=0, rasterized=True)
            late = last[order][~quiet]
            shown = axis.scatter(*data["X"][order][~quiet].T, s=0.6 + 6 * (late / rounds) ** 3, c=late,
                                 cmap="cividis", vmin=1, vmax=rounds, lw=0, rasterized=True)
            axis.set_aspect("equal")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if row == 0:
                axis.set_title(f"{problem['kind']}\n$n = {problem['n']}$, seed {problem['seed']}", fontsize=9)
            if column == 0:
                axis.set_ylabel(label)
            fig.colorbar(shown, ax=axis, shrink=0.8, pad=0.02, ticks=[1, rounds // 2, rounds])
    save(fig, RESULTS, "rounds_where")


if __name__ == "__main__":
    main()
