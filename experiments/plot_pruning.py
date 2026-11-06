"""Thesis figure: shielding fences of one x_A and the candidates that pruning drops (squared Euclidean cost).

Left: X with its Delaunay mesh, x_A and its 1-ring candidates S(x_A). Right: Y around t(x_A); for each candidate the
fence {y : <x_s - x_A, y - t(x_s)> = 0} (5.5), the unshielded polygon P(x_A) and the missed targets Y_hat = (Y in P) without t(x_A), which is in spt pi already.
Points of Y in P (Y_hat) are not shielded by any candidate. A candidate is dropped when its fence is no face of P (SquaredEuclidean(prune=True).needed_candidates).
t is the exact optimal assignment between equal masses on uniform random points.
"""
import shutil
from pathlib import Path

import matplotlib
import matplotlib.patches
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment, linprog
from scipy.spatial import HalfspaceIntersection, ConvexHull
from scipy.spatial.distance import cdist

from msot.cost import SquaredEuclidean
from msot.mesh import one_ring

TEXT_WIDTH = 345 / 72.27  # \textwidth of the article class at 10pt, in inches: the figure is used at 100%
LATEX = shutil.which("pdflatex") is not None
KEPT, DROPPED, INK, MUTED, FAINT, REGION = "#2b5d8c", "#d17a22", "#1d232b", "#6b7380", "#c9ced6", "#dbe6f1"


def problem(seed, n=300):
    rng = np.random.default_rng(seed)
    X, Y = rng.random((n, 2)), rng.random((n, 2))
    t = linear_sum_assignment(cdist(X, Y, "sqeuclidean"))[1]
    return X, Y, t


def fences(X, Y, t, a, S):
    index_pointer, indices = S
    candidates = indices[index_pointer[a]:index_pointer[a + 1]]
    normals = X[candidates] - X[a]
    ys = Y[t[candidates]]
    needed = SquaredEuclidean(prune=True).needed_candidates(X[a], X[candidates], ys, 1e-12)
    return candidates, normals, ys, needed


def polygon(normals, ys, centre, box):
    """P = {y : <n_s, y - y_s> <= 0} intersected with the plotting box, as ordered vertices."""
    (x0, x1), (y0, y1) = box
    halfspaces = [np.r_[n, -n @ y] for n, y in zip(normals, ys)]
    halfspaces += [[-1, 0, x0], [1, 0, -x1], [0, -1, y0], [0, 1, -y1]]
    vertices = HalfspaceIntersection(np.array(halfspaces, float), centre).intersections
    return vertices[ConvexHull(vertices).vertices]


def figure(seed, a, path=None):
    X, Y, t = problem(seed)
    S = one_ring(X)
    candidates, normals, ys, needed = fences(X, Y, t, a, S)
    target = Y[t[a]]
    half = 1.6 * np.abs(ys - target).max()
    box = ((target[0] - half, target[0] + half), (target[1] - half, target[1] + half))
    P = polygon(normals[needed], ys[needed], chebyshev(normals[needed], ys[needed], box), box)

    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.linewidth": 0.5, "axes.edgecolor": MUTED})
    if LATEX:
        plt.rcParams.update({"pgf.texsystem": "pdflatex", "pgf.rcfonts": False,
                             "pgf.preamble": r"\usepackage{amsmath}\usepackage{amssymb}"})
    else:
        plt.rcParams.update({"mathtext.fontset": "cm", "font.serif": ["cmr10"], "axes.formatter.use_mathtext": True})
    fig, (left, right) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 0.56 * TEXT_WIDTH), constrained_layout=True)

    xa = X[a]
    hx = 1.25 * np.abs(X[candidates] - xa).max()
    index_pointer, indices = S
    for i in range(len(X)):
        for j in indices[index_pointer[i]:index_pointer[i + 1]]:
            if i < j:
                left.plot(*X[[i, j]].T, color=FAINT, lw=0.4, zorder=1)
    left.scatter(*X.T, s=4, color=MUTED, zorder=2, lw=0)
    for x, keep in zip(X[candidates], needed):
        left.scatter(*x, s=20, zorder=4, color="white" if not keep else KEPT, edgecolor=KEPT if keep else DROPPED,
                     lw=1.1, marker="o" if keep else "s")
    left.scatter(*xa, s=28, color=INK, zorder=5)
    named = np.flatnonzero(needed)[np.argmax(X[candidates][needed, 1])]
    left.annotate(r"$x_s$", X[candidates][named], xytext=(5, -13), textcoords="offset points", color=KEPT)
    left.annotate(r"$x_A$", xa, xytext=(5, 9), textcoords="offset points", color=INK)
    left.set(xlim=(xa[0] - hx, xa[0] + hx), ylim=(xa[1] - hx, xa[1] + hx), aspect="equal", xticks=[], yticks=[])
    left.set_title(r"(a) $X$: candidates $x_s \in S(x_A)$", fontsize=9)

    right.fill(*P.T, color=REGION, zorder=0, lw=0)
    (bx0, bx1), (by0, by1) = box
    for n, y, keep in zip(normals, ys, needed):
        direction = np.array([-n[1], n[0]]) / np.linalg.norm(n)
        line = np.vstack([y - 4 * half * direction, y + 4 * half * direction])
        right.plot(*line.T, color=KEPT if keep else DROPPED, lw=0.9 if keep else 1.0,
                   ls="-" if keep else (0, (3, 2)), zorder=2)
        arrow = 0.13 * half * n / np.linalg.norm(n)
        right.annotate("", y + arrow, y, arrowprops=dict(arrowstyle="-|>", color=KEPT if keep else DROPPED,
                                                         lw=0.7, mutation_scale=6), zorder=3)
        right.scatter(*y, s=20, zorder=4, color=KEPT if keep else "white", edgecolor=KEPT if keep else DROPPED,
                      lw=1.1, marker="o" if keep else "s")
    right.annotate(r"$y_s$", ys[named], xytext=(-17, 5), textcoords="offset points", color=KEPT)
    inside = np.all(np.einsum("sd,psd->ps", normals[needed], Y[:, None] - ys[needed][None]) <= 1e-12, axis=1)
    visible = (Y[:, 0] > bx0) & (Y[:, 0] < bx1) & (Y[:, 1] > by0) & (Y[:, 1] < by1)
    right.scatter(*Y[visible & ~inside].T, s=4, color=MUTED, zorder=1, lw=0)
    right.scatter(*Y[inside].T, s=10, color=INK, zorder=3.5, lw=0, marker="D")
    right.scatter(*target, s=24, color=INK, zorder=5, lw=0, marker="D")
    right.annotate(r"$t(x_A)$", target, xytext=(4, -11), textcoords="offset points", color=INK)
    centre = P.mean(axis=0)
    right.annotate(r"$P(x_A)$", centre, xytext=(-14, 10), textcoords="offset points", color=KEPT)
    right.set(xlim=(bx0, bx1), ylim=(by0, by1), aspect="equal", xticks=[], yticks=[])
    right.set_title(r"(b) $Y$: fences through $y_s = t(x_s)$", fontsize=9)

    handles = [plt.Line2D([], [], color=KEPT, lw=1.0, marker="o", ms=4, label="kept fence"),
               plt.Line2D([], [], color=DROPPED, lw=1.0, ls=(0, (3, 2)), marker="s", mfc="white", ms=4,
                          label="dropped fence"),
               matplotlib.patches.Patch(color=REGION, label=r"unshielded $P(x_A)$"),
               plt.Line2D([], [], color=INK, lw=0, marker="D", ms=3, label=r"$\hat Y(x_A) = Y \cap P(x_A)$")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, fontsize=8,
               handlelength=1.8, handletextpad=0.5, columnspacing=1.2)
    if path:
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02, dpi=600, backend="pgf" if LATEX else None)
    return fig


def chebyshev(normals, ys, box):
    """Centre of the largest disc inside P and the box, an interior point for HalfspaceIntersection."""
    (x0, x1), (y0, y1) = box
    A = np.vstack([normals, [[-1, 0], [1, 0], [0, -1], [0, 1]]])
    b = np.r_[np.einsum("sd,sd->s", normals, ys), -x0, x1, -y0, y1]
    norm = np.linalg.norm(A, axis=1)
    result = linprog([0, 0, -1], A_ub=np.c_[A, norm], b_ub=b, bounds=[(None, None)] * 2 + [(0, None)])
    return result.x[:2]


if __name__ == "__main__":
    results = Path(__file__).parent / "results"
    results.mkdir(exist_ok=True)
    for suffix in ("pdf", "png"):
        figure(seed=1, a=283, path=results / f"pruning_fences.{suffix}")
