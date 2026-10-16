"""Figures of the mesh hierarchy (option B): the two matching rounds, the partition per level, overview vs. quadtree."""
from itertools import count
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, QhullError

from msot.hierarchy import _match, agglomerate, build_pair_quadtree
from msot.problems import sample_points

RESULTS = Path(__file__).parent / "results"
NUMBER_OF_POINTS = 250
N_DENSE = 6
SEED = 0
KINDS = ["uniform", "clustered"]
PALETTE = plt.get_cmap("tab10").colors
ROUND_ONE_COLOUR, ROUND_TWO_COLOUR, COPY_COLOUR = "tab:blue", "tab:orange", "tab:red"


def mesh_edges(neighbours):
    index_pointer, indices = neighbours
    edge_start = np.repeat(np.arange(len(index_pointer) - 1), np.diff(index_pointer))
    keep = edge_start < indices
    return edge_start[keep], indices[keep]


def matching_rounds(H, k):
    """Replays _coarsening_parent for step k -> k+1 and returns the edges each round used."""
    edge_start, edge_end = mesh_edges(H.neighbours[k])
    points = H.rep_points(k)
    order = np.argsort(np.linalg.norm(points[edge_start] - points[edge_end], axis=1), kind="stable")
    edge_start, edge_end = edge_start[order], edge_end[order]
    round_one = _match(np.arange(H.number_of_cells(k)), edge_start, edge_end)
    round_two = _match(round_one, edge_start, edge_end)
    assert np.array_equal(round_two, H.parent[k])
    paired = np.flatnonzero(round_one[edge_start] == round_one[edge_end])
    merging = np.flatnonzero((round_two[edge_start] == round_two[edge_end]) & (round_one[edge_start] != round_one[edge_end]))
    merging = merging[np.unique(round_two[edge_start[merging]], return_index=True)[1]]
    return edge_start, edge_end, paired, merging, round_one, round_two


def cell_adjacency(H, k):
    edge_start, edge_end = mesh_edges(H.neighbours[0])
    return H.cell_of[k][edge_start], H.cell_of[k][edge_end]


def cell_colours(H, k, rng):
    """Greedy colouring of the cells so that cells joined by a level-0 Delaunay edge differ."""
    neighbour_sets = [set() for _ in range(H.number_of_cells(k))]
    for a, b in zip(*(cells.tolist() for cells in cell_adjacency(H, k))):
        if a != b:
            neighbour_sets[a].add(b)
            neighbour_sets[b].add(a)
    colour = np.full(H.number_of_cells(k), -1)
    for cell in rng.permutation(H.number_of_cells(k)):
        used = {colour[neighbour] for neighbour in neighbour_sets[cell]}
        colour[cell] = next(c for c in count() if c not in used)
    return np.array(PALETTE)[colour % len(PALETTE)]


def number_of_split_cells(H, k):
    """Cells whose points are not connected by level-0 Delaunay edges inside the cell."""
    edge_start, edge_end = mesh_edges(H.neighbours[0])
    inside = H.cell_of[k][edge_start] == H.cell_of[k][edge_end]
    n = len(H.points)
    graph = coo_matrix((np.ones(inside.sum()), (edge_start[inside], edge_end[inside])), shape=(n, n))
    component = connected_components(graph, directed=False)[1]
    pairs = np.unique(np.stack([H.cell_of[k], component], axis=1), axis=0)
    return int((np.bincount(pairs[:, 0]) > 1).sum())


def children_histogram(H, k):
    number_of_children = np.bincount(H.parent[k - 1], minlength=H.number_of_cells(k))
    return np.bincount(number_of_children, minlength=5)


def elongation(H, k):
    """sqrt of the eigenvalue ratio of the point covariance, per cell with at least 3 points."""
    values = []
    for cell in range(H.number_of_cells(k)):
        cell_points = H.points[H.cell_of[k] == cell]
        if len(cell_points) >= 3:
            eigenvalues = np.linalg.eigvalsh(np.cov(cell_points.T))
            values.append(np.sqrt(eigenvalues[1] / max(eigenvalues[0], 1e-15)))
    return np.array(values)


def setup(axis, title):
    axis.set_aspect("equal")
    axis.set_xlim(-0.03, 1.03)
    axis.set_ylim(-0.03, 1.03)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(title, fontsize=10)


def draw_partition(axis, H, k, colours, circles=False, links=False):
    point_colours = colours[H.cell_of[k]]
    if k > 0:
        order = np.argsort(H.cell_of[k], kind="stable")
        boundaries = np.cumsum(np.bincount(H.cell_of[k], minlength=H.number_of_cells(k)))[:-1]
        for cell, members in enumerate(np.split(order, boundaries)):
            cell_points = H.points[members]
            if len(cell_points) == 2:
                axis.plot(*cell_points.T, color=colours[cell], lw=3, alpha=0.35, solid_capstyle="round")
            elif len(cell_points) > 2:
                try:
                    hull = cell_points[ConvexHull(cell_points).vertices]
                except QhullError:
                    continue
                axis.add_patch(Polygon(hull, facecolor=colours[cell], edgecolor=colours[cell], alpha=0.3, lw=1))
    axis.scatter(*H.points.T, c=point_colours, s=9, zorder=2, linewidths=0)
    if links and k > 0:
        child_reps = H.rep_points(k - 1)
        parent_reps = H.rep_points(k)[H.parent[k - 1]]
        axis.add_collection(LineCollection(np.stack([child_reps, parent_reps], axis=1), colors="black", lw=0.7, zorder=3))
        axis.scatter(*child_reps.T, s=18, facecolors="white", edgecolors="black", linewidths=0.7, zorder=4)
    if circles:
        for centre, radius, colour in zip(H.rep_points(k), H.rad[k], colours):
            axis.add_patch(Circle(centre, radius, fill=False, ls="--", lw=0.8, edgecolor=colour, zorder=1))
    if k > 0:
        axis.scatter(*H.rep_points(k).T, marker="*", s=70, c="black", edgecolors="white", linewidths=0.5, zorder=5)


def plot_matching(H, kind, path):
    steps = [k for k in range(H.K - 1) if H.number_of_cells(k) > 1]
    columns = min(len(steps), 2)
    rows = -(-len(steps) // columns)
    figure, axes = plt.subplots(rows, columns, figsize=(7 * columns, 7 * rows), squeeze=False)
    for axis, k in zip(axes.flat, steps):
        edge_start, edge_end, paired, merging, round_one, round_two = matching_rounds(H, k)
        reps = H.rep_points(k)
        segments = np.stack([reps[edge_start], reps[edge_end]], axis=1)
        axis.scatter(*H.points.T, s=3, c="0.75", zorder=0)
        axis.add_collection(LineCollection(segments, colors="0.75", lw=0.6, zorder=1))
        axis.add_collection(LineCollection(segments[paired], colors=ROUND_ONE_COLOUR, lw=2.4, zorder=2))
        axis.add_collection(LineCollection(segments[merging], colors=ROUND_TWO_COLOUR, lw=2.4, zorder=2))
        axis.scatter(*reps.T, s=10, c="black", zorder=3)
        single_after_one = np.bincount(round_one)[round_one] == 1
        copy = np.bincount(round_two)[round_two] == 1
        axis.scatter(*reps[single_after_one].T, s=90, facecolors="none", edgecolors=ROUND_ONE_COLOUR, lw=1.3, zorder=4)
        axis.scatter(*reps[copy].T, s=140, marker="s", facecolors="none", edgecolors=COPY_COLOUR, lw=1.6, zorder=4)
        setup(axis, f"{kind}: matching step {k} -> {k + 1}, {H.number_of_cells(k)} -> {H.number_of_cells(k + 1)} cells\n"
                    f"round 1: {len(paired)} pairs, {single_after_one.sum()} single; "
                    f"round 2: {len(merging)} merges; {copy.sum()} copy cells")
    for axis in axes.flat[len(steps):]:
        axis.axis("off")
    legend = [Line2D([], [], color="0.75", lw=0.6, label="Delaunay edge of level k reps"),
              Line2D([], [], color=ROUND_ONE_COLOUR, lw=2.4, label="round 1: pairs cells"),
              Line2D([], [], color=ROUND_TWO_COLOUR, lw=2.4, label="round 2: merges round-1 groups"),
              Line2D([], [], ls="", marker="o", mfc="none", mec=ROUND_ONE_COLOUR, ms=9, label="unmatched in round 1"),
              Line2D([], [], ls="", marker="s", mfc="none", mec=COPY_COLOUR, ms=10, label="copy cell (single child)")]
    figure.legend(handles=legend, loc="lower center", ncol=3, fontsize=10, frameon=False)
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_partition(H, k, kind, colours, path):
    figure, axis = plt.subplots(figsize=(8, 8))
    draw_partition(axis, H, k, colours, circles=True, links=True)
    setup(axis, f"{kind}: level {k} of K = {H.K}, {H.number_of_cells(k)} cells (points coloured by cell, convex hulls)\n"
                f"black stars: rep of level {k}; white dots: reps of level {k - 1} linked to their parent rep; "
                f"dashed: rad (4.1)")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def panel_text(H, k):
    lines = [f"{H.number_of_cells(k)} cells"]
    if k > 0:
        histogram = children_histogram(H, k)
        lines.append("children 1/2/3/4: " + "/".join(str(c) for c in histogram[1:5]))
        if len(histogram) > 5:
            lines.append(f"more than 4 children: {histogram[5:].sum()} (max {len(histogram) - 1})")
        lines.append(f"not contiguous: {number_of_split_cells(H, k)}")
    return "\n".join(lines)


def plot_overview(H, name, colour_sets, path):
    columns = 3
    rows = -(-(H.K + 1) // columns)
    figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 5.3 * rows), squeeze=False)
    for k, axis in enumerate(axes.flat[:H.K + 1]):
        draw_partition(axis, H, k, colour_sets[k])
        setup(axis, f"{name}: level {k} of K = {H.K}")
        axis.text(0.01, 0.99, panel_text(H, k), transform=axis.transAxes, va="top", fontsize=8,
                  bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.7"))
    for axis in axes.flat[H.K + 1:]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def report(H, name):
    print(f"== {name}, K = {H.K}")
    for k in range(1, H.K + 1):
        histogram = children_histogram(H, k)
        stretch = elongation(H, k)
        stretch_text = f"elongation median {np.median(stretch):.2f} max {stretch.max():.2f}" if len(stretch) else ""
        print(f"level {k}: {H.number_of_cells(k):4d} cells, children histogram {histogram[1:].tolist()}, "
              f"not contiguous {number_of_split_cells(H, k)}, median rad {np.median(H.rad[k]):.3f}, {stretch_text}")


def main():
    RESULTS.mkdir(exist_ok=True)
    written = []
    for kind in KINDS:
        points = sample_points(NUMBER_OF_POINTS, kind, np.random.default_rng(SEED))
        mass = np.full(len(points), 1 / len(points))
        hierarchies = {"mesh": agglomerate(points, mass, N_DENSE),
                       "quadtree": build_pair_quadtree(points, mass, points, mass)[0]}
        for name, H in hierarchies.items():
            report(H, f"{kind} {name}")
            colour_sets = [cell_colours(H, k, np.random.default_rng(k)) for k in range(H.K + 1)]
            path = RESULTS / f"hierarchy_overview_{name}_{kind}.png"
            plot_overview(H, f"{kind} {name}", colour_sets, path)
            written.append(path)
            if name == "mesh":
                path = RESULTS / f"hierarchy_matching_{kind}.png"
                plot_matching(H, kind, path)
                written.append(path)
                for k in range(1, H.K + 1):
                    path = RESULTS / f"hierarchy_partition_{kind}_level{k}.png"
                    plot_partition(H, k, kind, colour_sets[k], path)
                    written.append(path)
    print("written:", *written, sep="\n")


if __name__ == "__main__":
    main()
