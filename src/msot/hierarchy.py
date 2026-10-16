from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import QhullError

from .mesh import one_ring


@dataclass
class Hierarchy:
    points: np.ndarray
    mass: list = field(default_factory=list)
    rep: list = field(default_factory=list)
    rad: list = field(default_factory=list)
    parent: list = field(default_factory=list)
    cell_of: list = field(default_factory=list)
    neighbours: list = field(default_factory=list)
    _children: dict = field(default_factory=dict, repr=False)

    @property
    def K(self):
        return len(self.mass) - 1

    def number_of_cells(self, k):
        return len(self.mass[k])

    def rep_points(self, k):
        return self.points[self.rep[k]]

    def children(self, k):
        """Children of all cells of level k as CSR (index_pointer, indices) into level k-1."""
        if k not in self._children:
            parent = self.parent[k - 1]
            counts = np.bincount(parent, minlength=self.number_of_cells(k))
            index_pointer = np.concatenate([[0], np.cumsum(counts)])
            self._children[k] = index_pointer, np.argsort(parent, kind="stable")
        return self._children[k]


def _level_zero(points, mass):
    n = len(points)
    return Hierarchy(points, [mass], [np.arange(n)], [np.zeros(n)], [], [np.arange(n)], [_mesh(points)])


def _mesh(points):
    n = len(points)
    try:
        return one_ring(points)
    except (QhullError, ValueError):
        indices = np.array([j for i in range(n) for j in range(n) if j != i], dtype=np.int32)
        return np.arange(n + 1, dtype=np.int32) * (n - 1), indices


def _match(labels, edge_start, edge_end):
    """One round of greedy matching of label groups along edges, given sorted by length."""
    partner = np.full(labels.max() + 1, -1)
    for a, b in zip(labels[edge_start].tolist(), labels[edge_end].tolist()):
        if a != b and partner[a] < 0 and partner[b] < 0:
            partner[a], partner[b] = b, a
    groups = np.arange(len(partner))
    root = np.where(partner < 0, groups, np.minimum(groups, partner))
    return np.unique(root, return_inverse=True)[1][labels]


def _coarsening_parent(H):
    k = H.K
    index_pointer, indices = H.neighbours[k]
    edge_start = np.repeat(np.arange(H.number_of_cells(k)), np.diff(index_pointer))
    edge_end = indices
    keep = edge_start < edge_end
    edge_start, edge_end = edge_start[keep], edge_end[keep]
    points = H.rep_points(k)
    order = np.argsort(np.linalg.norm(points[edge_start] - points[edge_end], axis=1), kind="stable")
    edge_start, edge_end = edge_start[order], edge_end[order]
    labels = np.arange(H.number_of_cells(k))
    return _match(_match(labels, edge_start, edge_end), edge_start, edge_end)


def _add_level(H, parent):
    k = H.K
    n = parent.max() + 1
    cell_of = parent[H.cell_of[k]]
    mass = np.bincount(parent, H.mass[k], minlength=n)
    weights = H.mass[0]
    centroid = np.stack([np.bincount(cell_of, weights * H.points[:, i], minlength=n) for i in range(2)], axis=1)
    centroid /= np.bincount(cell_of, weights, minlength=n)[:, None]
    distance = np.linalg.norm(H.rep_points(k) - centroid[parent], axis=1)
    order = np.lexsort((distance, parent))
    first = order[np.r_[True, parent[order][1:] != parent[order][:-1]]]
    rep = np.empty(n, dtype=H.rep[k].dtype)
    rep[parent[first]] = H.rep[k][first]
    rad = np.zeros(n)
    np.maximum.at(rad, cell_of, np.linalg.norm(H.points - H.points[rep[cell_of]], axis=1))
    H.parent.append(parent)
    H.cell_of.append(cell_of)
    H.mass.append(mass)
    H.rep.append(rep)
    H.rad.append(rad)
    H.neighbours.append(_mesh(H.points[rep]))


def _coarsen(H):
    _add_level(H, _coarsening_parent(H) if H.number_of_cells(H.K) > 1 else np.zeros(1, dtype=int))


def _add_top(H):
    _add_level(H, np.zeros(H.number_of_cells(H.K), dtype=int))


def agglomerate(points, mass, n_dense=256):
    H = _level_zero(points, mass)
    while H.number_of_cells(H.K) > n_dense:
        _coarsen(H)
    _add_top(H)
    return H


def build_pair(X, mu, Y, nu, n_dense=256):
    HX, HY = _level_zero(X, mu), _level_zero(Y, nu)
    while max(HX.number_of_cells(HX.K), HY.number_of_cells(HY.K)) > n_dense:
        _coarsen(HX)
        _coarsen(HY)
    _add_top(HX)
    _add_top(HY)
    return HX, HY


def quadtree(points, mass, K, corner, side):
    """Def. 4.4 with K levels over the square [corner, corner + side]; reps are existing points as in agglomerate."""
    H = _level_zero(points, mass)
    finest = 2 ** (K - 1)
    cube = np.clip(np.floor((points - corner) / side * finest).astype(np.int64), 0, finest - 1)
    for k in range(1, K + 1):
        cube_k = cube >> (k - 1)
        label = np.unique(cube_k[:, 0] * finest + cube_k[:, 1], return_inverse=True)[1]
        parent = np.empty(H.number_of_cells(k - 1), dtype=np.int64)
        parent[H.cell_of[k - 1]] = label
        _add_level(H, parent)
    return H


def build_pair_quadtree(X, mu, Y, nu):
    """Quadtrees over one common square, with the smallest K such that the finest cubes hold about 4 points."""
    both = np.vstack([X, Y])
    corner = both.min(axis=0)
    side = max((both.max(axis=0) - corner).max(), np.finfo(float).tiny)
    K = 1
    while max(len(X), len(Y)) > 4 ** K:
        K += 1
    return quadtree(X, mu, K, corner, side), quadtree(Y, nu, K, corner, side)
