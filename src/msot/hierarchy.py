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
    nbrs: list = field(default_factory=list)
    _children: dict = field(default_factory=dict, repr=False)

    @property
    def K(self):
        return len(self.mass) - 1

    def n_cells(self, k):
        return len(self.mass[k])

    def rep_points(self, k):
        return self.points[self.rep[k]]

    def children(self, k):
        """Children of all cells of level k as CSR (indptr, indices) into level k-1."""
        if k not in self._children:
            p = self.parent[k - 1]
            indptr = np.concatenate([[0], np.cumsum(np.bincount(p, minlength=self.n_cells(k)))])
            self._children[k] = indptr, np.argsort(p, kind="stable")
        return self._children[k]


def _level0(points, mass):
    n = len(points)
    return Hierarchy(points, [mass], [np.arange(n)], [np.zeros(n)], [], [np.arange(n)], [_mesh(points)])


def _mesh(points):
    n = len(points)
    try:
        return one_ring(points)
    except (QhullError, ValueError):
        indices = np.array([j for i in range(n) for j in range(n) if j != i], dtype=np.int32)
        return np.arange(n + 1, dtype=np.int32) * (n - 1), indices


def _match(labels, ea, eb):
    """One round of greedy matching of label groups along edges (ea, eb), given sorted by length."""
    partner = np.full(labels.max() + 1, -1)
    for a, b in zip(labels[ea].tolist(), labels[eb].tolist()):
        if a != b and partner[a] < 0 and partner[b] < 0:
            partner[a], partner[b] = b, a
    ids = np.arange(len(partner))
    root = np.where(partner < 0, ids, np.minimum(ids, partner))
    return np.unique(root, return_inverse=True)[1][labels]


def _coarsening_parent(H):
    k = H.K
    indptr, indices = H.nbrs[k]
    ea = np.repeat(np.arange(H.n_cells(k)), np.diff(indptr))
    eb = indices
    keep = ea < eb
    ea, eb = ea[keep], eb[keep]
    P = H.rep_points(k)
    order = np.argsort(np.linalg.norm(P[ea] - P[eb], axis=1), kind="stable")
    ea, eb = ea[order], eb[order]
    labels = np.arange(H.n_cells(k))
    return _match(_match(labels, ea, eb), ea, eb)


def _add_level(H, parent):
    k = H.K
    n = parent.max() + 1
    cell_of = parent[H.cell_of[k]]
    mass = np.bincount(parent, H.mass[k], minlength=n)
    w = H.mass[0]
    centroid = np.stack([np.bincount(cell_of, w * H.points[:, i], minlength=n) for i in range(2)], axis=1)
    centroid /= np.bincount(cell_of, w, minlength=n)[:, None]
    d = np.linalg.norm(H.rep_points(k) - centroid[parent], axis=1)
    order = np.lexsort((d, parent))
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
    H.nbrs.append(_mesh(H.points[rep]))


def _coarsen(H):
    _add_level(H, _coarsening_parent(H) if H.n_cells(H.K) > 1 else np.zeros(1, dtype=int))


def _add_top(H):
    _add_level(H, np.zeros(H.n_cells(H.K), dtype=int))


def agglomerate(points, mass, n_dense=256):
    H = _level0(points, mass)
    while H.n_cells(H.K) > n_dense:
        _coarsen(H)
    _add_top(H)
    return H


def build_pair(X, mu, Y, nu, n_dense=256):
    HX, HY = _level0(X, mu), _level0(Y, nu)
    while max(HX.n_cells(HX.K), HY.n_cells(HY.K)) > n_dense:
        _coarsen(HX)
        _coarsen(HY)
    _add_top(HX)
    _add_top(HY)
    return HX, HY
