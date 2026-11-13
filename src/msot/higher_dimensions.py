"""Test problems and candidate pruning in d >= 2 dimensions (branch higher-dimensions).

Kept apart from the planar modules: problems.py and cost.FacePruning stay as they are. The mesh hierarchy
(Delaunay 1-ring), the search and the solver work in any dimension already.
"""
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, QhullError

from .cost import SquaredEuclidean
from .problems import quantize


def make_problem(n, kind, rng, dimension, atoms=10**9):
    """As problems.make_problem, in [0, 1]^dimension: uniform or clustered points, Gaussian-mixture masses."""
    X, Y = (sample_points(n, kind, rng, dimension) for _ in range(2))
    mu, nu = gaussian_masses(X, rng), gaussian_masses(Y, rng)
    return X, quantize(mu, atoms), Y, quantize(nu, atoms)


def sample_points(n, kind, rng, dimension):
    if kind == "uniform":
        return rng.random((n, dimension))
    if kind == "clustered":
        return _gaussian_mixture(n, rng, dimension)
    raise ValueError(kind)


def _gaussian_mixture(n, rng, dimension, n_clusters=4, std_range=(0.03, 0.15)):
    """Mixture of `n_clusters` Gaussians with random centres and covariances, truncated to [0, 1]^dimension."""
    centres = rng.uniform(0.15, 0.85, (n_clusters, dimension))
    covariances = [_random_covariance(rng, np.square(rng.uniform(*std_range, dimension))) for _ in range(n_clusters)]
    points = np.empty((0, dimension))
    while len(points) < n:
        clusters = rng.integers(n_clusters, size=n)
        samples = np.array([rng.multivariate_normal(centres[i], covariances[i]) for i in clusters])
        points = np.vstack([points, samples[np.all((samples >= 0) & (samples <= 1), axis=1)]])
    return points[:n]


def _random_covariance(rng, eigenvalues):
    """Covariance with the given eigenvalues in a uniformly random orthonormal frame (QR of a Gaussian matrix)."""
    q, r = np.linalg.qr(rng.normal(size=(len(eigenvalues), len(eigenvalues))))
    rotation = q * np.sign(np.diag(r))
    return rotation @ np.diag(eigenvalues) @ rotation.T


def gaussian_masses(points, rng, n_gauss=4, eigenvalue_range=(1.8, 100.0), floor=1e-3):
    """As problems.gaussian_masses (Schmitzer Sect. 6.1): eigenvalues in eigenvalue_range times spacing^2,
    spacing = n^(-1/dimension)."""
    n, dimension = points.shape
    spacing = n ** (-1 / dimension)
    density = np.zeros(n)
    for _ in range(n_gauss):
        eigenvalues = np.exp(rng.uniform(*np.log(eigenvalue_range), dimension)) * spacing**2
        covariance = _random_covariance(rng, eigenvalues)
        offsets = points - rng.random(dimension)
        quadratic_form = np.einsum("ij,jk,ik->i", offsets, np.linalg.inv(covariance), offsets)
        density += rng.random() * np.exp(-quadratic_form / 2) / np.sqrt((2 * np.pi) ** dimension * eigenvalues.prod())
    return density + floor * density.mean()


class SquaredEuclideanPruning(SquaredEuclidean):
    """SquaredEuclidean with pruning in any dimension; the half-spaces are those of (5.5), as in 2D.

    face_test "exact": keep the candidates whose half-space is a facet of P(x_A), found per x_A with Qhull
    (Chebyshev centre c by an LP, facets = vertices other than 0 of the convex hull of 0 and the dual points
    n_s / (b_s - <n_s, c>)). P without a finite inscribed ball or degenerate hulls keep all candidates. In 2D the vectorised polygon test of FacePruning is used.
    face_test "closest": keep the share keep_share of the half-spaces closest to the mean of the partners t(x_s),
    a cheap estimate of t(x_A); this may drop needed candidates, which only costs search work (filter_missed keeps
    the result exact).
    """

    def __init__(self, face_test="exact", keep_share=0.6):
        super().__init__(prune=True)
        self.face_test, self.keep_share = face_test, keep_share

    def needed_candidates_all(self, X, S, ys, tolerance, xs=None, chunk=1024):
        if self.face_test == "exact" and X.shape[1] == 2:
            return super().needed_candidates_all(X, S, ys, tolerance, xs, chunk)
        index_pointer, indices = S
        xs = X[indices] if xs is None else xs
        owner = np.repeat(np.arange(len(X)), np.diff(index_pointer))
        normals, offsets = self._face_data(X[owner], xs, ys, tolerance)
        if self.face_test == "closest":
            return _closest(owner, index_pointer, normals, offsets, ys, self.keep_share)
        needed = np.ones(len(indices), bool)
        for a in range(len(X)):
            entries = slice(index_pointer[a], index_pointer[a + 1])
            needed[entries] = _facets(normals[entries], offsets[entries])
        return needed


def _closest(owner, index_pointer, normals, offsets, ys, keep_share):
    counts = np.diff(index_pointer)
    centre = np.stack([np.bincount(owner, ys[:, i], len(counts)) for i in range(ys.shape[1])], axis=1)
    centre /= np.maximum(counts, 1)[:, None]
    distance = (offsets - np.einsum("ij,ij->i", normals, centre[owner])) / np.linalg.norm(normals, axis=1)
    order = np.lexsort((distance, owner))
    rank = np.empty(len(owner), int)
    rank[order] = np.arange(len(owner)) - np.repeat(index_pointer[:-1], counts)
    return rank < np.ceil(keep_share * counts[owner])


def _facets(normals, offsets):
    """Which half-spaces {<normal, y> <= offset} are facets of their intersection P; all if P is unbounded or flat."""
    keep_all = np.ones(len(normals), bool)
    if len(normals) <= normals.shape[1]:
        return keep_all
    lengths = np.linalg.norm(normals, axis=1)
    objective = np.r_[np.zeros(normals.shape[1]), -1.0]
    centre = linprog(objective, A_ub=np.c_[normals, lengths], b_ub=offsets,
                     bounds=[(None, None)] * normals.shape[1] + [(0, None)], method="highs")
    if centre.status != 0 or centre.x[-1] <= 1e-12 * lengths.max():
        return keep_all
    dual = normals / (offsets - normals @ centre.x[:-1])[:, None]
    try:
        hull = ConvexHull(np.vstack([dual, np.zeros(normals.shape[1])]))  # the origin covers unbounded P
    except QhullError:
        return keep_all
    needed = np.zeros(len(normals), bool)
    needed[hull.vertices[hull.vertices < len(normals)]] = True
    return needed
