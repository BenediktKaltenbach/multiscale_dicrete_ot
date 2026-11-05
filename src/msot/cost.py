import numpy as np


class FacePruning:
    """Candidate pruning by the faces of the polygon of half-planes {p : <normal, p> <= offset} per x_A.

    The cost provides _face_data(xA, xs, ys, tolerance) -> (normals, offsets) per candidate entry; for
    SquaredEuclidean these are the exact half-planes (5.5) in y, for MTW costs tangent half-planes in the
    c-exponential chart. Candidates that are no face are dropped from the search; shield then checks the missed
    cells exactly against them (filter_missed), so correctness never depends on the pruning.
    """
    prune = False

    def needed_candidates(self, xA, xs, ys, tolerance):
        """needed_candidates_all for a single x_A."""
        return self.needed_candidates_all(xA[None], (np.array([0, len(xs)]), np.arange(len(xs))), ys, tolerance, xs)

    def needed_candidates_all(self, X, S, ys, tolerance, xs=None, chunk=1024):
        """Mask of the candidates whose half-plane is a face of the polygon of their x_A (Fig. 3).

        S is CSR (index_pointer, indices) of candidates on X, ys their partners t(x_s). Unbounded or empty polygons
        keep all candidates. Vectorised over x_A: candidate lists padded to the longest, in chunks of x_A.
        """
        index_pointer, indices = S
        needed = np.ones(len(indices), bool)
        if not self.prune:
            return needed
        counts = np.diff(index_pointer)
        xs = X[indices] if xs is None else xs
        for start in range(0, len(counts), chunk):
            group = np.arange(start, min(start + chunk, len(counts)))
            entries = np.arange(index_pointer[group[0]], index_pointer[group[-1] + 1])
            slot = entries - np.repeat(index_pointer[group], counts[group])
            row = np.repeat(group - start, counts[group])
            normals, offsets = self._face_data(X[group][row], xs[entries], ys[entries], tolerance)
            needed[entries] = _faces(counts[group], row, slot, normals, offsets)
        return needed


def _faces(counts, row, slot, entry_normals, entry_offsets):
    """For half-planes {<normal, p> <= offset} grouped by row: is each a face of its group's polygon?"""
    groups, width = len(counts), max(int(counts.max()), 1)
    valid = np.zeros((groups, width), bool)
    valid[row, slot] = True
    normals = np.zeros((groups, width, 2))
    normals[row, slot] = entry_normals
    offsets = np.zeros((groups, width))
    offsets[row, slot] = entry_offsets
    angles = np.full((groups, width), 4 * np.pi)
    angles[row, slot] = np.arctan2(entry_normals[:, 1], entry_normals[:, 0])
    angles = np.sort(angles, axis=1)
    gaps = np.where(np.arange(width - 1) < (counts - 1)[:, None], np.diff(angles, axis=1), -np.inf)
    wrap = angles[:, 0] + 2 * np.pi - angles[np.arange(groups), np.maximum(counts - 1, 0)]
    bounded = (counts >= 3) & (np.maximum(gaps.max(axis=1, initial=-np.inf), wrap) < np.pi)

    i, j = np.triu_indices(width, 1)
    determinant = normals[:, i, 0] * normals[:, j, 1] - normals[:, i, 1] * normals[:, j, 0]
    length = np.linalg.norm(normals, axis=2)
    regular = valid[:, i] & valid[:, j] & (np.abs(determinant) > 1e-14 * length[:, i] * length[:, j])
    determinant = np.where(regular, determinant, 1.0)
    vertices = np.stack([offsets[:, i] * normals[:, j, 1] - offsets[:, j] * normals[:, i, 1],
                         normals[:, i, 0] * offsets[:, j] - normals[:, j, 0] * offsets[:, i]], axis=2)
    vertices /= determinant[..., None]
    slack = offsets[:, :, None] - np.einsum("gmd,gpd->gmp", normals, vertices)
    margin = 1e-9 * (np.abs(offsets)[:, :, None] + length[:, :, None] * np.abs(vertices).sum(axis=2)[:, None, :])
    feasible = regular & ((slack >= -margin) | ~valid[:, :, None]).all(axis=1)
    faces = ((np.abs(slack) <= margin) & feasible[:, None, :]).sum(axis=2) >= 2
    keep_all = ~bounded | ~feasible.any(axis=1)
    return (faces | keep_all[:, None])[row, slot]


class SquaredEuclidean(FacePruning):
    def __init__(self, prune=False):
        self.prune = prune

    def c(self, x, y):
        return np.sum((x - y) ** 2, axis=-1)

    def psi(self, xA, xs, y):
        """(5.1)"""
        return self.c(xA, y) - self.c(xs, y)

    def psi_hat(self, xA, xs, rep_y, rad_y):
        """(5.7)"""
        return self.psi(xA, xs, rep_y) - 2 * np.linalg.norm(xs - xA, axis=-1) * rad_y

    def _face_data(self, xA, xs, ys, tolerance):
        """The exact half-planes (5.5): <x_s - x_A, y> <= <x_s - x_A, y_s>."""
        normals = xs - xA
        return normals, np.einsum("ij,ij->i", normals, ys) + tolerance / 2


class SqrtQuadratic(FacePruning):
    """c(x, y) = sqrt(1 + |x - y|^2), an MTW cost (A3s).

    With d = x - y and w = c(x, y): grad_x c = d / w, D_xy c = -(I - d d^T / w^2) / w with eigenvalues -1/w and
    -1/w^3, so ||D_xy c|| <= 1 globally, and (D_xy c)^(-1) = -w (I + d d^T).
    """

    def __init__(self, prune=False):
        self.prune = prune

    def c(self, x, y):
        return np.sqrt(1 + np.sum((x - y) ** 2, axis=-1))

    def psi(self, xA, xs, y):
        """(5.1)"""
        return self.c(xA, y) - self.c(xs, y)

    def psi_hat(self, xA, xs, rep_y, rad_y):
        """Lipschitz bound: |grad_y psi| <= ||D_xy c|| |x_s - x_A| <= |x_s - x_A|."""
        return self.psi(xA, xs, rep_y) - np.linalg.norm(xs - xA, axis=-1) * rad_y

    def chart(self, xA, y):
        """c-exponential chart p = -grad_x c(x_A, y) (Loeper); used for pruning and plots, not by the search."""
        return (y - xA) / self.c(xA, y)[..., None]

    def _face_data(self, xA, xs, ys, tolerance):
        """Tangent half-planes {<n_s, p - p_s> <= 0} of the unshielded sets in the chart p = -grad_x c(x_A, .).

        n_s = -(D_xy c(x_A, y_s))^(-T) grad_y psi(y_s) = w_A (I + d_A d_A^T) (d_s / w_s - d_A / w_A),
        p_s = -d_A / w_A with d_A = x_A - y_s, d_s = x_s - y_s.
        """
        dA, ds = xA - ys, xs - ys
        wA, ws = self.c(xA, ys)[:, None], self.c(xs, ys)[:, None]
        v = ds / ws - dA / wA
        normals = wA * (v + dA * np.einsum("ij,ij->i", dA, v)[:, None])
        return normals, np.einsum("ij,ij->i", normals, -dA / wA) + tolerance / 2
