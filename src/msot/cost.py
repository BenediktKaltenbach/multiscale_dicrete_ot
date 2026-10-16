import numpy as np


class SquaredEuclidean:
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

    def needed_candidates(self, xA, xs, ys, tolerance):
        """needed_candidates_all for a single x_A."""
        return self.needed_candidates_all(xA[None], (np.array([0, len(xs)]), np.arange(len(xs))), ys, tolerance, xs)

    def needed_candidates_all(self, X, S, ys, tolerance, xs=None, chunk=1024):
        """Mask of the candidates whose half-plane (5.5) is a face of the polytope P of their x_A (Fig. 3).

        S is CSR (index_pointer, indices) of candidates on X, ys their partners t(x_s). The others are implied by
        the rest and leave Y_hat unchanged. Unbounded P (q <= 0) or empty P (possible for a non-optimal coupling)
        keeps all. Vectorised over x_A: candidate lists padded to the longest, in chunks of x_A.
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
            needed[entries] = self._faces(X[group], counts[group], row, slot, xs[entries], ys[entries], tolerance)
        return needed

    @staticmethod
    def _faces(xA, counts, row, slot, xs, ys, tolerance):
        groups, width = len(counts), max(int(counts.max()), 1)
        valid = np.zeros((groups, width), bool)
        valid[row, slot] = True
        normals = np.zeros((groups, width, 2))
        normals[row, slot] = xs - xA[row]
        offsets = np.zeros((groups, width))
        offsets[row, slot] = np.einsum("ij,ij->i", xs - xA[row], ys) + tolerance / 2
        angles = np.full((groups, width), 4 * np.pi)
        angles[row, slot] = np.arctan2(normals[row, slot, 1], normals[row, slot, 0])
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
