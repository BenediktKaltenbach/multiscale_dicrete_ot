import numpy as np


class SquaredEuclidean:
    def __init__(self, prune=True):
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
        """Mask of the candidates (xs, ys) whose half-plane (5.5) is a face of the polytope P (Fig. 3).

        The others are implied by the rest and leave Y_hat unchanged. Unbounded P (q <= 0) or empty P
        (possible for a non-optimal coupling) keeps all.
        """
        keep = np.ones(len(xs), bool)
        if not self.prune or len(xs) < 3:
            return keep
        normals = xs - xA
        offsets = np.einsum("ij,ij->i", normals, ys) + tolerance / 2
        angles = np.sort(np.arctan2(normals[:, 1], normals[:, 0]))
        if np.diff(np.append(angles, angles[0] + 2 * np.pi)).max() >= np.pi:
            return keep
        i, j = np.triu_indices(len(xs), 1)
        determinant = normals[i, 0] * normals[j, 1] - normals[i, 1] * normals[j, 0]
        regular = np.abs(determinant) > 1e-14 * np.linalg.norm(normals[i], axis=1) * np.linalg.norm(normals[j], axis=1)
        i, j, determinant = i[regular], j[regular], determinant[regular]
        vertices = np.stack([offsets[i] * normals[j, 1] - offsets[j] * normals[i, 1],
                             normals[i, 0] * offsets[j] - normals[j, 0] * offsets[i]], axis=1) / determinant[:, None]
        slack = offsets[:, None] - normals @ vertices.T
        margin = 1e-9 * (np.abs(offsets)[:, None] + np.linalg.norm(normals, axis=1)[:, None] * np.abs(vertices).sum(axis=1))
        feasible = (slack >= -margin).all(axis=0)
        if not feasible.any():
            return keep
        return (np.abs(slack[:, feasible]) <= margin[:, feasible]).sum(axis=1) >= 2
