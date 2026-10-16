import numpy as np
from scipy.spatial import Delaunay


def one_ring(points):
    indptr, indices = Delaunay(points).vertex_neighbor_vertices
    return indptr, indices


def cone_q(points, nbrs):
    """q (5.6) per vertex: cos of half the largest angular gap between 1-ring directions."""
    indptr, indices = nbrs
    q = np.empty(len(points))
    for a in range(len(points)):
        d = points[indices[indptr[a]:indptr[a + 1]]] - points[a]
        ang = np.sort(np.arctan2(d[:, 1], d[:, 0]))
        gaps = np.diff(np.append(ang, ang[0] + 2 * np.pi))
        q[a] = np.cos(gaps.max() / 2)
    return q
