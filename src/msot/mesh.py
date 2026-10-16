import numpy as np
from scipy.spatial import Delaunay, cKDTree


def one_ring(points):
    index_pointer, indices = Delaunay(points).vertex_neighbor_vertices
    return index_pointer, indices


def sector_neighbours(points, sectors, candidates=32, offset=0.0, keep=None):
    """S(xA): the nearest point in each of `sectors` equal angular sectors, the first starting at angle `offset`.

    Only the `candidates` nearest points are searched, so far sectors (at the boundary) stay empty.
    keep: of these sector neighbours, only the `keep` closest to xA. Returns CSR (index_pointer, indices).
    """
    n = len(points)
    if n < 2:
        return np.zeros(n + 1, dtype=np.int64), np.zeros(0, dtype=np.int64)
    nearest = cKDTree(points).query(points, min(candidates + 1, n))[1][:, 1:]
    directions = points[nearest] - points[:, None]
    angles = np.mod(np.arctan2(directions[..., 1], directions[..., 0]) - offset, 2 * np.pi)
    sector = np.minimum((angles / (2 * np.pi / sectors)).astype(np.int64), sectors - 1)
    keys = np.arange(n)[:, None] * sectors + sector
    _, first = np.unique(keys.ravel(), return_index=True)
    vertex = first // nearest.shape[1]
    if keep is not None:
        first = np.sort(first)
        vertex = first // nearest.shape[1]
        rank = np.arange(len(first)) - np.searchsorted(vertex, vertex)
        first, vertex = first[rank < keep], vertex[rank < keep]
    index_pointer = np.concatenate([[0], np.cumsum(np.bincount(vertex, minlength=n))])
    return index_pointer, nearest.ravel()[first]


def cone_q(points, neighbours):
    """q (5.6) per vertex: cos of half the largest angular gap between the directions to its neighbours."""
    index_pointer, indices = neighbours
    q = np.empty(len(points))
    for vertex in range(len(points)):
        directions = points[indices[index_pointer[vertex]:index_pointer[vertex + 1]]] - points[vertex]
        angles = np.sort(np.arctan2(directions[:, 1], directions[:, 0]))
        gaps = np.diff(np.append(angles, angles[0] + 2 * np.pi))
        q[vertex] = np.cos(gaps.max() / 2)
    return q


def candidate_statistics(points, S):
    """q (5.6) and D = max |xs - xA| over S(xA), per vertex."""
    index_pointer, indices = S
    vertex = np.repeat(np.arange(len(points)), np.diff(index_pointer))
    D = np.zeros(len(points))
    np.maximum.at(D, vertex, np.linalg.norm(points[indices] - points[vertex], axis=1))
    return cone_q(points, S), D
