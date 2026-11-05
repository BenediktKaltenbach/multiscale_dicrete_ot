import numpy as np
from scipy.spatial import Delaunay, cKDTree


def one_ring(points):
    index_pointer, indices = Delaunay(points).vertex_neighbor_vertices
    return index_pointer, indices


def sector_neighbours(points, num_sectors, num_candidates=32, offset=0.0):
    """ONLY for sectored shielding (if 1-ring is used not needed).
    S(xA): the nearest point in each of `sectors` equal angular sectors, the first starting at angle `offset`.

    Only the `candidates` nearest points are searched, so far sectors (at the boundary) stay empty.
    Returns CSR (index_pointer, indices).
    """
    n = len(points)
    if n < 2:
        return np.zeros(n + 1, dtype=np.int64), np.zeros(0, dtype=np.int64)
    nearest = cKDTree(points).query(points, min(num_candidates + 1, n))[1][:, 1:]  # [1] takes distances of the nearest, [:, 1:] drops the point itself
    directions = points[nearest] - points[:, None] #x_s-X_A via broadcasting per candidate
    angles = np.mod(np.arctan2(directions[..., 1], directions[..., 0]) - offset, 2 * np.pi)
    sector = np.minimum((angles / (2 * np.pi / num_sectors)).astype(np.int64), num_sectors - 1)
    keys = np.arange(n)[:, None] * num_sectors + sector
    _, first = np.unique(keys.ravel(), return_index=True) # for each key return index of first occurence
    vertex = first // nearest.shape[1] # get index of the vertex (row) for each first occurence (nearest.shape[1] = number of candidates per vertex)
    index_pointer = np.concatenate([[0], np.cumsum(np.bincount(vertex, minlength=n))])
    return index_pointer, nearest.ravel()[first]


def cone_q(points, neighbours):
    """
    Used in candidate_statistics below to compute
    q (5.6) per vertex: cos of half the largest angular gap between the directions to its neighbours.
    """
    index_pointer, indices = neighbours
    q = np.empty(len(points))
    for vertex in range(len(points)):
        directions = points[indices[index_pointer[vertex]:index_pointer[vertex + 1]]] - points[vertex]
        angles = np.sort(np.arctan2(directions[:, 1], directions[:, 0]))
        gaps = np.diff(np.append(angles, angles[0] + 2 * np.pi)) # the appending is only needed for last gap
        q[vertex] = np.cos(gaps.max() / 2)
    return q


def candidate_statistics(points, S):
    """q (via cone_q) and D = max |xs - xA| over S(xA), per vertex."""
    index_pointer, indices = S
    vertex = np.repeat(np.arange(len(points)), np.diff(index_pointer)) # repeat each vertex index according to the number of its neighbors (in S - makes sense for 1-ring)
    D = np.zeros(len(points))
    np.maximum.at(D, vertex, np.linalg.norm(points[indices] - points[vertex], axis=1)) # build D[vertex] = max(D[vertex], |xs - xA|) iterating over all xs in S(xA)
    return cone_q(points, S), D
