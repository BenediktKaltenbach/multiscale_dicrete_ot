import numpy as np
from scipy.spatial import Delaunay


def one_ring(points):
    index_pointer, indices = Delaunay(points).vertex_neighbor_vertices
    return index_pointer, indices


def cone_q(points, neighbours):
    """q (5.6) per vertex: cos of half the largest angular gap between 1-ring directions."""
    index_pointer, indices = neighbours
    q = np.empty(len(points))
    for vertex in range(len(points)):
        directions = points[indices[index_pointer[vertex]:index_pointer[vertex + 1]]] - points[vertex]
        angles = np.sort(np.arctan2(directions[:, 1], directions[:, 0]))
        gaps = np.diff(np.append(angles, angles[0] + 2 * np.pi))
        q[vertex] = np.cos(gaps.max() / 2)
    return q
