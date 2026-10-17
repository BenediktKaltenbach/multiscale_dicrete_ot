# Hierarchy builder (option B), specification

Mesh-based hierarchical partition in the sense of Schmitzer 2016, Def. 4.3, built bottom-up from the given points.

## Notation
- Level k = 0 holds the singletons (the given points); level K is the trivial partition {X}.
- A cell **x** in X_k is a set of points with `rep`, `rad`, `mass` and children in X_{k-1}.
- (4.1): |rep(**x**) - x| <= rad(**x**) for every point x in **x**.

## Data structure (`hierarchy.py`)
```python
@dataclass
class Hierarchy:
    points: np.ndarray        # (n, 2)
    mass: list[np.ndarray]    # mass[k][i] = mu(cell i of level k)
    rep: list[np.ndarray]     # rep[k][i] = index into points (reps are existing points)
    rad: list[np.ndarray]
    parent: list[np.ndarray]  # parent[k][i] = cell of level k+1 containing cell i of level k
    cell_of: list[np.ndarray] # cell_of[k][j] = cell of level k containing point j
    neighbours: list[tuple]   # 1-ring of each level as CSR (index_pointer, indices); empty for a single cell
```
`children(k, i)` is derived from `parent[k-1]`. `rep_points(k)` returns `points[rep[k]]`.

## Builder
`agglomerate(points, mass, n_dense=256, length_cap=3.0) -> Hierarchy` for a single set.
`build_pair(X, mu, Y, nu, n_dense=256, length_cap=3.0) -> (HX, HY)` coarsens both sets with the same number of steps, so both have depth K.
`length_cap=None` switches off the edge length cap and the attachment of leftovers (steps 0 and 3 below); this is the original matching, kept for comparison.

### Level 0
- `rep[0] = arange(n)`, `rad[0] = 0`, `mass[0] = mass`, `cell_of[0] = arange(n)`.
- Mesh: `scipy.spatial.Delaunay(points)`; the 1-ring comes from `vertex_neighbor_vertices`.

### One coarsening step k -> k+1
0. **Edge length cap:** let spacing[i] be the distance from rep i to its nearest other rep at level k (the shortest incident mesh edge, since the nearest-neighbour graph is contained in the Delaunay graph). An edge (a, b) may be used in the matching only if length(a, b) <= length_cap * max(spacing[a], spacing[b]). Taking the max lets an isolated outlier merge with its nearest neighbour, while long sliver edges along the flat boundary of the hull and late leftover pairings between normally spaced cells are refused. Without the cap, greedy matching never refuses an edge and produces a few strongly elongated cells (compactness up to about 26 at level 1, n = 4096, median about 0.7).
1. **Matching, round 1:** sort the allowed mesh edges of level k by length (distance between reps). Go through them; pair two cells if both are still unmatched. Unmatched cells stay single.
2. **Matching, round 2:** the same on the groups from round 1. Groups are adjacent if any allowed mesh edge connects them; the edge weight is the shortest such edge. This gives groups of 1 to 4 cells.
3. **Attach leftovers:** every group that is still a single cell joins the adjacent group (via an allowed mesh edge) with the shortest connecting edge whose group has at most 4 cells, so a cell has at most 5 children. Leftovers are processed in order of that edge length (a plain loop over the sorted allowed edges). If there is no such edge, the cell stays single (a copy cell).
4. **New cells:** each group is a cell of level k+1; set `parent[k]` and `cell_of[k+1] = parent[k][cell_of[k]]`.
5. **mass** via `np.bincount(parent[k], mass[k])`.
6. **rep:** compute the mass centroid of each new cell over its points (bincount over `cell_of[k+1]`). rep = the child rep closest to that centroid. This keeps reps nested.
7. **rad:** max over the cell's points of |rep - x|, via `np.maximum.at` over `cell_of[k+1]`. This is exact for (4.1).
8. **Mesh:** Delaunay of the new rep points on every level (S at that level needs it); a complete graph for fewer than 3 or collinear points, an empty adjacency for a single cell.

The matching and the attachment are plain Python loops over the edges (about 3 n_k per round). That is fast enough to start; use numba only if needed.

### Stopping and equal depth
- Coarsen X and Y in lockstep until both have at most n_dense cells. If one set reaches a single cell earlier, its further steps are copy steps (each cell is its own parent).
- On top, add level K = {X}. Alg. 4.6 then starts with the trivial problem at K; its refinement is the dense problem at K-1, exactly as in Schmitzer.

## Invariants (tests)
- Each level partitions the points: `cell_of[k]` is defined for all points, and `parent[k][cell_of[k]] == cell_of[k+1]`.
- Mass is conserved on every level.
- (4.1) holds; rep[k][i] is a point of cell i.
- Level K has one cell; X and Y have the same K.
- Every cell of a coarsening level has at most 5 children.

## Quality measures
- Distribution of the number of children per cell (target about 4; fraction of copy cells).
- rad shrinking per level: median of rad_{k+1} / rad_k (target about 2).
- Compactness per cell: rad / distance from its rep to the nearest other rep at that level (a global mean spacing misjudges clustered data); per level median, 90% quantile, max and number of cells above 2 (`experiments/hierarchy_measures.py`).
- Test on uniform, clustered (Gaussian mixture) and anisotropic samples.

## Open points
- Edge length in round 2: still the shortest connecting edge (alternative: distance between group centroids). The cap in round 2 is tested on that same edge.
- Value of length_cap (default 3.0); comparison of None, 2, 3, 4 in `edge_cap.py` (script removed, numbers kept in rationale.md), results in `edge_cap.txt` (from the removed script edge_cap.py).
- Value of n_dense (initially about 256).
