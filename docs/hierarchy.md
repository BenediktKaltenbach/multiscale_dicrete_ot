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
    nbrs: list[tuple]         # 1-ring of each level as CSR (indptr, indices); None where no mesh is needed
```
`children(k, i)` is derived from `parent[k-1]`. `rep_points(k)` returns `points[rep[k]]`.

## Builder
`agglomerate(points, mass, n_dense) -> Hierarchy` for a single set.
`build_pair(X, mu, Y, nu, n_dense) -> (HX, HY)` coarsens both sets with the same number of steps, so both have depth K.

### Level 0
- `rep[0] = arange(n)`, `rad[0] = 0`, `mass[0] = mass`, `cell_of[0] = arange(n)`.
- Mesh: `scipy.spatial.Delaunay(points)`; the 1-ring comes from `vertex_neighbor_vertices`.

### One coarsening step k -> k+1
1. **Matching, round 1:** sort the mesh edges of level k by length (distance between reps). Go through them; pair two cells if both are still unmatched. Unmatched cells stay single.
2. **Matching, round 2:** the same on the groups from round 1. Groups are adjacent if any mesh edge connects them; the edge weight is the shortest such edge. This gives groups of 1 to 4 cells.
3. **New cells:** each group is a cell of level k+1; set `parent[k]` and `cell_of[k+1] = parent[k][cell_of[k]]`.
4. **mass** via `np.bincount(parent[k], mass[k])`.
5. **rep:** compute the mass centroid of each new cell over its points (bincount over `cell_of[k+1]`). rep = the child rep closest to that centroid. This keeps reps nested.
6. **rad:** max over the cell's points of |rep - x|, via `np.maximum.at` over `cell_of[k+1]`. This is exact for (4.1).
7. **Mesh:** Delaunay of the new rep points. Skip it once the level has at most n_dense cells (no shielding needed there) or fewer than 3 non-collinear points.

The matching is a plain Python loop over the edges (about 3 n_k per round). That is fast enough to start; use numba only if needed.

### Stopping and equal depth
- Coarsen X and Y in lockstep until both have at most n_dense cells. If one set reaches a single cell earlier, its further steps are copy steps (each cell is its own parent).
- On top, add level K = {X}. Alg. 4.6 then starts with the trivial problem at K; its refinement is the dense problem at K-1, exactly as in Schmitzer.

## Invariants (tests)
- Each level partitions the points: `cell_of[k]` is defined for all points, and `parent[k][cell_of[k]] == cell_of[k+1]`.
- Mass is conserved on every level.
- (4.1) holds; rep[k][i] is a point of cell i.
- Level K has one cell; X and Y have the same K.

## Quality measures (first experiment)
- Distribution of the number of children per cell (target about 4; fraction of copy cells).
- rad shrinking per level: median of rad_{k+1} / rad_k (target about 2).
- Compactness: rad relative to the mean nearest-neighbour distance of the reps at that level.
- Test on uniform, clustered (Gaussian mixture) and anisotropic samples.

## Open points
- Edge length in round 2: shortest connecting edge vs. distance between group centroids.
- Value of n_dense (initially about 256).
