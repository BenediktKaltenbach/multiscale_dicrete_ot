# Sketch: candidate pruning in the $c$-exponential chart + exact leaf filter

Status: implemented on branch `mtw-pruning` as sketched (`cost.FacePruning`, `cost.Multiquadratic`, `shielding.filter_missed`), tests in `tests/test_mtw.py`, results in `rationale.md`, Section 4b. The text below is the original proposal.

## Idea

- Search (all levels) stays in $y$-space with `psi_hat` (Lipschitz bound). Nothing in the hierarchy moves to $p$-coordinates.
- Pruning: per $x_A$, map the candidates to the chart $p=\kappa_{x_A}(y)=-\nabla_x c(x_A,y)$ and keep only those whose tangent half-plane
  $$H_s=\{p:\ \langle n_s,\,p-p_s\rangle\le 0\},\qquad p_s=\kappa_{x_A}(y_s),\quad n_s=\nabla f_s(p_s),$$
  is a face of the polygon $\bigcap_s H_s$ (Option 1). This reuses `_faces`.
- Exact leaf filter: missed cells of the pruned search are checked exactly against the dropped candidates $S_{\mathrm{drop}}$:
  $$\hat Y=\hat Y'\cap\{y:\ g_s(y)\le g_s(y_s)\ \ \forall s\in S_{\mathrm{drop}}\},$$
  where $\hat Y'$ is the result of the search with $S_{\mathrm{keep}}$. The result is exactly $\hat Y$ of the search with all candidates, independent of how good the pruning is.
- Step-ii keeps using the full $S$ (unchanged).

Correctness never depends on the pruning. MTW only makes the pruning a good heuristic ($U_s\subseteq H_s$).

## Change 1: `shielding.py`

New helpers:

```python
def _subset(owner, indices, mask, n):
    """Compressed sparse row (CSR) structure (index_pointer, indices) of the candidate entries selected by mask."""
    return np.concatenate([[0], np.cumsum(np.bincount(owner[mask], minlength=n))]), indices[mask]


def filter_missed(pair_x, pair_cell, X, D, yd, Y, cost, tolerance):
    """Removes missed pairs (x_A, y) shielded by a dropped candidate. D is CSR of the dropped candidates, yd their partners."""
    index_pointer, indices = D
    counts = np.diff(index_pointer)
    pair_of = np.repeat(np.arange(len(pair_x)), counts[pair_x])
    entry = _ranges(index_pointer[pair_x], counts[pair_x])
    xA, xs = X[pair_x[pair_of]], X[indices[entry]]
    shielded = cost.psi(xA, xs, Y[pair_cell[pair_of]]) > cost.psi(xA, xs, yd[entry]) + tolerance
    hit = np.bincount(pair_of[shielded], minlength=len(pair_x)) > 0
    return pair_x[~hit], pair_cell[~hit]
```

In `shield` (only the pruning block and the loop change):

```python
    search, dropped = S, None
    prune = getattr(cost, "needed_candidates_all", None)
    if prune is not None:
        needed = prune(X, S, Y[t[indices]], tolerance)
        if not needed.all():
            search = _subset(owner, indices, needed, len(X))
            dropped = _subset(owner, indices, ~needed, len(X))
    rows, columns = [pi.row, owner], [pi.col, t[indices]]
    for start in range(0, len(X), chunk):
        sources = np.arange(start, min(start + chunk, len(X)))
        missed_x, missed_y = search_all(X, search, Y[t[search[1]]], Yh, k, cost, tolerance, sources)
        if dropped is not None:
            missed_x, missed_y = filter_missed(missed_x, missed_y, X, dropped, Y[t[dropped[1]]], Y, cost, tolerance)
        rows.append(missed_x)
        columns.append(missed_y)
```

`search_all` and `search_tree` stay unchanged.

## Change 2: `cost.py`, make the face detection cost-independent

Move `needed_candidates`, `needed_candidates_all` and `_faces` into a mixin. The only cost-specific part is the half-plane data per candidate entry, provided by `_face_data(xA, xs, ys, tolerance) -> (normals, offsets)`, describing half-planes
$$\{p:\ \langle \mathrm{normal},\,p\rangle\le\mathrm{offset}\}.$$

```python
class FacePruning:
    prune = False
    # needed_candidates: unchanged
    # needed_candidates_all: unchanged except the call
    #     needed[entries] = _faces(counts[group], row, slot,
    #                              *self._face_data(X[group][row], xs[entries], ys[entries], tolerance))


def _faces(counts, row, slot, entry_normals, entry_offsets):
    groups, width = len(counts), max(int(counts.max()), 1)
    valid = np.zeros((groups, width), bool)
    valid[row, slot] = True
    normals = np.zeros((groups, width, 2))
    normals[row, slot] = entry_normals
    offsets = np.zeros((groups, width))
    offsets[row, slot] = entry_offsets
    angles = np.full((groups, width), 4 * np.pi)
    angles[row, slot] = np.arctan2(entry_normals[:, 1], entry_normals[:, 0])
    # rest of the old _faces body unchanged
```

`SquaredEuclidean(FacePruning)` then only adds the Euclidean half-planes $\langle x_s-x_A,\,y\rangle\le\langle x_s-x_A,\,y_s\rangle$:

```python
    def _face_data(self, xA, xs, ys, tolerance):
        normals = xs - xA
        return normals, np.einsum("ij,ij->i", normals, ys) + tolerance / 2
```

which reproduces the current behaviour exactly.

## Change 3: first MTW cost, $c(x,y)=\sqrt{1+|x-y|^2}$ (satisfies A3s)

Closed forms with $d=x-y$ and $w=\sqrt{1+|d|^2}$:

$$\nabla_x c=\frac{d}{w},\qquad \nabla_y c=-\frac{d}{w},$$

$$D^2_{xy}c=-\frac1w\Big(I-\frac{dd^T}{w^2}\Big),\qquad\text{eigenvalues } -\frac1w,\ -\frac1{w^3},\qquad\text{so}\quad\|D^2_{xy}c\|\le1\ \text{(global, rigorous)},$$

$$\big(D^2_{xy}c\big)^{-1}=-w\,\big(I+dd^T\big).$$

With $d_A=x_A-y_s$, $d_s=x_s-y_s$, $w_A=c(x_A,y_s)$, $w_s=c(x_s,y_s)$:

$$p_s=\kappa_{x_A}(y_s)=\frac{y_s-x_A}{w_A},\qquad n_s=-\big(D^2_{xy}c(x_A,y_s)\big)^{-T}\big(\nabla_y c(x_A,y_s)-\nabla_y c(x_s,y_s)\big)=w_A\big(I+d_Ad_A^T\big)\Big(\frac{d_s}{w_s}-\frac{d_A}{w_A}\Big).$$

The hierarchical lower bound uses $\|D^2_{xy}c\|\le1$:
$$\hat g_s(\hat y)=g_s(\mathrm{rep}(\hat y))-|x_s-x_A|\,\mathrm{rad}(\hat y).$$

```python
class Multiquadratic(FacePruning):
    def __init__(self, prune=False):
        self.prune = prune

    def c(self, x, y):
        return np.sqrt(1 + np.sum((x - y) ** 2, axis=-1))

    def psi(self, xA, xs, y):
        return self.c(xA, y) - self.c(xs, y)

    def psi_hat(self, xA, xs, rep_y, rad_y):
        """Lipschitz bound with ||D_xy c|| <= 1."""
        return self.psi(xA, xs, rep_y) - np.linalg.norm(xs - xA, axis=-1) * rad_y

    def _face_data(self, xA, xs, ys, tolerance):
        """Tangent half-planes {<n_s, p - p_s> <= 0} in the chart p = -grad_x c(x_A, .)."""
        dA, ds = xA - ys, xs - ys
        wA, ws = self.c(xA, ys)[:, None], self.c(xs, ys)[:, None]
        v = ds / ws - dA / wA
        normals = wA * (v + dA * np.einsum("ij,ij->i", dA, v)[:, None])
        return normals, np.einsum("ij,ij->i", normals, -dA / wA) + tolerance / 2
```

## Tests (unittest)

1. Euclidean, `prune=True`: result of `shield` with filter equals the result without pruning (filter removes nothing, `_faces` is exact).
2. `Multiquadratic`, `prune=True` vs `prune=False`: identical $N$ (the filter makes it exact).
3. `Multiquadratic`: `missed_brute` equals `search_all` without pruning (validates `psi_hat`).
4. `Multiquadratic`: $n_s$ vs finite differences of $f_s=g_s\circ\exp^c_{x_A}$ at $p_s$ (already checked once by hand).

## Cost estimate

- Pruning: per $x_A$ and scale, $O(|S|)$ chart points and normals (closed form, a few flops) plus $O(|S|^2)$ for the polygon, as in the Euclidean case.
- Search: unchanged structure; only $c$ is a bit more expensive (square root). The bound $\|D^2_{xy}c\|\le1$ is global, so `psi_hat` is as sharp as a global Lipschitz bound allows.
- Filter: $|\hat Y'|\cdot|S_{\mathrm{drop}}|$ evaluations of $g_s$ per $x_A$, both factors $O(1)$. Negligible compared to the search.

## Open points

- Branch vs fork: suggested is a branch (e.g. `mtw-pruning`) in the same repo.
- `tolerance / 2` in the chart offsets: the scale of $p$ differs from $y$; probably harmless, check.
- Measure: number of dropped candidates, number of filtered pairs ($=$ wrong drops of Option 1), search time with and without pruning.
