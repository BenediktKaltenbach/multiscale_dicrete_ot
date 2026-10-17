# Design rationale

Why the project looks the way it does. `architecture.md` and the component specs say *what* is built; this file says *why*, with the evidence so far. Checked against the code on 2026-10-17 (commit after the HiGHS warm start).

## 1. Goal and setting
- Goal: construct shielding neighbourhoods (Schmitzer 2016) on unstructured planar point sets for MTW-type costs, with local "fences" around the target y = t(x_A), and prove that this is possible.
- Setting: discreteâ€“discrete OT between finite X, Y in R^2 (random tessellations, not grids). Schmitzer tested almost exclusively on Cartesian grids with a 4-neighbourhood; unstructured point sets in R^2 are essentially untested, and he reports that the variant without step-iii gets stuck on such point clouds. So step-iii (search for Y_hat) is always on.
- Squared Euclidean cost first, as the validated reference; MTW costs later through the same interfaces.

## 2. Why the shielding construction carries over to MTW costs
- Correctness of Alg. 5.1/5.2 holds for any cost: a shielding neighbourhood certifies global optimality (Cor. 3.10) independently of the cost. MTW only affects how sparse N becomes.
- Hence the core (shielding.py, multiscale.py, solvers) only calls `c`, `psi`, `psi_hat`. Everything Euclidean (bound (5.7), half-plane fences, candidate pruning) lives in `SquaredEuclidean`.
- MTW enters twice: (a) by Loeper, the unshielded set of a candidate is convex in the c-exponential chart at x_A, so fences stay convex there; (b) regularity of the optimal map, which is needed for the Lipschitz assumption in the sparsity bound (Prop. 5.3).

## 3. Hierarchy
**Option B (mesh-based, bottom-up) as the main hierarchy; option A (quadtree, Def. 4.4) as the baseline.**
- Our data is given at the finest level, so building bottom-up from the Delaunay mesh of the points is natural, deterministic and keeps a neighbour structure on every level (needed for S at coarse levels). k-means / top-down splitting was rejected for being random and ignoring the mesh, not for cost.
- The quadtree has a fixed cell geometry and uniform depth. On clustered data its fan-out is very unbalanced (up to 21 children at level 1 for 250 points), and `refine` builds children(x) x children(y), so the initial N1 explodes: clustered, n = 1000, |N1| / n = 28.7 at level 0 for the quadtree vs. 7.4 for the mesh (`hierarchy_measures.txt`). Both reach the dense optimum.
- **Coarse cost (4.2):** c_k(x, y) = c(rep x, rep y), as in Schmitzer, so every level is an ordinary OT problem and the shielding geometry applies unchanged on every level (Remark 4.8).
- **rep is an existing point** (the child rep closest to the mass centroid), not a new point as Schmitzer's cube centre. Reasons: for MTW costs the cost is only defined on Omega x Lambda (c-convex domains, which need not be convex in the plane), so a centroid could leave the domain; nested reps are also easier to debug. Cost: slightly larger rad. rad is computed exactly over the leaves, so (4.1) holds.
- **Matching:** two rounds of greedy matching along the shortest mesh edges give about 4 children per cell, close to a quadtree, and so comparable with Schmitzer.
- **Edge length cap (default 3.0) and attachment of leftovers:** without the cap, greedy matching pairs cells across gaps once nearby partners are taken, giving elongated cells (compactness up to 26 at level 1, uniform n = 4096) and weak bounds psi_hat. With the cap, max compactness drops to about 4, copy cells disappear (leftovers join a neighbour, at most 5 children), and the optimal cost is unchanged (`edge_cap.txt`).
- Compactness is measured per cell, rad / distance from its rep to the nearest other rep at that level, because a global mean spacing misjudges clustered data.

## 4. Candidate sets S(x_A)
- What matters is the cone constant q from Prop. 5.3 (5.6): the larger the angular gaps between candidate directions, the larger the fence P and Y_hat. The bound gives Y_hat inside a ball of radius about L D / q.
- **Decision (2026-10-17): Delaunay 1-ring as the default**, sector neighbours (`mesh.sector_neighbours`, nearest point per angular sector among the 32 nearest; filled sectors give q >= cos(2 pi / sectors)) as an option via `candidates=`. Measured at n = 4096 with warm start (`candidate_sets.txt`), both unpruned: run time is a tie (8 sectors 3–9% faster, within the noise of iteration counts); the 1-ring gives the sparser N on clustered (-25%) and anisotropic (-19%) points, 8 sectors on the grid (-11%), uniform equal. The 1-ring's low q tail (5% quantile 0.19–0.37) matters less in practice than the empty outward sectors at cluster boundaries (1.4–3.6% of vertices with q <= 0 for sectors vs. 0.4–0.6% for the 1-ring). 6 sectors are the weakest variant. Tried and removed from the code: keeping only the 6 closest of the 8 sector neighbours. Dropping two sectors leaves angular gaps up to 135° (q 5% quantile 0.09–0.16 instead of 0.40–0.50), so Y_hat grows; it gave the largest N on uniform points (10.9 per point vs. 8.75 for the 1-ring and 8 sectors) and no gain on clustered, anisotropic or grid points. Covering all directions matters more than closeness to x_A.
- **Pruning** (`SquaredEuclidean(prune=True)`, `needed_candidates`): a candidate whose half-plane is not a face of the polytope P leaves Y_hat unchanged. First version dropped it for the search *and* its step-ii entry: N 10–25% smaller, but 25–40% more solve_sparse iterations (the step-ii entries let mass move between neighbouring targets and speed up convergence) and twice the shield time. Hence now (fix 2) the redundant candidates are skipped only in the search; step-ii keeps the entries of all candidates, so N and the iterations equal the unpruned run (measured). The remaining saving is search work, which does not pay off yet: shield time +40–45% because the check costs more than the search it saves; it can only pay off once the pruning check itself is vectorised over all x_A (now one small numpy call sequence per x_A, dominated by call overhead). Off by default.
- Boundary vertices have q <= 0 (P unbounded); this is accepted, as in Schmitzer (Remark 5.6).
- Later: adaptive 2-ring where 0 < q < q_min, as a comparison.

## 5. Shielding search
- `search_tree` is Alg. 5.2 done level by level with numpy: all cells of a level are tested against all candidates at once, then the surviving cells are expanded. It visits the same cells as the recursive version but avoids millions of Python calls. Measured: 25.7x fewer psi_hat evaluations than brute force (uniform, n = 2000). Since 2026-10-17 `shield` runs this for all x_A at once (`search_all`, frontier of (x_A, cell) pairs; the pruning check is batched the same way), in chunks of 4096 x_A because the peak memory is about 75 kB per x_A. A test checks that it gives exactly the N of the per-x_A loop. Shield time at n = 4096 dropped from 15–28 s to 6–13 s; with that, search pruning is roughly break-even (between -30% and +25% shield time depending on the point set), so it stays off by default.
- Strict inequality with tolerance: shielded only if psi_hat > psi(y_s) + tol, so rounding can only enlarge N, never make it wrong. Strictness is needed only for termination of Alg. 3.7 (Remark 3.9); ties occur systematically on grids, essentially never on random points.

## 6. Solvers
- Requirement: floating point (no integer truncation as LEMON needs, Schmitzer Remark 6.2), simplex only (interior-point solutions are not sparse), Python, open source.
- **HiGHS via highspy** is the default. CPLEX (Schmitzer's network simplex) would likely be faster but needs a licence; LEMON has no maintained Python binding; OR-Tools min-cost flow is integer-only and returns no duals. All can be added behind `LocalSolver` if profiling demands it.
- **spt pi = all basic entries, including degenerate zeros.** They satisfy alpha + beta = c (2.4), keep spt pi a spanning tree like a network simplex basis, and keep refined N1 feasible in floating point.
- **Warm start:** one persistent HiGHS model per `solve_sparse`; new entries of N are added as columns, only nonbasic columns outside N are deleted, primal simplex continues from the old basis. Measured: solve time 128 s -> 6.7 s (uniform, n = 4096), same cost to 1e-16.
- Dense reference: POT `ot.emd` (network simplex). Dense HiGHS takes 250â€“520 s at n = 1000, multiscale 3â€“5 s.

## 7. Test problems and setup
- `problems.py`: points (uniform, clustered, grid, anisotropic) and masses (sums of random Gaussians, optional mask, small constant) are chosen independently, so geometric and mass difficulty vary separately; following Schmitzer Sect. 6.1. mu and nu are normalised to exactly equal mass.
- Code in a package (`src/msot`), tests with unittest, experiments as scripts with fixed seeds writing results and plots to `experiments/results/`, notebooks only for viewing.
- Parallelism: `N_WORKERS` (default 4) for independent experiments and block-wise checks; not inside one solve until profiling shows a need.

## 8. Where code and docs currently diverge (to resolve)
- (Resolved) Default S: the 1-ring (Section 4).
- (Resolved) `shield` and the pruning check are vectorised over x_A (Section 5).
- `config.py` / `N_WORKERS` and block-wise `verify.py` are not implemented yet.
- `experiments/results/warm_start.txt` and `candidate_sets.txt`: the warm-start branch is merged into main; both experiments are being rerun on main with integer masses and warm start.
- **Integer mass atoms vs. Section 6/7:** Section 6 lists "floating point (no integer truncation as LEMON needs, Remark 6.2)" and Section 7 "mu and nu normalised to exactly equal mass". Since 2026-10-17 `make_problem` returns integer mass atoms by default (10^9 in total, `atoms=None` for floats) and `Highs` solves them exactly (basic solutions of the totally unimodular transport LP are integral; HiGHS values are rounded and the marginals checked exactly). The solver itself is still floating point and costs stay floats; only the masses are quantised (relative accuracy about 1e-9, as in Schmitzer's Remark 6.2). Reason: with float masses the refined N1 is only feasible up to the HiGHS primal tolerance, and on the masked grid at n = 4096 no tolerance worked (1e-10 too tight for HiGHS on that mass range, 1e-9 too loose for the refinement). To reconcile in Sections 6 and 7.
- `docs/hierarchy.md`, "Quality measures", still describes compactness with the global mean spacing.
- `pot_dense.py` (dense POT reference) is not there yet; dense references currently use HiGHS.

## 9. Open questions
- (Resolved 2026-10-17) Grid with disc mask, infeasible local problem at n = 4096: float masses with a wide range (cell masses 5e-4 to 30 after scaling) left no workable HiGHS primal tolerance. Fixed by integer mass atoms (Section 8); all masked-grid runs at n = 4096 (seeds 0, 1; mesh and quadtree; cold and warm) now solve, warm and cold with identical cost.
- Whether the adaptive 2-ring helps the 1-ring's low-q vertices.
- n_dense and length_cap defaults.
- (Measured 2026-10-17) 256 x 256, all 16 configurations (4 point kinds x mesh/quadtree x 1-ring/8 sectors, `experiments/results/large_65536_summary.txt`): 22–94 min each, identical optimal cost within each kind; level-0 solves are 75–97% of the time, so the inner solver (general LP simplex with LU factorisation instead of Schmitzer's network simplex) is the bottleneck. The quadtree is faster on uniform, clustered and grid points but needs 94 min on anisotropic points (N1 of 79 entries per cell at level 0, unbalanced fan-out); the mesh hierarchy is robust across kinds. Dual check max(alpha + beta - c) <= 1e-9 in 13 of 16 runs, but 1.6e-7 and 1.3e-6 in two anisotropic runs: pairs outside N are certified through short-cut chains (Prop. 3.2), along which the solver's dual tolerance can accumulate; to be stated as "optimal up to accumulated solver tolerance" or tightened.
- N jumps to 13–16 entries per point after the first coupling of a level on uniform points (final optimum about 8); candidate fix: a two-phase solve_sparse (local iterations with spt + step-ii only, then full shielding).
- A network-simplex backend (CPLEX network optimizer with basis copy, as Schmitzer) for timings comparable to his.
- MTW: cost-specific psi_hat, and the a priori regularity of t needed for Prop. 5.3.
