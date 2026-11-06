# Code flowcharts

Visual map of `src/msot` as of 2026-11-05 (branch `mtw-pruning`). Algorithm numbers refer to Schmitzer 2016. GitHub renders the diagrams directly.

## 1. Pipeline

```mermaid
flowchart LR
    P["problems.make_problem<br/>X, mu, Y, nu<br/>integer mass atoms"] --> H["hierarchy.build_pair<br/>HX, HY with equal depth K"]
    H --> M["multiscale.solve_multiscale<br/>Alg. 4.6"]
    C["cost.SquaredEuclidean or<br/>cost.Multiquadratic<br/>prune=False default"] --> M
    M --> R["pi, alpha, beta, info"]
    R --> V["verify.dual_violation<br/>alpha + beta ≤ c on X x Y"]
    R --> E["experiments/*.py<br/>results/*.txt, *.png"]
```

## 2. Modules and who calls whom

```mermaid
flowchart TB
    problems["problems.py<br/>sample_points, gaussian_masses,<br/>make_problem, quantize"]
    hierarchy["hierarchy.py<br/>Hierarchy, build_pair,<br/>build_pair_quadtree, quadtree"]
    mesh["mesh.py<br/>one_ring, sector_neighbours,<br/>cone_q, candidate_statistics"]
    cost["cost.py<br/>FacePruning: needed_candidates_all, _faces<br/>SquaredEuclidean: c, psi, psi_hat, _face_data<br/>Multiquadratic: c, psi, psi_hat, chart, _face_data"]
    shielding["shielding.py<br/>extract_map, shield, search_all, filter_missed,<br/>search_tree, missed_brute"]
    multiscale["multiscale.py<br/>solve_multiscale, refine, dilate, solve_sparse"]
    solvers["solvers/<br/>base.LocalSolver, base.solve_dense,<br/>cplex.Cplex"]
    verify["verify.py<br/>dual_violation, is_shielding"]

    hierarchy --> mesh
    multiscale --> shielding
    multiscale --> solvers
    multiscale -. "candidates= optional" .-> mesh
    shielding --> cost
    multiscale --> cost
    verify --> cost
```

## 3. solve_multiscale (Alg. 4.6)

```mermaid
flowchart TB
    A["pi = trivial coupling at level K<br/>one cell x one cell"] --> B{"k = K-1 down to 0"}
    B --> C["refine(pi, HX, HY, k)<br/>N1 = children(x) x children(y)<br/>for all (x, y) in spt pi"]
    C --> C2["dilate(N1, HY.neighbours[k])<br/>add (x, y2) for y2 in the 1-ring of y<br/>(dilation=True, default)"]
    C2 --> D["S on X_k<br/>default: 1-ring HX.neighbours[k]<br/>option: candidates(HX, k), e.g. sector neighbours"]
    D --> E["solve_sparse on X_k, Y_k<br/>Alg. 4.1"]
    E --> F["store k, cells, |N1|, history"]
    F --> B
    B -- "after k = 0" --> G["return pi, alpha, beta, info"]
```

## 4. solve_sparse (Alg. 4.1)

```mermaid
flowchart TB
    A["N = N1, state = None"] --> B["c = cost.c on the entries of N"]
    B --> C["solver.solve_local(N, c, mu, nu, state)<br/>pi, alpha, beta, state"]
    C --> D["shield(pi, S, X, Yh, k, cost)<br/>new N, Alg. 5.1"]
    D --> E["C = cost of pi"]
    E --> F{"new N contained in N<br/>or C not decreasing?"}
    F -- yes --> G["return pi, alpha, beta, N, history"]
    F -- no --> H["N = new N"] --> B
```

## 5. Cplex.solve_local and the warm start

```mermaid
flowchart TB
    A["masses integral?<br/>yes: scale 1, round values<br/>no: scale to mean mass 1"] --> B{"warm_start and state<br/>of the same shape?"}
    B -- no --> C["build: new CPLEX model<br/>one arc x -> y per entry of N<br/>network optimizer"]
    B -- yes --> D["update_columns:<br/>delete nonbasic columns outside N,<br/>add the new entries of N as columns"]
    D --> E["continue from the current basis"]
    C --> F["run"]
    E --> F
    F --> G["pi = all basic columns<br/>incl. degenerate zeros"]
    G --> H{"integral?"}
    H -- yes --> I["check marginals exactly"]
    H -- no --> J["alpha, beta from row duals"]
    I --> J
    J --> K["return pi, alpha, beta, state = model + keys"]
```

## 6. shield (Alg. 5.1)

```mermaid
flowchart TB
    A["t = extract_map(pi)<br/>step-i: t(x) = argmax_y pi(x, y)"] --> B["entries: spt pi<br/>plus (x_A, t(x_s)) for all candidates, step-ii"]
    B --> C{"cost.prune?"}
    C -- yes --> D["needed_candidates_all:<br/>keep candidates whose half-plane is a face<br/>of the polygon of x_A (Fig. 3)<br/>SquaredEuclidean: exact half-planes (5.5) in y<br/>Multiquadratic: tangent half-planes in the<br/>chart p = -grad_x c(x_A, y)"]
    C -- no --> E["search_all with all candidates<br/>in chunks of 4096 x_A<br/>step-iii, Alg. 5.2"]
    D --> F["search_all with the kept candidates<br/>in chunks of 4096 x_A<br/>step-iii, Alg. 5.2"]
    F --> L["filter_missed: drop missed (x_A, y)<br/>that a dropped candidate shields, exact psi"]
    L --> G["add (x_A, missed cell of Y_k)"]
    E --> G
    G --> H["unique_entries: new N"]
```

## 7. search_all: level-synchronous Alg. 5.2

```mermaid
flowchart TB
    A["frontier = pairs (x_A, root cell of Y_K)"] --> B{"level = K down to k"}
    B --> C["for every pair and every candidate x_s:<br/>psi_hat(x_A, x_s, rep, rad), rad = 0 at level k"]
    C --> D{"psi_hat > psi(x_A, x_s, y_s) + tolerance<br/>for some candidate?"}
    D -- "yes: shielded" --> X["drop pair"]
    D -- no --> E{"level > k?"}
    E -- yes --> F["replace by (x_A, child) for all children"]
    F --> B
    E -- no --> G["pair is in Y_hat(x_A)"]
```

## 8. Building the mesh hierarchy (build_pair)

```mermaid
flowchart TB
    A["level 0: points as cells<br/>rep = point, rad = 0, Delaunay 1-ring"] --> B{"more than n_dense cells<br/>in X or Y?"}
    B -- yes --> C["_sorted_edges: mesh edges by length"]
    C --> D["_passes_cap: length ≤ 3 x max(spacing a, spacing b)"]
    D --> E["_match round 1: pairs"]
    E --> F["_match round 2: groups of up to 4"]
    F --> G["_attach leftovers to neighbour group<br/>at most 5 children"]
    G --> H["_add_level: mass by bincount,<br/>rep = child rep nearest to mass centroid,<br/>rad = max distance over leaves,<br/>Delaunay of the new reps"]
    H --> B
    B -- no --> I["_add_top: level K = one cell"]
```

## 9. Costs and their bounds

```mermaid
flowchart LR
    F["FacePruning<br/>prune, needed_candidates_all,<br/>_faces (polygon faces per x_A)"]
    F --> SE["SquaredEuclidean<br/>c = |x - y|^2<br/>psi_hat = psi(rep) - 2 |x_s - x_A| rad"]
    F --> MQ["Multiquadratic, MTW (A3s)<br/>c = sqrt(1 + |x - y|^2)<br/>psi_hat = psi(rep) - |x_s - x_A| rad<br/>since ||D_xy c|| ≤ 1"]
    SE --> S1["_face_data: normal x_s - x_A<br/>exact half-plane in y"]
    MQ --> S2["_face_data: tangent half-plane at p_s<br/>in the chart p = -grad_x c(x_A, y)<br/>normal w_A (I + d_A d_A^T)(d_s / w_s - d_A / w_A)"]
```
