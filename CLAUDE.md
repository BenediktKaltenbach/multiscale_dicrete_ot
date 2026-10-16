# Working notes for Claude

- Read `docs/architecture.md` first; it is the agreed design. `docs/rationale.md` explains the reasons; keep its section "Where code and docs currently diverge" up to date when you resolve one of the points.
- Benedikt wants to be involved in every design decision: propose, wait for agreement, then implement in small steps for review.
- Nimble, easily readable code with few comments. Do not merge or flatten the agreed module structure.
- Write names out, no programming abbreviations (`index_pointer`, `neighbours`, `columns`, `tolerance`, not `indptr`, `nbrs`, `cols`, `tol`). Symbols from Schmitzer's notation stay short (`rep`, `rad`, `psi`, `pi`, `mu`, `xA`, `K`, `S`, `N`).
- Follow Schmitzer 2016 (JMIV, "A sparse multi-scale algorithm for dense optimal transport") in notation and naming.
- Design discussions also happen in a separate Cowork session. Context is exchanged via Markdown files in `docs/`.
