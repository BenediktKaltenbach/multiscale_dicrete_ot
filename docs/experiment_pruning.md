# Experiment: effect of candidate pruning on the run time

Thesis experiment, run 2026-11-12, directly after `docs/experiment_candidate_sets.md`. Script `experiments/pruning.py`, figures `experiments/plot_pruning_runtime.py`.
Raw results in `experiments/results/pruning/` (`runs.json`, figures as PDF and 600 ppi PNG; not under version control).

## Question
Pruning (`cost.FacePruning`, Fig. 3 in Schmitzer 2016) drops the candidates x_s whose half-plane {y : <x_s - x_A, y - t(x_s)> <= 0} (5.5) is no face of the polygon P(x_A); they cannot change Y_hat(x_A). Does skipping them in the search pay for the face test, on the three point kinds?

## Setup
Identical to `docs/experiment_candidate_sets.md` (code and machine, problems with the same seeds and generators, mesh hierarchy, solver, measurements, run protocol, verification), with these differences:
- Candidate set S(x_A): Delaunay 1-ring at every level (the default), no other variant.
- Variable: `SquaredEuclidean(prune=False)` vs. `SquaredEuclidean(prune=True)`; order off/on for seeds 0 and 2, on/off for seed 1.
- With pruning on, `shield` computes per level and round the face mask over all x_A (`needed_candidates_all`: half-planes with normal x_s - x_A and offset <x_s - x_A, t(x_s)> + tolerance / 2, faces of their polygon found per x_A in chunks of 1024 points). The search (step iii) uses only the kept candidates; step ii still adds (x_A, t(x_s)) for all candidates, so N and the iterations equal the run without pruning. The missed cells of the search are then checked exactly against the dropped candidates (`shielding.filter_missed`), so Y_hat equals the unpruned search in every case.
- Recorded additionally (with pruning on): for every call of the face test the level size, the number of candidates and the number kept; for every call of `filter_missed` the number of missed pairs and how many it removed. Counting is done by thin wrappers in the experiment script (`CountedSquaredEuclidean`, `counted_filter_missed`), whose overhead is a few list appends per round.
- Shield time includes the face test and `filter_missed`.
- Expected outcome from earlier measurements (`docs/rationale.md` §4 and §4b, uniform and masked grid only): same cost and iterations, shield time within about ±15 %.

## Results
Run 2026-11-12 at commit `5ce45de` (the scripts of this experiment are from `9dcf9e8`; no uncommitted changes in `src/`), directly after the candidate-set runs, same machine state.

**Correctness.** Per problem, pruning on and off give the same optimal cost (difference exactly 0) and the identical sequence of |N| in every round of every level, hence the same iterations (grid 32, 31, 47; uniform 35, 28, 38; clustered 38, 33, 31). Dual violation at most 8.9e-10 in all 18 runs. `filter_missed` checked 8.9 M (grid), 5.6 M (uniform) and 5.8 M (clustered) missed pairs against the dropped candidates and removed **none**: the face test never dropped a needed candidate.

Means over the 3 seeds (single seeds in brackets; times in s):

| kind | pruning | total time | solve | shield | dropped candidates (all levels and rounds) |
|---|---|---|---|---|---|
| grid | off | 66.0 (50.0, 53.2, 94.8) | 21.9 | 43.0 (31.1, 30.3, 67.6) | – |
| grid | on | 62.8 (48.7, 47.5, 92.1) | 19.5 | 42.2 (31.9, 29.1, 65.6) | 7.6, 10.3, 8.5 % |
| uniform | off | 52.5 (52.7, 45.7, 59.0) | 20.4 | 31.4 (31.7, 26.2, 36.2) | – |
| uniform | on | 54.4 (55.3, 48.9, 59.1) | 20.1 | 33.6 (34.3, 29.0, 37.4) | 12.0, 16.4, 15.2 % |
| clustered | off | 51.4 (55.3, 45.2, 53.6) | 17.2 | 33.5 (36.4, 31.9, 32.2) | – |
| clustered | on | 54.8 (59.7, 49.7, 54.8) | 18.2 | 35.9 (40.7, 34.1, 32.8) | 4.7, 6.8, 13.9 % |

Shield time with / without pruning per seed: grid 1.03, 0.96, 0.97; uniform 1.08, 1.11, 1.04; clustered 1.12, 1.07, 1.02.

Dropped share per level (mean over seeds, figure `pruning_dropped`): uniform 13–15 % on all levels, clustered 8–9 % on all levels, grid falling from 15 % at level 4 to 8 % at level 0.

**Findings.**
- Pruning is exact in practice: identical N, iterations and cost, and the exact leaf filter never had to correct the search.
- It does not save time. On random points the shield time grows by 2–12 % (mean +7 %) in every seed: the face test costs more than the search over the 5–16 % dropped candidates saves. On the grid it is neutral (0.96–1.03).
- The total time follows the shield time but is noisier: the solve time differs by up to 11 % between the two runs of a problem although CPLEX solves exactly the same sequence of local problems (e.g. grid 21.9 s vs. 19.5 s on average). Differences in total time below about ±10 % are therefore not significant at 3 seeds.
- With the Delaunay 1-ring only a small share of the candidates is redundant (the 1-ring of a planar mesh has 6 points on average, and most of them are faces of P(x_A)), so there is little to prune. Pruning stays off by default. This agrees with the earlier measurements (`docs/rationale.md` §4b) and with the standalone check in higher dimensions, where the redundant share rises to 31–42 % for d = 3 to 5.

Figures (in `experiments/results/pruning/`): `pruning_time` (total and shield time), `pruning_ratio` (shield time on / off per seed), `pruning_dropped` (dropped share per level).
