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
(to be filled in after the run)
