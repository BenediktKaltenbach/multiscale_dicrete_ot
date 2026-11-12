# Experiment: discontinuous masks

Thesis experiment, run 2026-11-12. Script `experiments/masks.py`, figures `experiments/plot_masks.py`.
Raw results in `experiments/results/masks/` (`runs.json` with the mask geometry, figures as PDF and 600 ppi PNG; not under version control).

## Question
How much harder do the problems get when the densities have sharp discontinuities? Schmitzer 2016 (Sect. 6.1) multiplied "some densities" of his Gaussian-mixture test problems with discontinuous masks but does not describe them; the masks here are our own choice.

## Setup
Identical to `docs/experiment_candidate_sets.md` (code and machine, problems with the same seeds and generators, mesh hierarchy, solver, measurements, verification), with these differences:
- Candidate set: Delaunay 1-ring (default), pruning off.
- Variable: each of the 9 problems (uniform, clustered, grid; seeds 0, 1, 2; n = 16384) is solved as it is (`unmasked`, identical to `make_problem(16384, kind, default_rng(seed))`, checked) and with masks (`masked`); order unmasked/masked for seeds 0 and 2, masked/unmasked for seed 1.
- Masks (`masks.draw_mask`): one mask for mu and a different one for nu, both drawn once from `default_rng(2026)` (mu first) and used unchanged in all 9 problems. Each mask is the union of
  - an edge strip: the half-plane beyond a line through the point at depth 0.1 inside the middle of a random edge of the unit square, with the line tilted by a uniform angle in [-20°, 20°];
  - two discs of radius 0.06 with centres uniform in [0.15, 0.85]^2 (they may overlap).
- Drawn masks: mu: strip along the bottom edge (outward normal (0.098, -0.995), i.e. tilted by 5.6°), disc centres (0.477, 0.409) and (0.398, 0.703); nu: strip along the right edge (outward normal (0.960, 0.279), tilted by 16.2°), disc centres (0.274, 0.607) and (0.359, 0.827). Each strip covers about 10 % of the square, each disc 1.1 %.
- Whether a mask removes mass depends on where the random Gaussian bumps of a problem lie; this is deliberate (fixed masks, same for all problems). Share of the unmasked mass inside the masks (mu / nu): grid 0.0 / 27.9 %, 2.7 / 0.0 %, 0.0 / 19.3 %; uniform 23.2 / 21.9 %, 0.0 / 19.0 %, 19.0 / 31.8 %; clustered 6.2 / 0.0 %, 1.8 / 0.0 %, 11.6 / 0.0 % (seeds 0, 1, 2).
- Total masses stay equal: after masking, mu and nu are each quantised to exactly 10^9 atoms, so the remaining mass is scaled up uniformly; the solver checks the integer marginals exactly in every solve.
- Inside a mask the mass is set to the floor of `gaussian_masses` (1e-3 times the mean density before the floor), outside it is unchanged; then both measures are quantised to 10^9 atoms as before. So the masked problem has the same points, the same mass ratios outside the masks (all scaled up by the same factor), and a nearly empty region in each measure.
- The hierarchies are built separately for both versions (the reps depend on the masses).
- Recorded additionally per run: the number of points in the masks of X and Y and the share of the unmasked mass that lies in them.

## Results
A first run on 2026-11-12 with one disc per mask ran on battery power (about 60 % slower than on mains) and was discarded; its files are kept in `experiments/results/masks_on_battery/`.

Valid run: 2026-11-12 on mains power (checked before the start: battery status "on mains", charging), commit `84f15a0`. The pair uniform seed 2 was disturbed (both runs about 50 % slower than the identical problem in `docs/experiment_candidate_sets.md`, in solve and shield time alike: 92.0 s and 91.0 s) and was rerun right after the full run, with identical work (same rounds per level, same cost); `runs.json` keeps the disturbed times under "first run, disturbed".

**Correctness and reproducibility.** All 18 runs are optimal (dual violation at most 8.8e-10, exactly 0 on the grid), all hierarchies have K = 5. Every unmasked run repeats its problem from the candidate-set experiment exactly (same rounds per level, same cost); its time is 0–17 % shorter (mean 7 %), which shows the timing difference between two sessions.

Means over the 3 seeds (single seeds in brackets; times in s):

| kind | problem | total time | solve | shield | iterations at level 0 | all iterations | \|N\| / \|X\| at level 0 |
|---|---|---|---|---|---|---|---|
| grid | without masks | 60.5 (46.0, 46.1, 89.5) | 18.9 | 40.7 | 10, 9, 23 | 32, 31, 47 | 9.8, 11.8, 13.2 |
| grid | with masks | 73.2 (76.3, 53.8, 89.4) | 21.9 | 50.1 | 19, 11, 22 | 44, 32, 46 | 9.7, 11.4, 12.8 |
| uniform | without masks | 52.3 (55.5, 43.9, 57.7) | 20.2 | 31.3 | 8, 8, 9 | 35, 28, 38 | 14.8, 10.7, 9.7 |
| uniform | with masks | 95.1 (146.4, 76.4, 62.6) | 32.9 | 60.5 | 25, 11, 10 | 50, 35, 41 | 16.9, 8.3, 9.9 |
| clustered | without masks | 53.5 (55.3, 50.9, 54.3) | 18.3 | 34.4 | 12, 11, 9 | 38, 33, 31 | 11.2, 6.9, 10.7 |
| clustered | with masks | 54.9 (49.7, 49.6, 65.3) | 17.8 | 36.4 | 10, 12, 12 | 55, 36, 44 | 11.4, 6.6, 11.2 |

Per problem, time with / without masks and the share of the unmasked mass inside the masks (mu / nu):

| kind | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| grid | 1.66 (0 / 28 %) | 1.17 (3 / 0 %) | 1.00 (0 / 19 %) |
| uniform | 2.64 (23 / 22 %) | 1.74 (0 / 19 %) | 1.09 (19 / 32 %) |
| clustered | 0.90 (6 / 0 %) | 0.97 (2 / 0 %) | 1.20 (12 / 0 %) |

**Findings.**
- The masks make the problems harder on average, mainly through more rounds of Alg. 4.1 at level 0, not through larger neighbourhoods: |N| / |X| at level 0 changes by -22 % to +14 % (mean about -1 %), while the rounds at level 0 grow up to threefold (uniform seed 0: 8 -> 25, grid seed 0: 10 -> 19). The local problems keep their size; the multiscale initialisation is just further from the fine optimum, as expected when the coarse levels average over a sharp edge.
- Mean slowdown: uniform 1.82x, grid 1.21x, clustered 1.03x; per problem from 0.90x to 2.64x.
- The slowdown is not proportional to the removed mass: uniform seed 2 loses 19 % of mu and 32 % of nu and becomes only 9 % slower, uniform seed 0 loses about 22 % of each and becomes 2.6x slower; grid seed 2 loses 19 % of nu without any slowdown. What matters is how the cut changes the transport (whether mass must be moved across or around the cut over long distances), not how much mass it removes.
- On the clustered problems the masks of nu remove no mass in any seed and those of mu at most 12 %, so the clustered problems barely change (0.90x to 1.20x; seed 0 needs more rounds on the coarse levels but fewer at level 0).
- The masked problems still converge with neighbourhoods of the same size as the smooth ones, which is consistent with Schmitzer's observation (Fig. 7) that |N_x| rises only locally near discontinuities of the map (the locality itself was not measured here).

Figures (in `experiments/results/masks/`): `masks_problem` (masked mu and nu of grid seed 0 with the mask outlines), `masks_time` (total time), `masks_level0` (iterations and |N| / |X| at level 0).
