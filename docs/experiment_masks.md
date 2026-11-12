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
  - a disc of radius 0.06 with its centre uniform in [0.15, 0.85]^2.
- Drawn masks: mu: strip along the bottom edge (outward normal (0.098, -0.995), i.e. tilted by 5.6°), disc centre (0.477, 0.409); nu: strip along the right edge (outward normal (0.995, -0.101), tilted by -5.8°), disc centre (0.703, 0.784). Each strip covers about 10 % of the square, each disc 1.1 %.
- Inside a mask the mass is set to the floor of `gaussian_masses` (1e-3 times the mean density before the floor), outside it is unchanged; then both measures are quantised to 10^9 atoms as before. So the masked problem has the same points, the same masses outside the masks, and a nearly empty region in each measure.
- The hierarchies are built separately for both versions (the reps depend on the masses).
- Recorded additionally per run: the number of points in the masks of X and Y and the share of the unmasked mass that lies in them.

## Results
(to be filled in after the run)
