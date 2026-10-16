# Test problems, specification

`problems.py` generates reproducible discrete OT problems (X, mu, Y, nu) in R^2, in the style of Schmitzer 2016, Sect. 6.1, but on unstructured point sets.

## Two independent choices
Point placement and mass distribution are chosen independently, so geometric difficulty and mass difficulty can be varied separately.

**Points** `sample_points(n, kind, rng)`:
- `uniform`: uniform in the unit square.
- `clustered`: Gaussian mixture (random centres and covariances), clipped to the unit square.
- `grid`: regular sqrt(n) x sqrt(n) grid, for comparison with Schmitzer.

**Masses** `gaussian_masses(points, rng, n_gauss, ...)`:
- Density = sum of random Gaussians (random centres, random weights, covariance eigenvalues spanning a range as in Schmitzer, there 1.8 to about 100 in pixel units; here relative to the mean point spacing).
- Optional discontinuous mask (e.g. a random half-plane or disc set to a small value).
- A small constant is added so every point has positive mass.
- mass(x) = density(x), or density(x) * area(x) with `area_weighted=True` (Voronoi cell area, so that clustered points do not double count the density).

## Normalisation
mu and nu are each normalised to total mass 1 (in float64), and nu is rescaled once more by sum(mu) / sum(nu) so the sums agree exactly.

## Problem
`make_problem(n, points_kind, rng, **mass_kwargs) -> (X, mu, Y, nu)`: X and Y are sampled independently with the same settings, |X| = |Y| = n as in Schmitzer. Everything is driven by a `numpy.random.Generator` seed for reproducibility.

## Tests (unittest)
- Masses are positive and both sums are equal.
- The same seed gives the same problem.
- Point sets lie in the unit square and contain no duplicate points (Delaunay needs this).
