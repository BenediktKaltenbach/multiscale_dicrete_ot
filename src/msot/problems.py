import numpy as np


def sample_points(n, kind, rng):
    if kind == "uniform":
        return rng.random((n, 2))
    if kind == "grid":
        side = int(round(np.sqrt(n)))
        coordinates = (np.arange(side) + 0.5) / side
        return np.stack(np.meshgrid(coordinates, coordinates), axis=-1).reshape(-1, 2)
    if kind == "clustered":
        return _gaussian_mixture(n, rng)
    raise ValueError(kind)


def _gaussian_mixture(n, rng, n_clusters=4, standard_deviation_range=(0.03, 0.15)):
    centres = rng.uniform(0.15, 0.85, (n_clusters, 2))
    covariances = [_random_covariance(rng, np.square(rng.uniform(*standard_deviation_range, 2)))
                   for _ in range(n_clusters)]
    points = np.empty((0, 2))
    while len(points) < n:
        clusters = rng.integers(n_clusters, size=n)
        samples = np.array([rng.multivariate_normal(centres[i], covariances[i]) for i in clusters])
        samples = samples[np.all((samples >= 0) & (samples <= 1), axis=1)]
        points = np.vstack([points, samples])
    return points[:n]


def _random_covariance(rng, eigenvalues):
    angle = rng.uniform(0, np.pi)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return rotation @ np.diag(eigenvalues) @ rotation.T


def gaussian_masses(points, rng, n_gauss=4, eigenvalue_range=(1.8, 100.0), mask=None, floor=1e-3):
    spacing = 1 / np.sqrt(len(points))
    density = np.zeros(len(points))
    for _ in range(n_gauss):
        eigenvalues = np.exp(rng.uniform(*np.log(eigenvalue_range), 2)) * spacing**2
        covariance = _random_covariance(rng, eigenvalues)
        offsets = points - rng.random(2)
        quadratic_form = np.einsum("ij,jk,ik->i", offsets, np.linalg.inv(covariance), offsets)
        density += rng.random() * np.exp(-quadratic_form / 2) / (2 * np.pi * np.sqrt(eigenvalues.prod()))
    if mask == "halfplane":
        normal = rng.normal(size=2)
        density[(points - rng.random(2)) @ normal > 0] = 0
    elif mask == "disc":
        density[np.linalg.norm(points - rng.random(2), axis=1) < rng.uniform(0.1, 0.3)] = 0
    elif mask is not None:
        raise ValueError(mask)
    return density + floor * density.mean()


def make_problem(n, points_kind, rng, atoms=10**9, **mass_kwargs):
    """atoms: masses as integer numbers of mass atoms with exactly `atoms` in total (Schmitzer, Remark 6.2);
    None gives float masses normalised to 1."""
    X = sample_points(n, points_kind, rng)
    Y = sample_points(n, points_kind, rng)
    mu = gaussian_masses(X, rng, **mass_kwargs)
    nu = gaussian_masses(Y, rng, **mass_kwargs)
    if atoms is not None:
        return X, quantize(mu, atoms), Y, quantize(nu, atoms)
    mu /= mu.sum()
    nu /= nu.sum()
    for _ in range(10):
        nu[np.argmax(nu)] += mu.sum() - nu.sum()
    return X, mu, Y, nu


def quantize(mass, atoms):
    """Integer numbers of atoms (as floats, exact below 2**53), at least one per point, summing to exactly atoms."""
    counts = np.maximum(np.round(mass / mass.sum() * atoms), 1)
    counts[np.argmax(counts)] += atoms - counts.sum()
    return counts
