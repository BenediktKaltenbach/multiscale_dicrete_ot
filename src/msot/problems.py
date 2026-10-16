import numpy as np


def sample_points(n, kind, rng):
    if kind == "uniform":
        return rng.random((n, 2))
    if kind == "grid":
        m = int(round(np.sqrt(n)))
        g = (np.arange(m) + 0.5) / m
        return np.stack(np.meshgrid(g, g), axis=-1).reshape(-1, 2)
    if kind == "clustered":
        return _gaussian_mixture(n, rng)
    raise ValueError(kind)


def _gaussian_mixture(n, rng, n_clusters=4, std_range=(0.03, 0.15)):
    centres = rng.uniform(0.15, 0.85, (n_clusters, 2))
    covs = [_random_cov(rng, np.square(rng.uniform(*std_range, 2))) for _ in range(n_clusters)]
    points = np.empty((0, 2))
    while len(points) < n:
        k = rng.integers(n_clusters, size=n)
        z = np.array([rng.multivariate_normal(centres[i], covs[i]) for i in k])
        z = z[np.all((z >= 0) & (z <= 1), axis=1)]
        points = np.vstack([points, z])
    return points[:n]


def _random_cov(rng, eig):
    a = rng.uniform(0, np.pi)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return R @ np.diag(eig) @ R.T


def gaussian_masses(points, rng, n_gauss=4, eig_range=(1.8, 100.0), mask=None, floor=1e-3):
    h = 1 / np.sqrt(len(points))
    density = np.zeros(len(points))
    for _ in range(n_gauss):
        eig = np.exp(rng.uniform(*np.log(eig_range), 2)) * h**2
        cov = _random_cov(rng, eig)
        d = points - rng.random(2)
        quad = np.einsum("ij,jk,ik->i", d, np.linalg.inv(cov), d)
        density += rng.random() * np.exp(-quad / 2) / (2 * np.pi * np.sqrt(eig.prod()))
    if mask == "halfplane":
        normal = rng.normal(size=2)
        density[(points - rng.random(2)) @ normal > 0] = 0
    elif mask == "disc":
        density[np.linalg.norm(points - rng.random(2), axis=1) < rng.uniform(0.1, 0.3)] = 0
    elif mask is not None:
        raise ValueError(mask)
    return density + floor * density.mean()


def make_problem(n, points_kind, rng, **mass_kwargs):
    X = sample_points(n, points_kind, rng)
    Y = sample_points(n, points_kind, rng)
    mu = gaussian_masses(X, rng, **mass_kwargs)
    nu = gaussian_masses(Y, rng, **mass_kwargs)
    mu /= mu.sum()
    nu /= nu.sum()
    for _ in range(10):
        nu[np.argmax(nu)] += mu.sum() - nu.sum()
    return X, mu, Y, nu
