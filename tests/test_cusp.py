import numpy as np

from placepulse_cusp.cusp.density import CuspDensity


def test_cusp_density_is_normalised():
    model = CuspDensity(quadrature_points=240, domain=(-8, 8))
    model.params_ = np.array([0.2, 0.4, 0.0, 1.0, 0.0, 0.2, -0.2])
    grid = np.linspace(-8, 8, 4001)
    x = np.repeat(np.array([[0.3, -0.1]]), len(grid), axis=0)
    density = np.exp(model.logpdf(x, grid))
    integral = np.trapezoid(density, grid)
    assert abs(integral - 1.0) < 2e-3


def test_fold_mask_matches_cubic_discriminant():
    model = CuspDensity()
    model.params_ = np.array([0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    x = np.zeros((3, 2))
    assert model.fold_mask(x).all()

