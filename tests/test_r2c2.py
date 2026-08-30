"""Tests R2C2 : solaire sur l'air seulement, deux échelles."""

import numpy as np
import pytest

from basic_mpc.models.r2c2 import R2C2Params, continuous_matrices, discretize, simulate_r2c2


def test_solaire_pas_sur_la_masse() -> None:
    """C'est la différence avec le plant."""
    params = R2C2Params(rae=1e4, ram=3e3, cm=10.0, g_solar=1e-5, g_heating=1e-3)
    _a, mat_b = continuous_matrices(params)
    assert mat_b[1, 1] == pytest.approx(0.0)
    assert mat_b[0, 1] == pytest.approx(1e-5)


def test_masse_plus_lente_que_lair() -> None:
    """τ_masse > τ_air pour un bâtiment."""
    params = R2C2Params(rae=1e4, ram=3e3, cm=16.0, g_solar=0.0, g_heating=0.0)
    assert params.tau_mass_hours > params.tau_air_hours


def test_discretisation_2x2() -> None:
    """Ad 2×2, Bd 2×3."""
    params = R2C2Params(rae=8e3, ram=2e3, cm=12.0, g_solar=1e-6, g_heating=1e-3)
    ad, bd = discretize(params)
    assert ad.shape == (2, 2)
    assert bd.shape == (2, 3)


def test_simulate_deux_etats() -> None:
    """x a deux colonnes ; y suit l'air."""
    params = R2C2Params(
        rae=8e3,
        ram=2e3,
        cm=12.0,
        g_solar=0.0,
        g_heating=0.0,
        process_noise_std=0.0,
        process_noise_std_mass=0.0,
        sensor_noise_std=0.0,
    )
    n = 40
    t_ext = np.full(n, 5.0)
    x, y = simulate_r2c2(params, t_ext, np.zeros(n), np.zeros(n), x0=np.array([20.0, 18.0]))
    assert x.shape == (n, 2)
    assert y[0] == pytest.approx(20.0)
    assert x[-1, 0] < 20.0
