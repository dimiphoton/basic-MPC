"""Tests du R1C1 discret (grey-box : gain extérieur = 1 − a)."""

import numpy as np
import pytest

from basic_mpc.models.r1c1 import R1C1Params, discretize, simulate_r1c1


def test_gain_exterieur_est_un_moins_a() -> None:
    """Contrainte grey-box, pas un b libre."""
    params = R1C1Params(a=0.9, g_solar=0.0, g_heating=0.0)
    ad, bd = discretize(params)
    assert ad[0, 0] == pytest.approx(0.9)
    assert bd[0, 0] == pytest.approx(0.1)


def test_sans_apports_air_suit_lextérieur() -> None:
    """T_ext froid → l'air baisse, sans passer sous T_ext."""
    params = R1C1Params(
        a=0.9,
        g_solar=0.0,
        g_heating=0.0,
        process_noise_std=0.0,
        sensor_noise_std=0.0,
    )
    n = 80
    t_ext = np.full(n, 5.0)
    x, _y = simulate_r1c1(
        params, t_ext, np.zeros(n), np.zeros(n), x0=20.0, seed=0
    )
    assert x[-1] < 20.0
    assert x[-1] > 5.0


def test_tau_heures_coherent_avec_a() -> None:
    """a = exp(-dt/tau)."""
    dt = 300.0
    tau_h = 6.0
    a = float(np.exp(-dt / (tau_h * 3600.0)))
    params = R1C1Params(a=a, g_solar=0.0, g_heating=0.0, dt_seconds=dt)
    assert params.tau_hours == pytest.approx(tau_h, rel=1e-6)
