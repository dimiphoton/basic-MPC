"""Tests du filtre de Kalman (état vrai connu, cas synthétique)."""

import numpy as np

from basic_mpc.identification.pem import filter_r1c1
from basic_mpc.models.r1c1 import R1C1Params, simulate_r1c1


def _entrees(n: int, seed: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    hours = np.arange(n) * 0.25  # pas 15 min pour varier un peu
    t_ext = 8.0 + 6.0 * np.sin(2.0 * np.pi * hours / 24.0)
    solar = np.clip(np.sin(2.0 * np.pi * (hours - 6.0) / 24.0), 0.0, None) * 2000.0
    heating = np.where((hours % 24 < 8) | (hours % 24 > 20), 20.0, 0.0)
    heating = heating + rng.normal(0, 0.5, n).clip(min=0)
    return t_ext, solar, heating


def test_kalman_plus_proche_de_letat_que_la_mesure() -> None:
    """Le filtre bat la mesure brute quand le modèle est le bon."""
    params = R1C1Params(
        a=0.96,
        g_solar=2e-5,
        g_heating=4e-3,
        process_noise_std=0.03,
        sensor_noise_std=0.25,
    )
    n = 400
    t_ext, solar, heating = _entrees(n)
    x_true, y = simulate_r1c1(params, t_ext, solar, heating, x0=18.0, seed=2)
    u = np.column_stack([t_ext, solar, heating])
    result = filter_r1c1(y, u, params)
    x_hat = result.x_filt[:, 0]
    rmse_filtre = float(np.sqrt(np.mean((x_hat - x_true) ** 2)))
    rmse_mesure = float(np.sqrt(np.mean((y - x_true) ** 2)))
    assert rmse_filtre < rmse_mesure
    assert rmse_filtre < 0.15
