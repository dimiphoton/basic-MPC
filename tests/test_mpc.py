"""Tests MPC vs bang-bang : hystérésis, bornes, anticipation, artefacts."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from basic_mpc.config import ControlConfig
from basic_mpc.control.bangbang import bangbang_step
from basic_mpc.control.closed_loop import run_mpc_vs_bangbang
from basic_mpc.control.internal import p_to_hold, r2c2_internal_from_plant
from basic_mpc.control.mpc import mpc_first_move
from basic_mpc.models.kalman import KalmanTracker, run_kalman
from basic_mpc.models.plant import literature_plant_params
from basic_mpc.models.r2c2 import continuous_matrices, discretize


def test_bangbang_hysteresis() -> None:
    """Allume sous le bas, éteint au-dessus du haut, sinon tient."""
    p, on = bangbang_step(18.0, heating_on=False, t_low=19.5, t_high=21.0, p_max=10.0)
    assert on is True
    assert p == pytest.approx(10.0)
    p, on = bangbang_step(20.0, heating_on=True, t_low=19.5, t_high=21.0, p_max=10.0)
    assert on is True
    p, on = bangbang_step(21.2, heating_on=True, t_low=19.5, t_high=21.0, p_max=10.0)
    assert on is False
    assert p == pytest.approx(0.0)
    p, on = bangbang_step(20.0, heating_on=False, t_low=19.5, t_high=21.0, p_max=10.0)
    assert on is False


def test_modele_interne_sans_solaire_masse() -> None:
    """Misspecification : le plant a alpha_s_mass, pas le contrôleur."""
    plant = literature_plant_params()
    assert plant.alpha_s_mass > 0.0
    internal = r2c2_internal_from_plant(plant)
    _a, mat_b = continuous_matrices(internal)
    assert mat_b[1, 1] == pytest.approx(0.0)
    assert internal.g_heating > 0.0


def test_p_to_hold_positif_s_il_fait_froid() -> None:
    """Sans soleil, il faut chauffer pour tenir 20 °C à 0 °C."""
    p = p_to_hold(literature_plant_params(), t_set=20.0, t_ext=0.0)
    assert p > 50.0


def test_mpc_respecte_les_bornes() -> None:
    """0 ≤ P ≤ P_max."""
    plant = literature_plant_params()
    ad, bd = discretize(r2c2_internal_from_plant(plant))
    p0, moves = mpc_first_move(
        np.array([18.0, 16.0]),
        np.full(24, 5.0),
        np.zeros(24),
        ad,
        bd,
        p_max=100.0,
        t_set=20.0,
        t_min=19.5,
        t_max=21.0,
        block_len=4,
        q_track=1.0,
        q_band=20.0,
        r_u=1e-6,
    )
    assert 0.0 <= p0 <= 100.0 + 1e-9
    assert np.all(moves >= -1e-9)
    assert np.all(moves <= 100.0 + 1e-9)


def test_mpc_anticipe_une_nuit_froide() -> None:
    """Dans la bande, le bang-bang reste OFF ; le MPC préchauffe."""
    plant = literature_plant_params()
    ad, bd = discretize(r2c2_internal_from_plant(plant))
    p_max = 300.0
    p0, _moves = mpc_first_move(
        np.array([20.4, 20.0]),
        np.full(36, -4.0),
        np.zeros(36),
        ad,
        bd,
        p_max=p_max,
        t_set=20.0,
        t_min=19.5,
        t_max=21.0,
        block_len=6,
        q_track=1.0,
        q_band=25.0,
        r_u=4e-7,
    )
    y, on = 20.4, False
    p_bb, on = bangbang_step(y, on, 19.5, 21.0, p_max)
    assert p_bb == pytest.approx(0.0)
    assert p0 > 10.0


def test_kalman_tracker_suit_run_kalman() -> None:
    """Le filtre en ligne reproduit le batch sur 2 états."""
    plant = literature_plant_params()
    ad, bd = discretize(r2c2_internal_from_plant(plant))
    q = np.diag([0.02**2, 0.02**2])
    r = np.array([[0.05**2]])
    n = 40
    y = 18.0 + 0.1 * np.arange(n)
    u = np.column_stack([np.full(n, 8.0), np.zeros(n), np.full(n, 50.0)])
    x0 = np.array([18.0, 18.0])
    p0 = np.diag([4.0, 16.0])
    batch = run_kalman(y, u, ad, bd, np.array([[1.0, 0.0]]), q, r, x0, p0)
    tracker = KalmanTracker(ad, bd, q, r, x0, p0)
    online = np.empty((n, 2))
    u_prev = None
    for k in range(n):
        online[k] = tracker.step(float(y[k]), u_prev)
        u_prev = u[k]
    np.testing.assert_allclose(online, batch.x_filt, rtol=1e-5, atol=1e-4)


def test_run_mpc_ecrit_s4_s5_s6(tmp_path: Path) -> None:
    """JSON + trois figures du catalogue."""
    cfg = replace(ControlConfig(), n_hours=6.0, horizon_hours=2.0, block_minutes=20.0)
    rapport = run_mpc_vs_bangbang(
        cfg=cfg,
        pictures_dir=tmp_path / "pictures",
        experiments_dir=tmp_path / "exp",
        processed_dir=tmp_path / "processed",
    )
    pictures = tmp_path / "pictures"
    assert (pictures / "s4-mpc-vs-bang-bang.png").is_file()
    assert (pictures / "s5-commande-p.png").is_file()
    assert (pictures / "s6-confort-conso.png").is_file()
    assert (tmp_path / "processed" / "mpc_vs_bangbang_report.json").is_file()
    assert rapport["mpc"]["hours_outside_band"] >= 0.0
    assert rapport["bangbang"]["P_hours"] >= 0.0
    assert "hours_outside_after_2h" in rapport["mpc"]
    assert rapport["plant_alpha_s_mass"] > 0.0
