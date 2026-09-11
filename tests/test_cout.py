"""MPC v1.1 : bande n, J euros, hystérésis sur T_conf."""

from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from basic_mpc.config import ControlConfig
from basic_mpc.control.actuator import proportional_band_p, t_sp_from_p
from basic_mpc.control.bangbang import hysteresis_comfort_step
from basic_mpc.control.cout import euro_metrics, run_mpc_cout_consigne
from basic_mpc.control.internal import r2c2_internal_from_plant
from basic_mpc.control.mpc import mpc_euro_first_move
from basic_mpc.control.tariffs import (
    TZ_BRUSSELS,
    beta_kwh_per_p_second,
    comfort_setpoint,
    energy_price,
    scenario_index,
)
from basic_mpc.models.plant import literature_plant_params
from basic_mpc.models.r2c2 import discretize


def test_bande_n_aux_bornes() -> None:
    """À la consigne : arrêt ; n °C en dessous : à fond."""
    assert proportional_band_p(20.0, 20.0, n_band=1.0, p_max=100.0) == pytest.approx(0.0)
    assert proportional_band_p(20.0, 19.0, n_band=1.0, p_max=100.0) == pytest.approx(100.0)
    assert proportional_band_p(20.0, 19.5, n_band=1.0, p_max=100.0) == pytest.approx(50.0)
    assert proportional_band_p(20.0, 21.0, n_band=1.0, p_max=100.0) == pytest.approx(0.0)


def test_t_sp_borne_16_22() -> None:
    """La consigne reconstruite reste dans le thermostat."""
    t_sp = t_sp_from_p(18.0, 50.0, n_band=1.0, p_max=100.0, t_sp_min=16.0, t_sp_max=22.0)
    assert 16.0 <= t_sp <= 22.0
    assert t_sp == pytest.approx(18.5)
    clipped = t_sp_from_p(21.5, 100.0, n_band=1.0, p_max=100.0, t_sp_min=16.0, t_sp_max=22.0)
    assert clipped == pytest.approx(22.0)


def test_hysteresis_autour_de_t_conf() -> None:
    """n=1 °C : bande 19,5–20,5 si T_conf=20."""
    p, on = hysteresis_comfort_step(19.4, False, t_conf=20.0, n_band=1.0, p_max=10.0)
    assert on is True
    assert p == pytest.approx(10.0)
    p, on = hysteresis_comfort_step(20.0, True, t_conf=20.0, n_band=1.0, p_max=10.0)
    assert on is True
    p, on = hysteresis_comfort_step(20.6, True, t_conf=20.0, n_band=1.0, p_max=10.0)
    assert on is False


def test_mpc_euro_prechauffe_avant_occupation() -> None:
    """À 5 h, au-dessus du réduit : hystérésis OFF, le MPC chauffe pour 7 h."""
    plant = literature_plant_params()
    ad, bd = discretize(r2c2_internal_from_plant(plant))
    p_max = 300.0
    start = datetime(2021, 1, 15, 5, 0, tzinfo=TZ_BRUSSELS)
    index = scenario_index(start, n_steps=72, dt_seconds=plant.dt_seconds)
    conf = comfort_setpoint(index)
    pi = energy_price(index)
    x0 = np.array([18.2, 17.8])
    p0, moves = mpc_euro_first_move(
        x0,
        np.full(72, 0.0),
        np.zeros(72),
        ad,
        bd,
        p_max,
        conf,
        pi,
        block_len=6,
        n_band=1.0,
        t_sp_min=16.0,
        t_sp_max=22.0,
        beta=beta_kwh_per_p_second(plant.alpha_h),
        lambda_comfort=1.0,
        dt_seconds=plant.dt_seconds,
    )
    p_bb, on = hysteresis_comfort_step(18.2, False, t_conf=17.0, n_band=1.0, p_max=p_max)
    assert on is False
    assert p_bb == pytest.approx(0.0)
    assert p0 > 10.0
    assert np.all(moves >= -1e-9)
    assert np.all(moves <= p_max + 1e-9)
    t_sp = t_sp_from_p(18.2, p0, 1.0, p_max, 16.0, 22.0)
    assert 16.0 <= t_sp <= 22.0


def test_euro_metrics_unites() -> None:
    """1 h à 1 K sous T_conf → 1 € d'inconfort si λ=1."""
    ta = np.array([19.0, 19.0])
    conf = np.array([20.0, 20.0])
    p = np.zeros(2)
    pi = np.array([0.40, 0.40])
    # deux pas de 30 min
    m = euro_metrics(ta, p, conf, pi, beta=1.0, dt_seconds=1800.0, lambda_comfort=1.0)
    assert m["hours_under_conf"] == pytest.approx(1.0)
    assert m["discomfort_eur"] == pytest.approx(1.0)
    assert m["bill_eur"] == pytest.approx(0.0)
    # 0,05 °C sous le capteur : on ne compte pas l'heure.
    m_tiny = euro_metrics(
        np.array([19.95]),
        np.zeros(1),
        np.array([20.0]),
        np.array([0.40]),
        beta=1.0,
        dt_seconds=3600.0,
        lambda_comfort=1.0,
    )
    assert m_tiny["hours_under_conf"] == pytest.approx(0.0)


def test_run_mpc_cout_ecrit_s4_s5_s6(tmp_path: Path) -> None:
    """JSON + trois figures ; métriques en euros."""
    cfg = replace(ControlConfig(), n_hours=6.0, horizon_hours=2.0, block_minutes=20.0)
    rapport = run_mpc_cout_consigne(
        cfg=cfg,
        pictures_dir=tmp_path / "pictures",
        experiments_dir=tmp_path / "exp",
        processed_dir=tmp_path / "processed",
    )
    pictures = tmp_path / "pictures"
    assert (pictures / "s4-mpc-vs-bang-bang.png").is_file()
    assert (pictures / "s5-commande-p.png").is_file()
    assert (pictures / "s6-confort-conso.png").is_file()
    assert (tmp_path / "processed" / "mpc_cout_consigne_report.json").is_file()
    assert rapport["mpc"]["bill_eur"] >= 0.0
    assert rapport["hysteresis"]["hours_under_conf"] >= 0.0
    assert rapport["n_band"] == pytest.approx(1.0)
    assert "energy_kwh" in rapport["mpc"]
