"""Labo dynamique : météo, burn-in, scores hors transitoire."""

from datetime import datetime
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pytest

from basic_mpc.config import ControlConfig
from basic_mpc.control.tariffs import TZ_BRUSSELS, scenario_index
from basic_mpc.sim.dynamic import (
    BURN_IN_HOURS,
    LAB_START_LOCAL,
    build_scenario,
    draw_weather,
    lab_config,
    lag_hours_xcorr,
    run_strategy,
)


def test_meteo_jours_non_identiques() -> None:
    """Deux jours successifs n'ont pas la même moyenne (marche aléatoire)."""
    start = datetime(2021, 1, 18, 0, 0, tzinfo=TZ_BRUSSELS)
    index = scenario_index(start, n_steps=2 * 24 * 12, dt_seconds=300.0)
    weather = draw_weather(index, seed=7)
    day0 = weather["t_ext"].iloc[: 24 * 12].mean()
    day1 = weather["t_ext"].iloc[24 * 12 :].mean()
    assert abs(float(day0) - float(day1)) > 0.15


def test_x0_pas_le_couple_magique() -> None:
    """Après burn-in, air et murs ne sont plus [18, 16]."""
    scenario = build_scenario(n_days=2.0, weather_seed=3, plant_seed=0)
    x0 = np.asarray(scenario["x0"], dtype=float)
    assert x0.shape == (2,)
    assert not np.allclose(x0, [18.0, 16.0])
    # Maison habitée : près de la consigne, pas gelée à l'extérieur.
    assert x0[0] > 15.0
    assert x0[1] > 14.0


def test_scores_hors_burn_in() -> None:
    """L'index scoré commence à t=0 labo, sans les 24 h d'habitation."""
    cfg = lab_config(n_days=2.0)
    scenario = build_scenario(cfg=cfg, weather_seed=2)
    assert scenario["cfg"].start_local == LAB_START_LOCAL
    first = scenario["index"][0]
    expected = datetime.fromisoformat(LAB_START_LOCAL).replace(tzinfo=TZ_BRUSSELS)
    assert first == expected
    dt_h = scenario["plant_p"].dt_seconds / 3600.0
    duration_h = len(scenario["index"]) * dt_h
    assert duration_h == pytest.approx(cfg.n_hours, abs=0.02)
    run = run_strategy("hysteresis", scenario)
    assert len(run["traj"]) == len(scenario["index"])
    assert BURN_IN_HOURS == 24.0


def test_deux_strategies_meme_meteo_et_x0() -> None:
    """Hystérésis et préchauffage voient le même monde."""
    scenario = build_scenario(n_days=2.0, weather_seed=11, plant_seed=4)
    hys = run_strategy("hysteresis", scenario)
    pre = run_strategy("preheat", scenario)
    np.testing.assert_allclose(hys["x0"], pre["x0"])
    np.testing.assert_allclose(
        hys["traj"]["t_ext"].to_numpy(),
        pre["traj"]["t_ext"].to_numpy(),
    )
    np.testing.assert_allclose(scenario["t_ext"], hys["traj"]["t_ext"].to_numpy())
    assert "kwh_hp" in hys["metrics"]
    assert "kwh_hc" in hys["metrics"]
    different_comfort = hys["metrics"]["hours_under_conf"] != pre["metrics"][
        "hours_under_conf"
    ]
    different_bill = hys["metrics"]["bill_eur"] != pre["metrics"]["bill_eur"]
    assert different_comfort or different_bill


def test_xcorr_retrouve_un_retard_connu() -> None:
    """Une sinusoïde décalée de 6 h donne ~6 h après avoir jeté 24 h."""
    dt_h = 0.25
    hours = np.arange(0.0, 5 * 24, dt_h)
    outdoor = np.sin(2.0 * np.pi * hours / 24.0)
    indoor = np.sin(2.0 * np.pi * (hours - 6.0) / 24.0)
    lag = lag_hours_xcorr(outdoor, indoor, dt_hours=dt_h, discard_hours=24.0)
    assert abs(lag - 6.0) < 0.4


def test_lecon_rc_retard_apres_equilibre() -> None:
    """Départ à T_ext(0), 5 j sans chauffage : le retard n'est pas t=0."""
    from basic_mpc.sim.payload import plant_for_js

    p = plant_for_js()
    dt = p["dt_seconds"]
    n = int(5 * 24 * 3600 / dt)
    t_ext0 = 5.0 + 6.0 * np.sin(2.0 * np.pi * (0.0 - 9.0) / 24.0)
    ta = t_ext0
    tm = t_ext0
    outdoor = np.empty(n)
    indoor = np.empty(n)
    ca, cm, ram, rae = p["ca"], p["cm"], p["ram"], p["rae"]
    for k in range(n):
        hod = (k * dt / 3600.0) % 24.0
        t_ext = 5.0 + 6.0 * np.sin(2.0 * np.pi * (hod - 9.0) / 24.0)
        d_ta = (tm - ta) / (ram * ca) + (t_ext - ta) / (rae * ca)
        d_tm = (ta - tm) / (ram * cm)
        ta += dt * d_ta
        tm += dt * d_tm
        outdoor[k] = t_ext
        indoor[k] = ta
    lag = lag_hours_xcorr(outdoor, indoor, dt_hours=dt / 3600.0, discard_hours=24.0)
    assert 1.0 < lag < 14.0


def test_export_sans_arene(tmp_path) -> None:
    """Le JSON Pages n'embarque plus l'arène précalculée."""
    from basic_mpc.sim.export import run_export_simulator
    from basic_mpc.sim.payload import plant_for_js

    plant = plant_for_js()
    assert "x0" not in plant
    assert plant["n_lab_days"] == 5.0
    out = tmp_path / "sim"
    pics = tmp_path / "pics"
    rapport = run_export_simulator(out_dir=out, pictures_dir=pics)
    payload = json.loads(Path(rapport["json_path"]).read_text(encoding="utf-8"))
    assert "arena" not in payload
    assert "plant" in payload
    assert "fitted_z_24h" in payload


def test_trois_panneaux_datetime() -> None:
    """Les graphes du labo ont 3 axes et l'index scoré."""
    from basic_mpc.sim.charts import fig_lab_panels

    scenario = build_scenario(n_days=2.0, weather_seed=5)
    run = run_strategy("hysteresis", scenario)
    fig = fig_lab_panels(
        run["index"],
        run["traj"],
        run["t_conf"],
        run["pi"],
        run["cfg"].pi_hc,
        run["metrics"]["bill_cum"],
        overlay=None,
    )
    assert len(fig.axes) >= 3
    plt.close(fig)


def test_cfg_duree_et_consignes() -> None:
    """Un cfg custom (durée, T_conf) reste respecté."""
    cfg = lab_config(n_days=3.0, t_conf_occupied=19.5, t_conf_setback=16.0)
    assert isinstance(cfg, ControlConfig)
    scenario = build_scenario(cfg=cfg, weather_seed=1)
    assert scenario["cfg"].t_conf_occupied == 19.5
    hours = len(scenario["index"]) * scenario["plant_p"].dt_seconds / 3600.0
    assert hours == pytest.approx(72.0, abs=0.1)
