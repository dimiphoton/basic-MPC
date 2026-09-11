"""Calendriers confort / HP-HC et conversion kWh."""

from datetime import datetime

import numpy as np
import pytest

from basic_mpc.config import ControlConfig
from basic_mpc.control.tariffs import (
    TZ_BRUSSELS,
    beta_kwh_per_p_second,
    comfort_setpoint,
    energy_price,
    hour_of_day,
    parse_start_local,
    scenario_index,
    winter_weather,
)
from basic_mpc.models.plant import literature_plant_params


def test_parse_start_naive_est_bruxelles() -> None:
    """ISO sans fuseau → Europe/Brussels."""
    cfg = ControlConfig()
    start = parse_start_local(cfg)
    assert start.tzinfo is not None
    assert start.hour == 18


def test_confort_et_prix_suivent_la_meme_horloge() -> None:
    """Jour = occupé + HP ; nuit = réduit + HC."""
    start = datetime(2021, 1, 15, 6, 0, tzinfo=TZ_BRUSSELS)
    index = scenario_index(start, n_steps=24, dt_seconds=3600.0)
    conf = comfort_setpoint(index)
    pi = energy_price(index)
    hod = hour_of_day(index)
    night = (hod < 7.0) | (hod >= 22.0)
    assert np.all(conf[night] == pytest.approx(17.0))
    assert np.all(conf[~night] == pytest.approx(20.0))
    assert np.all(pi[night] == pytest.approx(0.20))
    assert np.all(pi[~night] == pytest.approx(0.40))


def test_beta_plant_litterature() -> None:
    """β = α_h / 3,6e6 — pas un compteur."""
    beta = beta_kwh_per_p_second(literature_plant_params().alpha_h)
    assert beta == pytest.approx(4.0 / 3.6e6)
    # 1 h à P=1 → quelques millionièmes de kWh.
    assert beta * 3600.0 < 0.01


def test_hiver_solaire_nul_la_nuit() -> None:
    """Le solaire suit midi local, pas t=0 du scénario."""
    start = datetime(2021, 1, 15, 0, 0, tzinfo=TZ_BRUSSELS)
    index = scenario_index(start, n_steps=24, dt_seconds=3600.0)
    weather = winter_weather(index, seed=0)
    # 0 h et 3 h : nuit
    assert weather["S"].iloc[0] == pytest.approx(0.0)
    assert weather["S"].iloc[3] == pytest.approx(0.0)
    # Midi : soleil
    assert weather["S"].iloc[12] > 1000.0
