"""Tests des proxies P (chauffage) et S (solaire)."""

from pathlib import Path

import pandas as pd
import pytest

from basic_mpc.config import DataConfig
from basic_mpc.features.inputs import heating_proxy, run_build_inputs, solar_proxy_from_phases


def test_heating_proxy_zero_si_eau_pas_plus_chaude() -> None:
    """Sans écart eau/air, pas d'apport même sous consigne."""
    index = pd.RangeIndex(2)
    p = heating_proxy(
        water_temperature=pd.Series([20.0, 20.0], index=index),
        indoor_y=pd.Series([21.0, 21.0], index=index),
        setpoint=pd.Series([22.0, 22.0], index=index),
    )
    assert list(p) == [0.0, 0.0]


def test_heating_proxy_zero_si_pas_dappel() -> None:
    """Eau chaude mais air déjà au-dessus de la consigne → P = 0."""
    p = heating_proxy(
        water_temperature=pd.Series([60.0]),
        indoor_y=pd.Series([21.5]),
        setpoint=pd.Series([21.0]),
    )
    assert p.iloc[0] == pytest.approx(0.0)


def test_heating_proxy_ecart_eau_air_si_appel() -> None:
    """Cas nominal : P = T_eau - T_air."""
    p = heating_proxy(
        water_temperature=pd.Series([50.0]),
        indoor_y=pd.Series([20.0]),
        setpoint=pd.Series([21.0]),
    )
    assert p.iloc[0] == pytest.approx(30.0)


def test_solar_proxy_coupe_les_negatifs() -> None:
    """Le -1 des phases PV est du bruit, pas de l'ombre."""
    s = solar_proxy_from_phases(
        pd.Series([-1.0, 10.0]),
        pd.Series([0.0, 5.0]),
        pd.Series([0.0, 1.0]),
    )
    assert s.iloc[0] == pytest.approx(0.0)
    assert s.iloc[1] == pytest.approx(16.0)


def _mini_raw_complet(dossier: Path) -> None:
    """CSV minimaux salon, extérieur, chauffage, PV."""
    (dossier / "temperature_livingroom.csv").write_text(
        "time,current_value,setpoint\n"
        "2020-05-24T17:00:00Z,20.0,21.0\n"
        "2020-05-24T17:05:00Z,21.5,21.0\n",
        encoding="utf-8",
    )
    (dossier / "temperature_outside.csv").write_text(
        "time,current_value\n"
        "2020-05-24T17:00:00Z,14\n"
        "2020-05-24T17:05:00Z,15\n",
        encoding="utf-8",
    )
    (dossier / "temperature_heating_system.csv").write_text(
        "time,water_pressure,water_temperature\n"
        "2020-05-24T17:00:00Z,1.3,50\n"
        "2020-05-24T17:05:00Z,1.3,50\n",
        encoding="utf-8",
    )
    (dossier / "pv_production_load.csv").write_text(
        "time,'L1 PV','L2 PV','L3 PV','L1 Load','L2 Load','L3 Load'\n"
        "2020-05-24T17:00:00Z,10.0,0.0,0.0,1,1,1\n"
        "2020-05-24T17:01:00Z,10.0,0.0,0.0,1,1,1\n"
        "2020-05-24T17:05:00Z,0.0,0.0,0.0,1,1,1\n",
        encoding="utf-8",
    )


def test_run_build_inputs_ecrit_p_et_s(tmp_path: Path) -> None:
    """La table d'identification contient P (appel) et S (PV)."""
    brut = tmp_path / "raw"
    brut.mkdir()
    traite = tmp_path / "processed"
    _mini_raw_complet(brut)
    rapport = run_build_inputs(DataConfig(raw_dir=brut, processed_dir=traite))
    table = pd.read_csv(traite / "identification_5min.csv")
    assert "P" in table.columns
    assert "S" in table.columns
    # 17:00 : air 20 < consigne 21, eau 50 → P = 30
    assert table.loc[0, "P"] == pytest.approx(30.0)
    # 17:05 : air 21.5 > consigne 21 → pas d'appel
    assert table.loc[1, "P"] == pytest.approx(0.0)
    assert rapport["heating_formula"].startswith("P =")
    assert (traite / "inputs_report.json").is_file()
