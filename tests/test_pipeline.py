"""Tests du prétraitement : maille régulière, trous longs non pontés."""

from pathlib import Path

import pandas as pd
import pytest

from basic_mpc.config import DataConfig
from basic_mpc.data.loading import load_raw_csv
from basic_mpc.data.pipeline import regularize_series, run_preprocess


def test_load_raw_csv_trie_et_nettoie_lentete(tmp_path: Path) -> None:
    """Les guillemets d'en-tête et le désordre temporel sont gérés."""
    csv_path = tmp_path / "capteur.csv"
    csv_path.write_text(
        "time,current_value\n"
        "2020-05-24T17:20:03Z,21.0\n"
        "2020-05-24T17:15:03Z,20.8\n",
        encoding="utf-8",
    )
    frame = load_raw_csv(csv_path)
    assert list(frame["current_value"]) == [20.8, 21.0]


def test_load_raw_csv_fichier_absent(tmp_path: Path) -> None:
    """Cas limite : fichier manquant."""
    with pytest.raises(FileNotFoundError):
        load_raw_csv(tmp_path / "absent.csv")


def test_regularize_comble_un_point_saute() -> None:
    """Un trou de 10 min (un pas manquant) est interpolé."""
    index = pd.to_datetime(
        ["2020-01-01 00:00:00Z", "2020-01-01 00:10:00Z"],
        utc=True,
    )
    series = pd.Series([10.0, 12.0], index=index)
    regular = regularize_series(series, freq="5min", max_fill_periods=2)
    milieu = regular.loc["2020-01-01 00:05:00+00:00"]
    assert milieu == pytest.approx(11.0)


def test_regularize_ne_ponte_pas_un_trou_long() -> None:
    """Un trou d'une heure reste NaN au milieu."""
    index = pd.to_datetime(
        ["2020-01-01 00:00:00Z", "2020-01-01 01:00:00Z"],
        utc=True,
    )
    series = pd.Series([10.0, 20.0], index=index)
    regular = regularize_series(series, freq="5min", max_fill_periods=2)
    milieu = regular.loc["2020-01-01 00:30:00+00:00"]
    assert pd.isna(milieu)


def _ecrire_mini_raw(dossier: Path) -> None:
    """Deux CSV alignés, maille 5 min, un trou extérieur."""
    (dossier / "temperature_livingroom.csv").write_text(
        "time,current_value,setpoint\n"
        "2020-05-24T17:00:00Z,20.0,21.0\n"
        "2020-05-24T17:05:00Z,20.1,21.0\n"
        "2020-05-24T17:10:00Z,20.2,21.0\n",
        encoding="utf-8",
    )
    (dossier / "temperature_outside.csv").write_text(
        "time,current_value\n"
        "2020-05-24T17:00:00Z,14\n"
        "2020-05-24T17:05:00Z,15\n"
        "2020-05-24T17:10:00Z,14\n",
        encoding="utf-8",
    )


def test_run_preprocess_ecrit_csv_et_rapport(tmp_path: Path) -> None:
    """Le pipeline v1 produit la table 5 min et le JSON capteurs."""
    brut = tmp_path / "raw"
    brut.mkdir()
    traite = tmp_path / "processed"
    _ecrire_mini_raw(brut)
    config = DataConfig(raw_dir=brut, processed_dir=traite)
    rapport = run_preprocess(config)
    assert (traite / "livingroom_outdoor_5min.csv").is_file()
    assert (traite / "quality_report.json").is_file()
    assert rapport["sensors"]["livingroom"]["resolution"] == pytest.approx(0.1)
    assert rapport["sensors"]["outdoor"]["resolution"] == pytest.approx(1.0)
    table = pd.read_csv(traite / "livingroom_outdoor_5min.csv")
    assert "livingroom_y" in table.columns
    assert "outdoor_y" in table.columns
