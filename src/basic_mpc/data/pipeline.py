"""Prétraitement v1 : maille régulière + modèle de capteur.

On n'interpole que les trous courts (un échantillon sauté). Un trou
de plusieurs heures reste NaN : ce n'est pas de la physique.
"""

from __future__ import annotations

import json
import logging

import pandas as pd

from basic_mpc.config import DataConfig
from basic_mpc.data.loading import load_raw_csv
from basic_mpc.data.sensors import SensorModel, infer_resolution

logger = logging.getLogger(__name__)


def sampling_report(
    frame: pd.DataFrame,
    value_column: str,
    long_gap_seconds: float,
) -> dict:
    """Statistiques d'échantillonnage et de quantification.

    Parameters
    ----------
    frame : DataFrame
        Sortie de ``load_raw_csv``.
    value_column : str
        Colonne de mesure.
    long_gap_seconds : float
        Seuil (s) au-delà duquel un écart est un « long trou ».

    Returns
    -------
    dict
        Résumé JSON-serializable.
    """
    times = frame["time"]
    delta_sec = times.diff().dt.total_seconds()
    values = pd.to_numeric(frame[value_column], errors="coerce")
    resolution = infer_resolution(values.dropna().to_numpy())
    return {
        "n_rows": int(len(frame)),
        "t_start": times.iloc[0].isoformat(),
        "t_end": times.iloc[-1].isoformat(),
        "dt_seconds_median": float(delta_sec.median()),
        "dt_seconds_max": float(delta_sec.max()),
        "n_long_gaps": int((delta_sec > long_gap_seconds).sum()),
        "n_missing_values": int(values.isna().sum()),
        "value_min": float(values.min()),
        "value_max": float(values.max()),
        "resolution": resolution,
        "n_unique": int(values.nunique()),
    }


def regularize_series(
    series: pd.Series,
    freq: str,
    max_fill_periods: int,
) -> pd.Series:
    """Passe une série à une maille régulière.

    Parameters
    ----------
    series : Series
        Index datetime, une mesure.
    freq : str
        Règle pandas (ex. ``5min``).
    max_fill_periods : int
        Interpolation temporelle limitée (pas de pontage des longs trous).

    Returns
    -------
    Series
        Maille régulière, NaN conservés sur les longs trous.
    """
    if series.empty:
        return series
    regular = series.resample(freq).mean()
    if max_fill_periods <= 0:
        return regular
    return regular.interpolate(method="time", limit=max_fill_periods)


def to_indexed_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Série numérique indexée par le temps."""
    values = pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(values.to_numpy(), index=pd.DatetimeIndex(frame["time"]), name=column)


def run_preprocess(config: DataConfig | None = None) -> dict:
    """Charge salon + extérieur, régularise, écrit ``data/processed``.

    Parameters
    ----------
    config : DataConfig, optional
        Chemins et maille. Défaut : config du repo.

    Returns
    -------
    dict
        Rapport qualité (aussi écrit en JSON).
    """
    config = config or DataConfig()
    living = load_raw_csv(config.raw_dir / config.living_room_file)
    outdoor = load_raw_csv(config.raw_dir / config.outdoor_file)

    living_sensor = SensorModel(
        name="livingroom",
        resolution=infer_resolution(
            pd.to_numeric(living["current_value"], errors="coerce").dropna().to_numpy()
        ),
        unit="celsius",
        quantity="temperature",
    )
    outdoor_sensor = SensorModel(
        name="outdoor",
        resolution=infer_resolution(
            pd.to_numeric(outdoor["current_value"], errors="coerce").dropna().to_numpy()
        ),
        unit="celsius",
        quantity="temperature",
    )

    y_in = regularize_series(
        to_indexed_series(living, "current_value"),
        config.resample_rule,
        config.max_fill_periods,
    )
    setpoint = regularize_series(
        to_indexed_series(living, "setpoint"),
        config.resample_rule,
        config.max_fill_periods,
    )
    y_out = regularize_series(
        to_indexed_series(outdoor, "current_value"),
        config.resample_rule,
        config.max_fill_periods,
    )
    processed = pd.DataFrame(
        {
            "livingroom_y": y_in,
            "livingroom_setpoint": setpoint,
            "outdoor_y": y_out,
        }
    )
    processed.index.name = "time"

    report = {
        "resample_rule": config.resample_rule,
        "max_fill_periods": config.max_fill_periods,
        "n_rows_processed": int(len(processed)),
        "n_nan_livingroom": int(processed["livingroom_y"].isna().sum()),
        "n_nan_outdoor": int(processed["outdoor_y"].isna().sum()),
        "livingroom_raw": sampling_report(
            living, "current_value", config.long_gap_seconds
        ),
        "outdoor_raw": sampling_report(
            outdoor, "current_value", config.long_gap_seconds
        ),
        "sensors": {
            "livingroom": {
                "resolution": living_sensor.resolution,
                "unit": living_sensor.unit,
                "equation": living_sensor.observation_equation(),
            },
            "outdoor": {
                "resolution": outdoor_sensor.resolution,
                "unit": outdoor_sensor.unit,
                "equation": outdoor_sensor.observation_equation(),
            },
        },
    }

    config.processed_dir.mkdir(parents=True, exist_ok=True)
    csv_path = config.processed_dir / "livingroom_outdoor_5min.csv"
    json_path = config.processed_dir / "quality_report.json"
    processed.to_csv(csv_path)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("écrit %s (%s lignes)", csv_path, len(processed))
    logger.info("écrit %s", json_path)
    return report
