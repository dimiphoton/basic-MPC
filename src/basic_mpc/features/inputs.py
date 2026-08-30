"""Entrées du modèle RC : chauffage et solaire, construits, pas mesurés.

Ce ne sont pas des watts ni des W/m². Les coefficients RC absorberont
l'échelle. On documente la formule pour ne pas les confondre avec P, S
physiques du brief.
"""

from __future__ import annotations

import json
import logging

import pandas as pd

from basic_mpc.config import DataConfig
from basic_mpc.data.loading import load_raw_csv
from basic_mpc.data.pipeline import regularize_series, run_preprocess, to_indexed_series

logger = logging.getLogger(__name__)

PHASES_PV = ("L1 PV", "L2 PV", "L3 PV")

# Formules figées (README, decisions.md, tests)
HEATING_FORMULA = "P = max(T_eau - T_air, 0) * 1_{T_air < consigne}  [K, pas des W]"
SOLAR_FORMULA = "S = max(L1_PV,0) + max(L2_PV,0) + max(L3_PV,0)  [proxy, pas des W/m²]"


def heating_proxy(
    water_temperature: pd.Series,
    indoor_y: pd.Series,
    setpoint: pd.Series,
) -> pd.Series:
    """Proxy d'apport chauffage : écart eau/air, seulement si la zone demande.

    Parameters
    ----------
    water_temperature : Series
        Température d'eau du circuit, même index que l'air.
    indoor_y : Series
        Température mesurée de la zone.
    setpoint : Series
        Consigne de la zone.

    Returns
    -------
    Series
        ``P`` en kelvin (écart), pas une puissance. 0 si pas d'appel
        (air ≥ consigne) ou si l'eau n'est pas plus chaude que l'air.
    """
    table = pd.concat(
        [water_temperature, indoor_y, setpoint],
        axis=1,
        keys=["water", "indoor", "setpoint"],
    )
    delta_eau_air = (table["water"] - table["indoor"]).clip(lower=0)
    appel_zone = (table["indoor"] < table["setpoint"]).astype(float)
    proxy = delta_eau_air * appel_zone
    proxy.name = "P"
    return proxy


def solar_proxy_from_phases(
    phase_l1: pd.Series,
    phase_l2: pd.Series,
    phase_l3: pd.Series,
) -> pd.Series:
    """Proxy solaire : somme des productions PV, négatifs ramenés à 0.

    Parameters
    ----------
    phase_l1, phase_l2, phase_l3 : Series
        Puissances des trois phases (bruit parfois légèrement négatif).

    Returns
    -------
    Series
        Proxy ``S``, même échelle que le PV (pas une irradiance).
    """
    proxy = (
        phase_l1.clip(lower=0) + phase_l2.clip(lower=0) + phase_l3.clip(lower=0)
    )
    proxy.name = "S"
    return proxy


def _regularize_heating(config: DataConfig) -> pd.Series:
    """Température d'eau à la maille 5 min."""
    heating = load_raw_csv(config.raw_dir / config.heating_file)
    return regularize_series(
        to_indexed_series(heating, "water_temperature"),
        config.resample_rule,
        config.max_fill_periods,
    )


def _regularize_solar(config: DataConfig) -> pd.Series:
    """PV 1 min → somme des phases → maille 5 min (moyenne)."""
    pv = load_raw_csv(config.raw_dir / config.pv_file)
    manquantes = [nom for nom in PHASES_PV if nom not in pv.columns]
    if manquantes:
        msg = f"colonnes PV absentes : {manquantes} (vu {list(pv.columns)})"
        raise ValueError(msg)
    index = pd.DatetimeIndex(pv["time"])
    s_1min = solar_proxy_from_phases(
        pd.Series(pd.to_numeric(pv["L1 PV"], errors="coerce").to_numpy(), index=index),
        pd.Series(pd.to_numeric(pv["L2 PV"], errors="coerce").to_numpy(), index=index),
        pd.Series(pd.to_numeric(pv["L3 PV"], errors="coerce").to_numpy(), index=index),
    )
    return regularize_series(s_1min, config.resample_rule, config.max_fill_periods)


def run_build_inputs(config: DataConfig | None = None) -> dict:
    """Ajoute P et S à la table 5 min, écrit le CSV d'identification.

    Parameters
    ----------
    config : DataConfig, optional
        Chemins et maille.

    Returns
    -------
    dict
        Rapport des entrées (formules + stats).
    """
    config = config or DataConfig()
    run_preprocess(config)
    processed_path = config.processed_dir / "livingroom_outdoor_5min.csv"
    table = pd.read_csv(processed_path, index_col="time", parse_dates=True)
    if table.index.tz is None:
        table.index = table.index.tz_localize("UTC")

    water = _regularize_heating(config).reindex(table.index)
    solar = _regularize_solar(config).reindex(table.index)
    heating = heating_proxy(
        water,
        table["livingroom_y"],
        table["livingroom_setpoint"],
    )
    table = table.copy()
    table["water_temperature"] = water
    table["P"] = heating
    table["S"] = solar

    ident_path = config.processed_dir / "identification_5min.csv"
    table.to_csv(ident_path)

    appel = (table["livingroom_y"] < table["livingroom_setpoint"]).mean()
    report = {
        "heating_formula": HEATING_FORMULA,
        "solar_formula": SOLAR_FORMULA,
        "n_rows": int(len(table)),
        "n_nan_P": int(table["P"].isna().sum()),
        "n_nan_S": int(table["S"].isna().sum()),
        "fraction_heating_call": float(appel) if pd.notna(appel) else None,
        "P_mean_when_positive": float(table.loc[table["P"] > 0, "P"].mean())
        if (table["P"] > 0).any()
        else 0.0,
        "S_max": float(table["S"].max()) if table["S"].notna().any() else None,
    }
    report_path = config.processed_dir / "inputs_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("écrit %s", ident_path)
    logger.info("écrit %s", report_path)
    return report
