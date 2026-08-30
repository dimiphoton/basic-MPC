"""Plant thermique : R2C2 « littérature » distinct du modèle identifié.

Le modèle d'identification (prochaines étapes) met le solaire sur l'air
seulement. Ici le solaire chauffe aussi la masse (``alpha_s_mass``) :
rejouer le MPC sur ce plant n'est pas circulaire.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import expm

from basic_mpc.config import DataConfig
from basic_mpc.data.sensors import quantize_measurement

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlantParams:
    """Paramètres fixes du plant (ne pas réutiliser à l'identification).

    Units : C en J/K, R en K/W, P et S dans les unités proxy du repo.
    ``alpha_s_mass`` n'existe pas dans le R2C2 du brief d'identification.
    """

    ca: float
    cm: float
    ram: float
    rae: float
    alpha_s_air: float
    alpha_s_mass: float
    alpha_h: float
    dt_seconds: float = 300.0
    sensor_resolution: float = 0.1
    process_noise_std: float = 0.01
    sensor_noise_std: float = 0.02


def literature_plant_params() -> PlantParams:
    """Jeu type maison : air ~40 min, masse plus lente, solaire sur les murs."""
    return PlantParams(
        ca=5.0e5,
        cm=8.0e6,
        ram=5.0e-3,
        rae=2.0e-2,
        alpha_s_air=8.0e-3,
        alpha_s_mass=4.0e-3,
        alpha_h=4.0,
    )


def _continuous_matrices(params: PlantParams) -> tuple[np.ndarray, np.ndarray]:
    """A (2x2) et B (2x3) pour u = [T_ext, S, P]."""
    inv_ram_ca = 1.0 / (params.ram * params.ca)
    inv_rae_ca = 1.0 / (params.rae * params.ca)
    inv_ram_cm = 1.0 / (params.ram * params.cm)
    mat_a = np.array(
        [
            [-(inv_ram_ca + inv_rae_ca), inv_ram_ca],
            [inv_ram_cm, -inv_ram_cm],
        ]
    )
    mat_b = np.array(
        [
            [inv_rae_ca, params.alpha_s_air / params.ca, params.alpha_h / params.ca],
            [0.0, params.alpha_s_mass / params.cm, 0.0],
        ]
    )
    return mat_a, mat_b


def discretize(params: PlantParams) -> tuple[np.ndarray, np.ndarray]:
    """Discrétisation exacte à ``dt`` : Ad, Bd.

    Returns
    -------
    Ad, Bd : ndarray
        ``x_{k+1} = Ad x_k + Bd u_k``.
    """
    mat_a, mat_b = _continuous_matrices(params)
    dt = params.dt_seconds
    mat_ad = expm(mat_a * dt)
    # Bd = A^{-1} (Ad - I) B  (A est inversible : fuite vers l'extérieur)
    mat_bd = np.linalg.solve(mat_a, (mat_ad - np.eye(2)) @ mat_b)
    return mat_ad, mat_bd


class ThermalPlant:
    """Simulateur : état vrai [T_air, T_masse], mesure quantifiée de l'air."""

    def __init__(
        self,
        params: PlantParams | None = None,
        x0: np.ndarray | None = None,
        seed: int = 0,
    ) -> None:
        """Parameters
        ----------
        params : PlantParams, optional
            Défaut : ``literature_plant_params()``.
        x0 : ndarray, optional
            [T_air, T_masse] initiale. Défaut 20 °C / 20 °C.
        seed : int
            Bruit de process et de capteur reproductible.
        """
        self.params = params or literature_plant_params()
        self.ad, self.bd = discretize(self.params)
        if x0 is None:
            self.x = np.array([20.0, 20.0], dtype=float)
        else:
            self.x = np.asarray(x0, dtype=float).reshape(2).copy()
        self.rng = np.random.default_rng(seed)

    def observe(self) -> float:
        """Mesure du capteur : T_air + bruit, puis quantification.

        Returns
        -------
        float
            y, pas l'état vrai.
        """
        bruit = self.rng.normal(0.0, self.params.sensor_noise_std)
        return quantize_measurement(
            float(self.x[0] + bruit),
            self.params.sensor_resolution,
        )

    def step(self, t_ext: float, solar: float, heating: float) -> float:
        """Un pas de 5 min.

        Parameters
        ----------
        t_ext, solar, heating : float
            Entrées (mêmes définitions que le CSV d'identification).

        Returns
        -------
        float
            Observation y après le pas.
        """
        u = np.array([t_ext, solar, heating], dtype=float)
        process = self.rng.normal(0.0, self.params.process_noise_std, size=2)
        self.x = self.ad @ self.x + self.bd @ u + process
        return self.observe()

    def simulate(
        self,
        t_ext: np.ndarray,
        solar: np.ndarray,
        heating: np.ndarray,
    ) -> pd.DataFrame:
        """Trajectoire complète ; colonnes d'état vrai + y.

        Parameters
        ----------
        t_ext, solar, heating : ndarray
            Séries alignées, un point = un pas ``dt``.

        Returns
        -------
        DataFrame
            ``t_ext``, ``S``, ``P``, ``ta_true``, ``tm_true``, ``y``.
        """
        n = len(t_ext)
        ta = np.empty(n)
        tm = np.empty(n)
        y = np.empty(n)
        for i in range(n):
            y[i] = self.step(float(t_ext[i]), float(solar[i]), float(heating[i]))
            ta[i] = self.x[0]
            tm[i] = self.x[1]
        return pd.DataFrame(
            {
                "t_ext": t_ext,
                "S": solar,
                "P": heating,
                "ta_true": ta,
                "tm_true": tm,
                "y": y,
            }
        )


def synthetic_weather(n_steps: int, dt_seconds: float, seed: int = 1) -> pd.DataFrame:
    """Scénario  : extérieur sinusoïdal, solaire diurne, pas de chauffage.

    Parameters
    ----------
    n_steps : int
        Nombre de pas.
    dt_seconds : float
        Durée d'un pas.
    seed : int
        Bruit léger sur T_ext.

    Returns
    -------
    DataFrame
        Colonnes ``t_ext``, ``S``, ``P``.
    """
    rng = np.random.default_rng(seed)
    hours = np.arange(n_steps) * dt_seconds / 3600.0
    t_ext = 8.0 + 6.0 * np.sin(2.0 * np.pi * hours / 24.0) + rng.normal(0, 0.3, n_steps)
    # Jour : sin positif midi
    angle = 2.0 * np.pi * (hours - 6.0) / 24.0
    solar = np.clip(np.sin(angle), 0.0, None) * 2000.0
    heating = np.zeros(n_steps)
    return pd.DataFrame({"t_ext": t_ext, "S": solar, "P": heating})


def run_simulate_plant(
    n_hours: float = 48.0,
    seed: int = 0,
    processed_dir: Path | None = None,
) -> dict:
    """Génère une trajectoire synthétique et un JSON de paramètres.

    Parameters
    ----------
    n_hours : float
        Durée du scénario météo.
    seed : int
        Bruit du plant.
    processed_dir : Path, optional
        Défaut : ``DataConfig().processed_dir``.

    Returns
    -------
    dict
        Paramètres et chemins écrits.
    """
    config = DataConfig()
    out_dir = processed_dir if processed_dir is not None else config.processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    params = literature_plant_params()
    n_steps = int(n_hours * 3600.0 / params.dt_seconds)
    weather = synthetic_weather(n_steps, params.dt_seconds, seed=seed + 1)
    plant = ThermalPlant(params=params, x0=np.array([18.0, 16.0]), seed=seed)
    traj = plant.simulate(
        weather["t_ext"].to_numpy(),
        weather["S"].to_numpy(),
        weather["P"].to_numpy(),
    )
    csv_path = out_dir / "plant_synthetic.csv"
    traj.to_csv(csv_path, index=False)
    report = {
        "distinction": (
            "alpha_s_mass > 0 sur le plant ; le R2C2 identifié n'a le "
            "solaire que sur l'air (brief). Paramètres littérature, pas appris."
        ),
        "n_steps": n_steps,
        "dt_seconds": params.dt_seconds,
        "alpha_s_mass": params.alpha_s_mass,
        "sensor_resolution": params.sensor_resolution,
        "y_min": float(traj["y"].min()),
        "y_max": float(traj["y"].max()),
        "tm_minus_ta_mean": float((traj["tm_true"] - traj["ta_true"]).mean()),
    }
    json_path = out_dir / "plant_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("écrit %s", csv_path)
    logger.info("écrit %s", json_path)
    return report

