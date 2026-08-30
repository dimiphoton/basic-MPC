"""Plant simulé et modèles RC identifiés."""

from basic_mpc.models.kalman import KalmanResult, run_kalman
from basic_mpc.models.plant import (
    PlantParams,
    ThermalPlant,
    discretize as discretize_plant,
    literature_plant_params,
    synthetic_weather,
    run_simulate_plant,
)
from basic_mpc.models.r1c1 import R1C1Params, discretize as discretize_r1c1, simulate_r1c1

__all__ = [
    "KalmanResult",
    "PlantParams",
    "R1C1Params",
    "ThermalPlant",
    "discretize_plant",
    "discretize_r1c1",
    "literature_plant_params",
    "run_kalman",
    "run_simulate_plant",
    "simulate_r1c1",
    "synthetic_weather",
]
