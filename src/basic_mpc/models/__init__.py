"""Plant simulé et modèles RC identifiés."""

from basic_mpc.models.kalman import KalmanResult, KalmanTracker, run_kalman
from basic_mpc.models.plant import (
    PlantParams,
    ThermalPlant,
    discretize as discretize_plant,
    literature_plant_params,
    synthetic_weather,
    run_simulate_plant,
)
from basic_mpc.models.r1c1 import R1C1Params, discretize as discretize_r1c1, simulate_r1c1
from basic_mpc.models.r2c2 import R2C2Params, discretize as discretize_r2c2, simulate_r2c2

__all__ = [
    "KalmanResult",
    "KalmanTracker",
    "PlantParams",
    "R1C1Params",
    "R2C2Params",
    "ThermalPlant",
    "discretize_plant",
    "discretize_r1c1",
    "discretize_r2c2",
    "literature_plant_params",
    "run_kalman",
    "run_simulate_plant",
    "simulate_r1c1",
    "simulate_r2c2",
    "synthetic_weather",
]
