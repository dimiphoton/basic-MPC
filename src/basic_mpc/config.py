"""Chemins et paramètres du prétraitement.

Les fréquences et seuils de trous vivent ici, pas dans le code métier.
"""

from dataclasses import dataclass
from pathlib import Path

# src/basic_mpc/config.py → racine du repo
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DataConfig:
    """Configuration des données v1 (salon + extérieur).

    Parameters
    ----------
    raw_dir : Path
        Dossier des CSV bruts (jamais modifié).
    processed_dir : Path
        Sortie régénérable du prétraitement.
    resample_rule : str
        Maille cible pandas (5 minutes, maille nominale des capteurs).
    max_fill_periods : int
        Nombre max de pas 5 min à interpoler. 2 = 10 min : un point
        sauté (dt ≈ 600 s) est comblé ; un trou de plusieurs heures non.
    """

    raw_dir: Path = REPO_ROOT / "data" / "raw"
    processed_dir: Path = REPO_ROOT / "data" / "processed"
    resample_rule: str = "5min"
    max_fill_periods: int = 2
    living_room_file: str = "temperature_livingroom.csv"
    outdoor_file: str = "temperature_outside.csv"
    heating_file: str = "temperature_heating_system.csv"
    pv_file: str = "pv_production_load.csv"
    long_gap_seconds: float = 3600.0


@dataclass(frozen=True)
class ControlConfig:
    """Réglages du MPC et du bang-bang (pas dans le code métier).

    Parameters
    ----------
    t_set, t_min, t_max : float
        Consigne et bande de confort (°C).
    horizon_hours, block_minutes : float
        Horizon de prédiction et durée d'un mouvement (move blocking).
    q_track, q_band, r_rel : float
        Poids : suivi de consigne, hors-bande, effort (relatif à P_max²).
    n_hours : float
        Durée de la comparaison en boucle fermée.
    p_max_margin : float
        ``P_max`` = marge × puissance de maintien au T_ext le plus froid.
    seed : int
        Bruit du plant (même graine pour les deux contrôleurs).
    """

    t_set: float = 20.0
    t_min: float = 19.5
    t_max: float = 21.0
    horizon_hours: float = 6.0
    block_minutes: float = 30.0
    q_track: float = 1.0
    q_band: float = 20.0
    r_rel: float = 0.04
    n_hours: float = 48.0
    p_max_margin: float = 1.4
    seed: int = 0
