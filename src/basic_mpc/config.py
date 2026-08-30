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
    long_gap_seconds: float = 3600.0
