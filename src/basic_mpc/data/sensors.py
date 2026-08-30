"""Modèle de capteur : observation ≠ état thermique."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SensorModel:
    """Équation d'observation ``y = h(x) + v``.

    Pour une température d'air mesurée directement, ``h(x) = T_air``.
    ``v`` regroupe bruit et **quantification** (pas du capteur), à ne
    pas interpréter comme une dynamique des murs.

    Parameters
    ----------
    name : str
        Identifiant du capteur (ex. ``livingroom``).
    resolution : float
        Plus petit pas observé sur les mesures.
    unit : str
        Unité (``celsius``, ``bar``…).
    quantity : str
        Grandeur physique mesurée.
    """

    name: str
    resolution: float
    unit: str
    quantity: str = "temperature"

    def observation_equation(self) -> str:
        """Forme symbolique pour le README / les slides.

        Returns
        -------
        str
            Équation d'observation en texte.
        """
        return f"y = T_air + v  (quantification {self.resolution:g} {self.unit})"


def infer_resolution(values: np.ndarray, decimals: int = 6) -> float:
    """Infère le pas de quantification à partir des valeurs uniques.

    Parameters
    ----------
    values : ndarray
        Mesures brutes (1d).
    decimals : int
        Arrondi anti-bruit flottant avant de calculer les écarts.

    Returns
    -------
    float
        Plus petit écart strictement positif.

    Raises
    ------
    ValueError
        S'il n'y a pas assez de valeurs distinctes.
    """
    uniq = np.unique(np.round(np.asarray(values, dtype=float), decimals))
    uniq = uniq[np.isfinite(uniq)]
    if uniq.size < 2:
        msg = "pas assez de valeurs distinctes pour inférer la résolution"
        raise ValueError(msg)
    diffs = np.diff(np.sort(uniq))
    # Écarts nuls après arrondi : on les ignore
    positif = diffs[diffs > 10 ** (-decimals)]
    if positif.size == 0:
        msg = "toutes les valeurs sont identiques après arrondi"
        raise ValueError(msg)
    return float(np.round(positif.min(), decimals))


def quantize_measurement(value: float, resolution: float) -> float:
    """Projette une température vraie sur la grille du capteur.

    Parameters
    ----------
    value : float
        Température physique (avant mesure).
    resolution : float
        Pas de quantification (> 0).

    Returns
    -------
    float
        Valeur telle qu'un capteur à pas ``resolution`` l'écrirait.
    """
    if resolution <= 0:
        msg = "la résolution du capteur doit être strictement positive"
        raise ValueError(msg)
    return float(np.round(value / resolution) * resolution)
