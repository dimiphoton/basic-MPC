"""Tests du modèle de capteur (quantification ≠ physique)."""

import numpy as np
import pytest

from basic_mpc.data.sensors import SensorModel, infer_resolution


def test_infer_resolution_dixieme() -> None:
    """Un capteur à 0.1 °C doit donner 0.1, pas une dynamique."""
    valeurs = np.array([20.0, 20.1, 20.2, 20.1, 19.9])
    assert infer_resolution(valeurs) == pytest.approx(0.1)


def test_infer_resolution_entier() -> None:
    """Extérieur entier : pas de 1 °C."""
    valeurs = np.array([14.0, 15.0, 14.0, 13.0])
    assert infer_resolution(valeurs) == pytest.approx(1.0)


def test_infer_resolution_une_seule_valeur() -> None:
    """Cas limite : pas assez de valeurs distinctes."""
    with pytest.raises(ValueError, match="distinctes"):
        infer_resolution(np.array([21.0, 21.0, 21.0]))


def test_observation_equation_mentionne_la_quantification() -> None:
    """L'équation d'observation sépare T_air et v."""
    capteur = SensorModel(
        name="outdoor",
        resolution=1.0,
        unit="celsius",
    )
    texte = capteur.observation_equation()
    assert "T_air" in texte
    assert "1" in texte
