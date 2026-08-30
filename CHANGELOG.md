# Changelog

## [Non publié]

- MPC vs bang-bang (48 h, plant littérature) : 0 h hors bande après 2 h
  contre 7 h pour le thermostat ; proxy conso −3 %. CLI
  `mpc-vs-bang-bang`. Le modèle interne n'est pas le R2C2 maison.
- Schémas RC/Kalman générés (PNG+PDF) ; comparaison R1C1/R2C2 :
  RMSE 24 h 1,73 vs 1,93 °C ; le second état n'est pas une masse rapide.
- Identification R1C1 (PEM + Kalman) : τ ≈ 104 h, RMSE 0,56 °C / 1 h
  et 1,96 °C / 24 h sur le test. Figures I3, I4, S1.
- Catalogue des figures (`docs/visualisations.md`) ; choix slides plus tard.
- Plant simulé : R2C2 littérature + solaire sur la masse, capteur 0,1 °C.
- Entrées : `P` (écart eau/air × appel de zone) et `S` (PV), pas des grandeurs SI.
- Prétraitement : maille 5 min, modèle de capteur (0,1 °C salon / 1 °C extérieur).
- Cadrage : identité ML · Bâtiment, roadmap R1C1/R2C2 + Kalman + plant + MPC.
- Initialisation du projet à partir du template portfolio.
- Archivage de l'ancien code sur la branche `old`.
