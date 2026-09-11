# Changelog

## [Non publié]

- Note de méthode : split d'identification chronologique, pas de
  semaines mélangées ni RC ouvré / week-end
  (`docs/identification-split.md`).
- Labo Streamlit dynamique : météo multi-jours tirée, burn-in 24 h
  (maison habitée), une stratégie, graphes datetime (air / murs /
  consigne / P / facture). Pages = leçon RC (équilibre + corrélation
  croisée), plus d'arène figée.
- Simulateur visiteur (Pages + Streamlit) : RC, déphasage, vecteur Z
  fitté, arène de stratégies. CLI `export-simulator`.
- MPC v1.1 : consigne thermostat, bande n = 1 °C, J = facture HP/HC +
  inconfort. 48 h : **1,8 h** sous T_conf vs **19,8 h** (hystérésis),
  facture ~16 € des deux côtés. CLI `mpc-cout-consigne`. Heures d'inconfort
  au-delà de 0,1 °C (capteur).
- Polish portfolio : 4 decks Marp, README v1.0, limites, liens GitHub Pages.
- MPC vs bang-bang (48 h, plant littérature) : 0 h hors bande après 2 h
  contre 7 h pour le thermostat ; proxy conso −3 %. CLI
  `mpc-vs-bang-bang`. Le modèle interne n'est pas le R2C2 maison.
- Schémas RC/Kalman générés (PNG+PDF) ; comparaison R1C1/R2C2 :
  RMSE 24 h 1,73 vs 1,93 °C ; le second état n'est pas une masse rapide.
- Identification R1C1 (PEM + Kalman) : τ ≈ 104 h, RMSE 0,56 °C / 1 h
  et 1,96 °C / 24 h sur le test. Figures I3, I4, S1.
- Catalogue des figures (`docs/visualisations.md`).
- Plant simulé : R2C2 littérature + solaire sur la masse, capteur 0,1 °C.
- Entrées : `P` (écart eau/air × appel de zone) et `S` (PV), pas des grandeurs SI.
- Prétraitement : maille 5 min, modèle de capteur (0,1 °C salon / 1 °C extérieur).
- Cadrage : identité ML · Bâtiment, roadmap R1C1/R2C2 + Kalman + plant + MPC.
- Initialisation du projet à partir du template portfolio.
- Archivage de l'ancien code sur la branche `old`.
