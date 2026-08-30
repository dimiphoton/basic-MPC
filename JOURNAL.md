# Journal de développement

## 2026-08-30 — R2C2 vs R1C1 et schémas de publication

- Schémas matplotlib (PNG + PDF) : R1C1, R2C2, plant (`α_s,mass`), Kalman,
  planche famille. CLI : `python -m basic_mpc draw-schemas`.
- R2C2 identifié (PEM Nelder-Mead, même fenêtre que le R1C1). RMSE test
  0,56 °C / 1 h (égal) et 1,73 °C / 24 h contre 1,93 °C pour le R1C1.
  Les deux τ (~139 h et ~148 h) sont lentes ; `R_ae` sature. Le 2e état
  n'est pas une masse rapide clairement séparée.
- Figures I1, I2, Z1, Z2, S2, S3, I5. CLI : `compare-r1c1-r2c2`.

## 2026-08-30 — Identification R1C1 + Kalman

- Kalman linéaire à la main (chemin 1D scalaire). PEM : vraisemblance
  des innovations. Split temporel 70/30 ; fit sur les 50 derniers jours
  du train (16 déc. 2020 – 4 fév. 2021), métriques sur fév.–mai 2021.
- τ ≈ 104 h ; RMSE test 0,56 °C à 1 h et 1,96 °C à 24 h. Gain solaire
  quasi nul (fenêtre d'hiver). ACF des innovations ~2 h → le R1C1 rate
  une dynamique, justification du R2C2.
- Figures I3, I4, S1 dans `pictures/experiments/`. CLI :
  `python -m basic_mpc identify-r1c1`.

## 2026-08-30 — Catalogue des visualisations

- Menu complet dans `docs/visualisations.md` (impédance, ident., plant/MPC).
- Production au fil des features ; sélection des slides au polish.

## 2026-08-30 — Simulateur plant

- R2C2 littérature avec solaire aussi sur la masse (`alpha_s_mass`) :
  distinct du R2C2 d'identification (solaire sur l'air seulement).
- Capteur : bruit puis quantification 0,1 °C. État vrai `T_masse` caché.
- CLI : `python -m basic_mpc simulate-plant`.
- Scénario 48 h : y entre 10,3 et 17,5 °C ; masse ~1 K au-dessus de l'air.

## 2026-08-30 — Entrées chauffage et solaire

- `P = max(T_eau - T_air, 0) * 1_{T_air < consigne}` (kelvin, pas des watts).
- `S` = somme des 3 phases PV, négatifs à 0 (proxy, pas des W/m²).
- CLI : `python -m basic_mpc build-inputs`.
- Sur une année : appel de zone ~6 % du temps ; `P` moyen ≈ 22 K quand > 0.

## 2026-08-30 — Prétraitement et modèle de capteur

- Maille cible 5 min ; interpolation limitée à 10 min (un point sauté).
- Salon : quantification 0,1 °C. Extérieur : 1 °C. Équation `y = T_air + v`.
- CLI : `python -m basic_mpc preprocess`.
- 14 longs trous (> 1 h) côté brut ; 660 NaN restants après maille 5 min
  (dont décalage de début salon / extérieur). Les longs trous restent NaN.

## 2026-08-30 — Cadrage

- Identité : Machine learning · Bâtiment · Python / NumPy / SciPy.
- Objectif : grey-box R1C1 vs R2C2, modèle de capteur, Kalman, plant
  simulé distinct, MPC vs bang-bang. v1 = salon.
- Roadmap en 7 étapes nommées (branches `feature/...` dans `ROADMAP.md`).
- Briefs utilisateur versionnés dans `brief/`.

## 2026-08-30 — Initialisation du projet

- Ancien code archivé sur la branche `old` (ne plus le modifier depuis
  `main`).
- Repo recalé sur le template portfolio : package `basic_mpc`, brief,
  présentations Marp, structure `src/` / `tests/` / `docs/`.
- Données brutes déjà présentes dans `data/raw/` (températures par pièce,
  chauffage, extérieur, PV) — conservées, pas retravaillées.
