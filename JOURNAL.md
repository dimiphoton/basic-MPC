# Journal de développement

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
