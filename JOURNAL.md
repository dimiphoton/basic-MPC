# Journal de développement

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
