# Décisions

| Date | Décision | Alternative envisagée | Raison |
|---|---|---|---|
| 2026-08-30 | Métier Machine learning, domaine Bâtiment, stack Python / NumPy / SciPy | Domaine Énergie ; stack avec cvxpy / MLflow | Le recruteur coche un poste ML (baseline, validation). L'objet est la maison. SciPy suffit en v1. |
| 2026-08-30 | Noms **R1C1** (baseline) et **R2C2** (cible) | R1C / R2C comme dans le cadrage manuscrit | Les équations du brief sont 1R1C et 2R2C. Un seul vocabulaire dans le code et les slides. |
| 2026-08-30 | v1 mono-zone salon | Moyenne toutes pièces ; multi-zone d'emblée | Un capteur, une consigne, moins de fuites d'identifiabilité. Multi-zone hors v1. |
| 2026-08-30 | Construire `P` à partir du circuit d'eau et des consignes | Attendre une puissance mesurée (absente) | Sans entrée chauffage, ni identification ni MPC. Documenter que ce n'est pas des watts. |
| 2026-08-30 | Plant **codé**, distinct du modèle identifié | Rejouer le MPC sur le RC appris | Circularité : le contrôleur se prédit lui-même. Le brief phare l'interdit comme validation. |
| 2026-08-30 | Kalman à la main | `filterpy` / `pykalman` | Preuve de compréhension ; test unitaire sur cas synthétique. |
| 2026-08-30 | Hors v1 : Airflow, S3, MLflow | Suivre le brief phare à la lettre | Dilue le récit Kalman / RC / MPC. Log d'expériences : `experiments/runs.jsonl`. |
