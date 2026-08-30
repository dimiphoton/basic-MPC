# Objectif du projet

- **But** : inférer un modèle thermique grey-box (RC) crédible à partir
  de capteurs réels, avec modèle de capteur et filtre de Kalman, puis
  s'en servir comme modèle interne d'un MPC du chauffage. Le simulateur
  (plant) n'est pas fourni : il est à coder, distinct du modèle identifié.
- **Origine** : reprise d'un projet ULG (code archivé sur la branche
  `old`). Briefs dans `brief/01-thermique-grey-box-mpc.md` et
  `brief/Cadrage de la modélisation thermique.md`.
- **Contraintes de départ** : données imposées (températures par pièce,
  circuit de chauffage, extérieur, PV/charge). Pas de puissance de
  chauffage ni d'ensoleillement mesurés tels quels. Kalman à la main.
  Comparer **R1C1** (baseline) et **R2C2** (cible si les horizons longs
  le justifient).

Décisions de cadrage (2026-08-30) :

- v1 **mono-zone** : salon (`temperature_livingroom.csv`).
- Entrée chauffage `P` **construite** (eau du circuit + consignes),
  documentée comme n'étant pas une puissance mesurée.
- PV comme **proxy solaire** possible, à valider sur les données.
- Identification sur le **réel** ; boucle fermée MPC sur le **plant
  simulé** (évite la circularité).
- Hors v1 : Airflow, AWS S3, MLflow, multi-zone. Docker seulement
  en polish si le temps le permet.
- Contrôle : le livrable est le MPC ; le métier affiché reste le ML
  (baseline, validation, incertitude).

Métier, domaine et stack : `brief/identite.md`.
