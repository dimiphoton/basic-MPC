# Roadmap

Grey-box RC, Kalman, plant simulé, MPC vs bang-bang.
v1.1 : problème de contrôle (`brief/controle-mpc.md`), puis dashboard.

Une case = une branche. Le nom de branche est dans la case ; ne pas
en inventer un autre. Une seule case par tour d'autopilot.

Figures : catalogue dans `docs/visualisations.md`. On les génère dans
`pictures/experiments/` au fil de l'eau. Le choix RH/technique se fait
à `feature/portfolio-slides`.

- [x] Cadrage identité, objectif, roadmap (`feature/cadrage-grey-box-mpc`)
- [x] Prétraitement des séries et modèle de capteur (`feature/pretraitement-capteurs`)
- [x] Construction des entrées chauffage et solaire (`feature/entrees-chauffage-solaire`)
- [x] Simulateur plant distinct du modèle identifié (`feature/simulateur-plant`)
- [x] Identification R1C1 + filtre de Kalman (`feature/identification-r1c1`)
- [x] Identification R2C2 et comparaison multi-horizon (`feature/comparaison-r1c1-r2c2`)
- [x] MPC à horizon glissant vs thermostat bang-bang (`feature/mpc-vs-bang-bang`)
- [x] Polish portfolio : README, slides, limites (`feature/portfolio-slides`)
- [x] Cadrage contrôle : consigne, bande n, J euros (`brief/controle-mpc.md`)
- [x] MPC : bande n, coût €, baseline hystérésis (`feature/mpc-cout-consigne`)
- [x] Simulateur visiteur : RC + déphasage, vecteur \(Z\) fitté, stratégies
  ludiques — GitHub Pages **et** Streamlit (`feature/dashboard-simulateur`)
- [x] Labo dynamique : météo tirée, burn-in, Streamlit ; Pages = leçon RC
  (`feature/sim-dynamique`)
- [x] Leçon scolaire : schémas R1C1/R2C2, capteur, Kalman, notebooks
  (`feature/lecon-notebooks`)
