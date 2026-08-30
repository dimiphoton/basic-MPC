# Roadmap

Grey-box RC, Kalman, plant simulé, MPC vs bang-bang.

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
- [ ] Identification R2C2 et comparaison multi-horizon (`feature/comparaison-r1c1-r2c2`)
- [ ] MPC à horizon glissant vs thermostat bang-bang (`feature/mpc-vs-bang-bang`)
- [ ] Polish portfolio : README, slides, limites (`feature/portfolio-slides`)
