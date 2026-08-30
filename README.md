# Grey-box thermal model and heating MPC

| | |
|---|---|
| **Role** | Machine learning |
| **Domain** | Buildings |
| **Stack** | ![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white) ![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?logo=scipy&logoColor=white) |
| **Level** | Intermediate |
| **Status** | In progress |

Machine learning · Buildings · Python / NumPy / SciPy

## Objective

Infer a credible RC thermal model from real room sensors (explicit sensor
noise, Kalman filter), compare **R1C1** vs **R2C2**, then use the retained
model inside a heating MPC. The plant is not given: it is coded separately
from the identified model so closed-loop tests are not circular.

## Data

Instrumented house, 2020, indoor temperatures ~5 min (per room, plus
setpoint), outdoor temperature, hydronic heating (water temperature and
pressure), PV production and electrical load. Heating *power* and
irradiance are not measured; they are reconstructed as model inputs.

## Result

Work in progress — see `ROADMAP.md`. Target: a documented R1C1/R2C2
comparison, a Kalman state estimator, and an MPC that beats bang-bang
on the simulated plant.

## Reproduce

```bash
pip install -e .
python -m basic_mpc.cli --help
```

## Repo structure

```
brief/          # identity, objective, thermal briefs
data/raw/       # sensor CSV (do not edit)
src/basic_mpc/  # package (data, features, models, CLI)
tests/
docs/presentations/   # Marp sources
```

French notes: `ROADMAP.md`, `JOURNAL.md`, `docs/decisions.md`.

## Presentations

Two audiences × two languages (Marp theme `portfolio`, HTML on GitHub Pages).
The recruiter deck is a ~6-minute pitch; the technical deck is a ~12-minute
deep dive. They may diverge a lot — the bar is attractive and informative
for each audience, not a mirrored pair of slides.

- [Recruiter overview (EN)](docs/slides/presentation-recruteur-en.html)
- [Technical deep dive (EN)](docs/slides/presentation-technique-en.html)
- [Présentation grand public (FR)](docs/slides/presentation-recruteur-fr.html)
- [Présentation technique (FR)](docs/slides/presentation-technique-fr.html)
