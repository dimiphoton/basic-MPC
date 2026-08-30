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

Instrumented house, 2020 (May–May), indoor temperatures at a nominal
**5 min** grid (occasional skipped samples; longest gap ~13 h). Living-room
sensor quantized at **0.1 °C**, outdoor at **1 °C**. There is **no measured
heating power nor irradiance**. Model inputs are constructed and documented:

- `P = max(T_water - T_air, 0)` when `T_air < setpoint`, else 0 (kelvin, not W)
- `S` = sum of three PV phases, negatives clipped (proxy, not W/m²)

## Result

R1C1 baseline (PEM + Kalman, held-out Feb–May 2021): **RMSE 0.56 °C at 1 h**,
**2.0 °C at 24 h**. Time constant ~104 h (one slow lump). Solar gain is
essentially unidentified on the winter PEM window; heating gain is not.
Innovations stay correlated for ~2 h — a second state (R2C2) is the next
test. Figures: `pictures/experiments/`. Full comparison vs R2C2 is the
next roadmap step.

## Reproduce

```bash
pip install -e ".[dev]"
python -m basic_mpc preprocess
python -m basic_mpc build-inputs
python -m basic_mpc simulate-plant
python -m basic_mpc identify-r1c1
pytest
```

`identify-r1c1` fits a one-state RC on a winter slice of the train split
(Kalman prediction-error), then scores 1–24 h forecasts on the test split.
`simulate-plant` writes a 48 h trajectory on the **literature plant**
(solar also hits the thermal mass — not the RC we identify).

## Repo structure

```
brief/          # identity, objective, thermal briefs
data/raw/       # sensor CSV (do not edit)
src/basic_mpc/  # package (data, features, models, CLI)
tests/
docs/presentations/   # Marp sources
```

French notes: `ROADMAP.md`, `JOURNAL.md`, `docs/decisions.md`,
`docs/visualisations.md`.

## Presentations

Two audiences × two languages (Marp theme `portfolio`, HTML on GitHub Pages).
The recruiter deck is a ~6-minute pitch; the technical deck is a ~12-minute
deep dive. They may diverge a lot — the bar is attractive and informative
for each audience, not a mirrored pair of slides.

- [Recruiter overview (EN)](docs/slides/presentation-recruteur-en.html)
- [Technical deep dive (EN)](docs/slides/presentation-technique-en.html)
- [Présentation grand public (FR)](docs/slides/presentation-recruteur-fr.html)
- [Présentation technique (FR)](docs/slides/presentation-technique-fr.html)
