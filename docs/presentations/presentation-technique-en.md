---
marp: true
theme: portfolio
paginate: true
---

<!-- _class: cover -->
<!-- _paginate: false -->

![bg brightness:0.40](../../pictures/presentations/photos/hero.jpg)

# Can we close the loop
# without cheating?

Machine learning · Buildings · Python / NumPy / SciPy

Instrumented house · 2020 · ~5 min

---

<!-- _class: split -->

![bg left:46%](../../pictures/presentations/photos/action.jpg)

# A one-step RMSE
# does not hire a model.

R1C1 and R2C2 tie at 1 h: 0.56 °C.

**At 24 h the extra state wins 0.2 °C. That is the hiring bar for MPC.**

---

<!-- _class: split -->

![bg left:46%](../../pictures/presentations/photos/cta.jpg)

# The artefact
# is the controller.

Not a dashboard. Not an identification memo.

Consultancies and operators get a command, a comfort band, a plant that is not the fitted RC.

---

<!-- _class: full -->

![bg brightness:0.38](../../pictures/presentations/photos/cta.jpg)

# Fast air.
# Hidden mass.

The sensor sees air, stepped at 0.1 °C.
Walls are a memory the thermostat never uses.

---

<!-- _class: split -->

![bg left:46%](../../pictures/presentations/photos/action.jpg)

# What the pipeline
# refuses to invent.

Living room + outdoor, 5-minute grid. Fill at most 10 minutes.

A 12-hour hole stays NaN. 0.1 °C vs 1 °C is the sensor, not the RC.

No measured watts: `P` is water–air lift if the zone calls, `S` is PV.

---

<!-- _class: split -->

![bg left:46%](../../pictures/presentations/photos/hero.jpg)

# What we isolate.

PEM on Kalman innovations, 70/30 split, 50 winter days for the fit.

Quantization is not a time constant.
What remains: `R`, `C`, and proxy gains.

---

<!-- _class: chart -->

## Kalman by hand: predict, innovate, update.

![w:920](../../pictures/presentations/schema-kalman.png)

---

<!-- _class: dark -->

# Scope.

Identify an RC on real sensors.
Test the MPC on a literature plant.

Plant solar also hits the mass. This is not a live-house trial.

---

<!-- _class: chart -->

## The second state shows up after 6 h, not at the next step.

![w:920](../../pictures/presentations/i1-rmse-horizons.png)

---

<!-- _class: full -->

![bg brightness:0.38](../../pictures/presentations/photos/hero.jpg)

# 0 h outside the band
# after a 2 h heat-up.

Bang-bang: 7 h. Same weather, same plant, same seed.

---

<!-- _class: chart -->

## On/off versus a 6-hour receding move.

![w:980](../../pictures/presentations/s5-commande-p.png)

---

<!-- _class: split -->

![bg left:40%](../../pictures/presentations/photos/action.jpg)

# Receding horizon.

Six hours ahead. Piecewise-constant moves of 30 minutes.

Kalman at every step. SciPy, not cvxpy. Future weather is known (oracle).

---

<!-- _class: split -->

![bg left:40%](../../pictures/presentations/photos/cta.jpg)

# Not circular.

Same `u`, two physics: the plant has `α_s,mass`.

The house R2C2 (τ ~ 140 h) cannot drive this plant.
The internal model keeps the identification structure, no solar on the mass.

---

<!-- _class: dark -->

# Why not the salon R2C2.

Both time constants sit near 140 h.
That is not fast air plus slow mass.

We state it. We do not feed it to the plant controller.

---

<!-- _class: dark -->

# Where it breaks.

Perfect weather forecasts (oracle).

`P` is not watts.

One house, one winter fit, not a stock of buildings.

---

<!-- _class: cta -->

![bg brightness:0.30](../../pictures/presentations/photos/cta.jpg)

# Reproduce.

[Slides](https://dimiphoton.github.io/basic-MPC/slides/presentation-technique-en.html)
[Repo](https://github.com/dimiphoton/basic-MPC)

`python -m basic_mpc mpc-vs-bang-bang`

`pytest`

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?logo=scipy&logoColor=white)
