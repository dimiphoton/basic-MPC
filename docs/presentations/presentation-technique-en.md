---
marp: true
theme: portfolio
paginate: true
---

<!-- _class: cover -->
<!-- _paginate: false -->

![bg brightness:0.40](../../pictures/presentations/photos/hero.jpg)

# Can a small model
# heat a house
# ahead of time?

Machine learning · Buildings · Python / NumPy / SciPy

Instrumented house · 2020 · ~5 min

---

<!-- _class: split -->

![bg left:46%](../../pictures/presentations/photos/action.jpg)

# One state is enough
# for this room.

At 1 h: 0.56 °C with one state, 0.63 °C with two.

**At 24 h: 1.93 °C and 1.95 °C. Inside a room's time-constant box, the extra state does not win.**

---

<!-- _class: split -->

![bg left:46%](../../pictures/presentations/photos/cta.jpg)

# The artefact
# is the controller.

Not a dashboard. Not an identification memo.

Consultancies and operators get a thermostat setpoint, a comfort band, a simulated house that is not the fitted model.

---

<!-- _class: chart -->

## Simulated house: air moves first, the walls follow.

![w:980](../../pictures/presentations/h2-reponse-indicielle.png)

---

<!-- _class: split -->

![bg left:46%](../../pictures/presentations/photos/physique.jpg)

# Where the numbers
# come from.

Living room and outdoor, every five minutes. A gap longer than ten minutes stays a gap.

The sensor steps by tenths of a degree — that is not a time constant.

No measured watts: we rebuild heating from water and air, only when the room asks.

---

<!-- _class: chart -->

## Kalman by hand: predict, compare, correct.

![w:920](../../pictures/presentations/schema-kalman.png)

---

<!-- _class: dark -->

# Scope.

We fit the model on real sensors.

We test the controller on a coded house.

Sunlight there hits the walls too. This is not a live-house trial.

---

<!-- _class: chart -->

## One state stays ahead, from 1 h to 24 h.

![w:920](../../pictures/presentations/i1-rmse-horizons.png)

---

<!-- _class: chart -->

## Same weather, two physics: the controller is not scored on itself.

![w:980](../../pictures/presentations/s3-plant-vs-r2c2.png)

---

<!-- _class: full -->

![bg brightness:0.38](../../pictures/presentations/photos/hero.jpg)

# 1.8 h below comfort.
# Thermostat: 19.8 h.

48 h, simulated house, weather known ahead. Proxy bill ~€16. Gap: 18 h.

---

<!-- _class: chart -->

## The MPC is up before 7 a.m. The thermostat waits for 7 a.m.

![w:980](../../pictures/presentations/s4-mpc-vs-bang-bang.png)

---

<!-- _class: chart -->

## The decision is a thermostat setpoint, not watts.

![w:980](../../pictures/presentations/s5-commande-p.png)

---

<!-- _class: chart -->

## A one-degree band: the larger the gap, the harder we heat.

![w:920](../../pictures/presentations/h1-loi-bande-n.png)

---

<!-- _class: chart -->

## If weather arrives six hours late, comfort drops.

![w:980](../../pictures/presentations/h9-oracle-vs-retard.png)

---

<!-- _class: dark -->

# The living room does not
# split air and walls.

6 h and 8 h, both on the edge of the allowed box.

The controller is scored on another house: air ~3 h, walls ~11 h.

---

<!-- _class: dark -->

# Where it breaks.

Weather known in advance (oracle).

Heating is not watts.

One house, one winter fit, not a stock of buildings.

---

<!-- _class: cta -->

![bg brightness:0.30](../../pictures/presentations/photos/cta.jpg)

# Reproduce.

[Slides](https://dimiphoton.github.io/basic-MPC/slides/presentation-technique-en.html)
[Repo](https://github.com/dimiphoton/basic-MPC)

`notebooks/03-mpc-cout-consigne.ipynb`

`pytest`

[Lesson](https://dimiphoton.github.io/basic-MPC/simulator/)

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?logo=scipy&logoColor=white)
