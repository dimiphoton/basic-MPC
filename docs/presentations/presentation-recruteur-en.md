---
marp: true
theme: portfolio
paginate: true
---

<!-- _class: cover -->
<!-- _paginate: false -->

![bg brightness:0.40](../../pictures/presentations/photos/hero.jpg)

# Can we heat
# a house
# by looking ahead?

Machine learning · Buildings

Instrumented house · 2020 · ~5 min

---

<!-- _class: split -->

![bg left:46%](../../pictures/presentations/photos/action.jpg)

# Comfort is lost
# at night.

A thermostat turns on too late, too hard, then too long.

**You pay for heat. You also sit in the cold.**

---

<!-- _class: split -->

![bg left:46%](../../pictures/presentations/photos/cta.jpg)

# Who has
# a call to make.

Consultancies. Building operators. Flexibility aggregators.

**Heat on a threshold, or with a model that sees the night coming.**

---

<!-- _class: chart -->

## Simulated house: air moves first, the walls follow.

![w:980](../../pictures/presentations/h2-reponse-indicielle.png)

---

<!-- _class: split -->

![bg left:46%](../../pictures/presentations/photos/physique.jpg)

# How we read
# the numbers.

One house, 2020, a reading every five minutes.

The 21.8 °C indoor figure is not the air: the sensor steps by tenths of a degree.

Heating is not measured in watts. We rebuild it from water and air, only when the room asks.

---

<!-- _class: dark -->

# This project is not.

Not a magic thermostat fitted on the same model it then controls.

Not a live-house trial.

**A coded house, slightly different on purpose, to test the controller.**

---

<!-- _class: full -->

![bg brightness:0.38](../../pictures/presentations/photos/hero.jpg)

# 1.8 h below comfort.
# The thermostat: 19.8 h.

48 simulated hours. 18 h less time in the cold. Proxy bill ~€16 either way.

---

<!-- _class: chart -->

## 48 simulated hours: 18 h less time in the cold, same proxy bill.

![w:980](../../pictures/presentations/s6-confort-conso.png)

---

<!-- _class: chart -->

## The controller is already rising before 7 a.m. The thermostat waits.

![w:980](../../pictures/presentations/s4-mpc-vs-bang-bang.png)

---

<!-- _class: chart -->

## We did not replay the controller on the model it learned.

![w:980](../../pictures/presentations/s3-plant-vs-r2c2.png)

---

<!-- _class: actions -->

![bg right:38%](../../pictures/presentations/photos/action.jpg)

# Monday.

**Operator** — keep the thermostat, but count hours too cold.

**Consultancy** — a small model is enough to look ahead; the bill barely moves, the cold does.

Not a product. A method, with the limits on the slide.

---

<!-- _class: cta -->

![bg brightness:0.30](../../pictures/presentations/photos/cta.jpg)

# Your turn.

[Open the slides](https://dimiphoton.github.io/basic-MPC/slides/presentation-recruteur-en.html)

[Open the lesson](https://dimiphoton.github.io/basic-MPC/simulator/)

[Source code](https://github.com/dimiphoton/basic-MPC)
