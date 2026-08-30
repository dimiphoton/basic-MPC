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

<!-- _class: full -->

![bg brightness:0.38](../../pictures/presentations/photos/cta.jpg)

# Air heats fast.
# Walls remember.

Two rooms at 20 °C do not share the same tomorrow
if one has cold walls.

---

<!-- _class: split -->

![bg left:46%](../../pictures/presentations/photos/action.jpg)

# How we read
# the numbers.

One house, 2020, a 5-minute grid — sometimes one sample skipped.

The 21.8 °C indoor reading is not the air: the sensor steps by 0.1 °C.

Heating is not in watts: we build a water–air gap, only when the room asks.

---

<!-- _class: dark -->

# This project is not.

Not a magic thermostat on the same model we just fitted.

Not measured watts, and not a live-house trial.

**A coded plant, slightly different on purpose, to test the controller.**

---

<!-- _class: full -->

![bg brightness:0.38](../../pictures/presentations/photos/hero.jpg)

# Zero hours
# outside comfort.

The thermostat: seven.

---

<!-- _class: chart -->

## Same weather: comfort moves, the bill barely does.

![w:980](../../pictures/presentations/s6-confort-conso.png)

---

<!-- _class: chart -->

## The thermostat hunts. Looking ahead holds 20 °C.

![w:980](../../pictures/presentations/s4-mpc-vs-bang-bang.png)

---

<!-- _class: split -->

![bg left:40%](../../pictures/presentations/photos/hero.jpg)

# This is not
# a drawing trick.

We did not replay the controller on the model it learned.

The simulated house lets the sun hit the walls too.
That is not the house we identified.

---

<!-- _class: actions -->

![bg right:38%](../../pictures/presentations/photos/action.jpg)

# Monday.

**Operator** — keep the thermostat, but count hours outside the band.

**Consultancy** — a small model is enough to look ahead; the win is not a big saving.

Not a product. A method, with the limits on the slide.

---

<!-- _class: cta -->

![bg brightness:0.30](../../pictures/presentations/photos/cta.jpg)

# Your turn.

[Open the slides](https://dimiphoton.github.io/basic-MPC/slides/presentation-recruteur-en.html)

[Source code](https://github.com/dimiphoton/basic-MPC)
