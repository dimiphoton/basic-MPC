---
marp: true
theme: portfolio
paginate: true
---

<!-- _class: cover -->
<!-- _paginate: false -->

<!-- Photo: pictures/presentations/photos/hero.png -->
<!-- ![bg brightness:0.40](../../pictures/presentations/photos/hero.png) -->

# Can an identifiable RC model
# drive a heating MPC?

Machine learning · Buildings · Python / NumPy / SciPy

Instrumented house · 2020 · ~5 min

---

<!-- _class: split -->

<!-- ![bg left:46%](../../pictures/presentations/photos/motivation.png) -->

# [Why measure
# this.]

[Quantified stake if we have it. Cost of a bad call.]

**[What is missing today to decide.]**

---

<!-- _class: split -->

<!-- ![bg left:46%](../../pictures/presentations/photos/hero.png) -->

# [Who consumes
# the output.]

[Agency / regulator]. [Operator / insurer / consultancy].

The deliverable: [indicator, view, recommendation], not a report.

---

<!-- _class: full -->

<!-- ![bg brightness:0.38](../../pictures/presentations/photos/physique.png) -->

# [Mechanism.]

[Physics or process: rain → waterlogged → yield, load curve, Espec…]

---

<!-- _class: split -->

<!-- ![bg left:46%](../../pictures/presentations/photos/motivation.png) -->

# [Data-processing
# logic.]

Living room + outdoor, 5-minute grid. Interpolation capped at 10 minutes.

A 12-hour hole stays NaN. 0.1 °C (indoor) vs 1 °C (outdoor) quantization is the sensor model, not the RC dynamics.

---

<!-- _class: split -->

<!-- ![bg left:46%](../../pictures/presentations/photos/physique.png) -->

# [What we isolate.]

Strip out [confounder]. What remains is [target].

Not [what we are not claiming].

---

<!-- _class: dark -->

# Scope.

We do [orientation: diagnosis / identifiability / decision].

We are not [2050 scenario / flashy model / a map with no GIS].

---

<!-- _class: chart -->

[Baseline / trend / raw data — sentence headline.]

<!-- ![w:920](../../pictures/presentations/baseline-en.png) -->

---

<!-- _class: full -->

<!-- ![bg brightness:0.38](../../pictures/presentations/photos/physique.png) -->

# [Main result]
# [metric + n]

---

<!-- _class: chart -->

[Chart for *this* technical story — not necessarily the recruiter one.]

<!-- ![w:980](../../pictures/presentations/key-chart-en.png) -->

---

<!-- _class: split -->

<!-- ![bg left:40%](../../pictures/presentations/photos/hero.png) -->

# [Robustness.]

[Spatial / years / n]. [What is not independent.]

<!-- ![w:480](../../pictures/presentations/map-or-robustness-en.png) -->

---

<!-- _class: chart -->

Why not [flashy model]? n = […]. [Chosen model + validation.]

<!-- ![w:640](../../pictures/presentations/validation-en.png) -->

---

<!-- _class: dark -->

# Where it breaks.

[Limit 1.]

[Limit 2.]

[Limit 3. Correlation ≠ causation if relevant.]

---

<!-- _class: cta -->

<!-- ![bg brightness:0.30](../../pictures/presentations/photos/cta.png) -->

# Reproduce.

[Explore online](../explore-en.html)

`python -m basic_mpc run`

`python -m basic_mpc dashboard`

<!-- Stack badges here, not on the recruiter cover. -->
