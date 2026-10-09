---
marp: true
theme: portfolio
paginate: true
---

<!-- _class: cover -->
<!-- _paginate: false -->

![bg brightness:0.40](../../pictures/presentations/photos/hero.jpg)

# Peut-on chauffer
# une maison
# en anticipant ?

Machine learning · Bâtiment

Maison instrumentée · 2020 · ~5 min

---

<!-- _class: split -->

![bg left:46%](../../pictures/presentations/photos/action.jpg)

# Le confort
# se joue la nuit.

Un thermostat allume trop tard, trop fort, puis trop longtemps.

**On paie la chaleur. On subit aussi le froid.**

---

<!-- _class: split -->

![bg left:46%](../../pictures/presentations/photos/cta.jpg)

# Qui a une
# décision à prendre.

Bureaux d'études. Exploitants. Agrégateurs de flexibilité.

**Chauffer au seuil, ou avec un modèle qui voit venir la nuit.**

---

<!-- _class: chart -->

## Maison simulée : l'air part, les murs suivent.

![w:980](../../pictures/presentations/h2-reponse-indicielle.png)

---

<!-- _class: split -->

![bg left:46%](../../pictures/presentations/photos/physique.jpg)

# Comment on
# lit les chiffres.

Une maison, 2020, un point toutes les cinq minutes.

Le 21,8 °C du salon n'est pas l'air : le capteur marche par dixièmes de degré.

On ne mesure pas des watts. On reconstruit le chauffage à partir de l'eau et de l'air, seulement si la pièce demande.

---

<!-- _class: dark -->

# Ce projet, ce n'est pas.

Pas un thermostat magique recalé sur le même modèle.

Pas un essai dans la vraie maison.

**Une maison simulée, un peu différente exprès, pour tester le contrôleur.**

---

<!-- _class: full -->

![bg brightness:0.38](../../pictures/presentations/photos/hero.jpg)

# 1,8 h sous le confort.
# Le thermostat : 19,8 h.

48 h simulées. 18 h de froid en moins. Facture proxy ~16 € des deux côtés.

---

<!-- _class: chart -->

## 48 h simulées : 18 h de froid en moins, facture proxy identique.

![w:980](../../pictures/presentations/s6-confort-conso.png)

---

<!-- _class: chart -->

## Le contrôleur monte avant 7 h. Le thermostat attend 7 h.

![w:980](../../pictures/presentations/s4-mpc-vs-bang-bang.png)

---

<!-- _class: chart -->

## On n'a pas rejoué le contrôleur sur le modèle appris.

![w:980](../../pictures/presentations/s3-plant-vs-r2c2.png)

---

<!-- _class: actions -->

![bg right:38%](../../pictures/presentations/photos/action.jpg)

# Lundi.

**Exploitant** — garder le thermostat, mais compter les heures trop froid.

**Bureau d'études** — un modèle simple suffit pour anticiper ; la facture bouge peu, le froid disparaît.

Pas un produit. Une preuve de méthode.

---

<!-- _class: cta -->

![bg brightness:0.30](../../pictures/presentations/photos/cta.jpg)

# À vous.

[Voir les slides](https://dimiphoton.github.io/basic-MPC/slides/presentation-recruteur-fr.html)

[Jouer la leçon](https://dimiphoton.github.io/basic-MPC/simulator/)

[Code source](https://github.com/dimiphoton/basic-MPC)
