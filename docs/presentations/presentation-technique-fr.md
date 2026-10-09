---
marp: true
theme: portfolio
paginate: true
---

<!-- _class: cover -->
<!-- _paginate: false -->

![bg brightness:0.40](../../pictures/presentations/photos/hero.jpg)

# Un modèle simple
# suffit-il à chauffer
# en avance ?

Machine learning · Bâtiment · Python / NumPy / SciPy

Maison instrumentée · 2020 · ~5 min

---

<!-- _class: split -->

![bg left:46%](../../pictures/presentations/photos/action.jpg)

# Un état suffit
# sur cette pièce.

À 1 h : 0,56 °C avec un état, 0,63 °C avec deux.

**À 24 h : 1,93 °C et 1,95 °C. Le second état, tenu dans une fourchette de pièce, ne gagne pas.**

---

<!-- _class: split -->

![bg left:46%](../../pictures/presentations/photos/cta.jpg)

# Le livrable
# est le contrôleur.

Pas un dashboard. Pas un rapport d'identification.

Bureaux d'études et exploitants : une consigne thermostat, une bande de confort, une maison simulée distincte.

---

<!-- _class: chart -->

## Maison simulée : l'air part, les murs suivent.

![w:980](../../pictures/presentations/h2-reponse-indicielle.png)

---

<!-- _class: split -->

![bg left:46%](../../pictures/presentations/photos/physique.jpg)

# D'où viennent
# les chiffres.

Salon et extérieur, toutes les cinq minutes. Un trou de plus de dix minutes reste un trou.

Le capteur marche par dixièmes de degré — ce n'est pas une constante de temps.

Pas de watts mesurés : on reconstruit le chauffage à partir de l'eau et de l'air, seulement si la pièce demande.

---

<!-- _class: chart -->

## Kalman à la main : on avance, on compare, on corrige.

![w:920](../../pictures/presentations/schema-kalman.png)

---

<!-- _class: dark -->

# Périmètre.

On identifie le modèle sur des capteurs réels.

On teste le contrôleur sur une maison simulée.

Le soleil y tape aussi les murs. Ce n'est pas une validation en maison.

---

<!-- _class: chart -->

## Un état reste devant, de 1 h à 24 h.

![w:920](../../pictures/presentations/i1-rmse-horizons.png)

---

<!-- _class: chart -->

## Même météo, deux physiques : le contrôleur n'est pas recalé sur lui-même.

![w:980](../../pictures/presentations/s3-plant-vs-r2c2.png)

---

<!-- _class: full -->

![bg brightness:0.38](../../pictures/presentations/photos/hero.jpg)

# 1,8 h sous le confort.
# Thermostat : 19,8 h.

48 h, maison simulée, météo connue. Facture proxy ~16 €. Écart : 18 h.

---

<!-- _class: chart -->

## Le MPC monte avant 7 h. Le thermostat attend 7 h.

![w:980](../../pictures/presentations/s4-mpc-vs-bang-bang.png)

---

<!-- _class: chart -->

## La décision est une consigne thermostat, pas des watts.

![w:980](../../pictures/presentations/s5-commande-p.png)

---

<!-- _class: chart -->

## Une bande d'un degré : plus l'écart grandit, plus on chauffe.

![w:920](../../pictures/presentations/h1-loi-bande-n.png)

---

<!-- _class: chart -->

## Si la météo arrive avec six heures de retard, le confort se dégrade.

![w:980](../../pictures/presentations/h9-oracle-vs-retard.png)

---

<!-- _class: dark -->

# Le salon ne sépare pas
# l'air et les murs.

6 h et 8 h, les deux au bord de la fourchette.

Le contrôleur est jugé sur une autre maison : air ~3 h, murs ~11 h.

---

<!-- _class: dark -->

# Où ça casse.

Météo connue à l'avance (oracle).

Le chauffage n'est pas des watts.

Une maison, un hiver de calage, pas un parc.

---

<!-- _class: cta -->

![bg brightness:0.30](../../pictures/presentations/photos/cta.jpg)

# Reproduire.

[Slides](https://dimiphoton.github.io/basic-MPC/slides/presentation-technique-fr.html)
[Repo](https://github.com/dimiphoton/basic-MPC)

`notebooks/03-mpc-cout-consigne.ipynb`

`pytest`

[Leçon](https://dimiphoton.github.io/basic-MPC/simulator/)

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?logo=scipy&logoColor=white)
