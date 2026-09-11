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

<!-- _class: full -->

![bg brightness:0.38](../../pictures/presentations/photos/cta.jpg)

# L'air chauffe vite.
# Les murs s'en souviennent.

Deux pièces à 20 °C n'ont pas le même lendemain
si l'une a des murs froids.

---

<!-- _class: split -->

![bg left:46%](../../pictures/presentations/photos/action.jpg)

# Comment on
# lit les chiffres.

Une maison, 2020, maille 5 min — parfois un point sauté.

Le 21,8 °C du salon n'est pas l'air : le capteur marche par 0,1 °C.

Le chauffage n'est pas en watts : on construit un écart eau / air, seulement si la pièce demande.

---

<!-- _class: dark -->

# Ce projet, ce n'est pas.

Pas un thermostat magique recalé sur le même modèle.

Pas des watts mesurés, ni un essai dans la vraie maison.

**Un plant simulé, un peu différent exprès, pour tester le contrôleur.**

---

<!-- _class: full -->

![bg brightness:0.38](../../pictures/presentations/photos/hero.jpg)

# 2 h trop froid.
# Le thermostat : 20.

Même facture. Le gain est le confort, pas l'euro.

---

<!-- _class: chart -->

## Même facture. Vingt heures de froid en moins.

![w:980](../../pictures/presentations/s6-confort-conso.png)

---

<!-- _class: chart -->

## Le thermostat oscille. Le MPC préchauffe en heures creuses.

![w:980](../../pictures/presentations/s4-mpc-vs-bang-bang.png)

---

<!-- _class: split -->

![bg left:40%](../../pictures/presentations/photos/hero.jpg)

# Ce n'est pas
# un artefact.

On n'a pas rejoué le contrôleur sur le modèle appris.

La maison simulée reçoit le soleil aussi dans les murs.
Ce n'est pas celle qu'on a identifiée.

---

<!-- _class: actions -->

![bg right:38%](../../pictures/presentations/photos/action.jpg)

# Lundi.

**Exploitant** — garder le thermostat, mais compter les heures sous la consigne de confort.

**Bureau d'études** — un modèle simple suffit pour anticiper ; la facture bouge peu, le froid disparaît.

Pas un produit. Une preuve de méthode.

---

<!-- _class: cta -->

![bg brightness:0.30](../../pictures/presentations/photos/cta.jpg)

# À vous.

[Voir les slides](https://dimiphoton.github.io/basic-MPC/slides/presentation-recruteur-fr.html)

[Jouer le simulateur](https://dimiphoton.github.io/basic-MPC/simulator/)

[Code source](https://github.com/dimiphoton/basic-MPC)
