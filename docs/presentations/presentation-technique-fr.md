---
marp: true
theme: portfolio
paginate: true
---

<!-- _class: cover -->
<!-- _paginate: false -->

<!-- Photo : pictures/presentations/photos/hero.png -->
<!-- ![bg brightness:0.40](../../pictures/presentations/photos/hero.png) -->

# Un modèle RC identifiable
# suffit-il au MPC ?

Machine learning · Bâtiment · Python / NumPy / SciPy

Maison instrumentée · 2020 · ~5 min

---

<!-- _class: split -->

<!-- ![bg left:46%](../../pictures/presentations/photos/motivation.png) -->

# [Pourquoi
# mesurer ça.]

[Enjeu chiffré si on l'a. Coût d'une mauvaise décision.]

**[Ce qui manque aujourd'hui pour trancher.]**

---

<!-- _class: split -->

<!-- ![bg left:46%](../../pictures/presentations/photos/hero.png) -->

# [Qui consomme
# le résultat.]

[Agence / régulateur]. [Opérateur / assureur / bureau d'études].

Le livrable : [indicateur, vue, reco], pas un rapport.

---

<!-- _class: full -->

<!-- ![bg brightness:0.38](../../pictures/presentations/photos/physique.png) -->

# [Mécanisme.]

[Physique ou processus : pluie → saturé → rendement, courbe de charge, Espec…]

---

<!-- _class: split -->

<!-- ![bg left:46%](../../pictures/presentations/photos/motivation.png) -->

# [Logique du
# traitement.]

Salon + extérieur, maille 5 min. Interpolation limitée à 10 min.

On ne ponte pas un trou de 12 h. Quantification 0,1 °C (salon) vs 1 °C (extérieur) : modèle de capteur, pas de RC.

Pas de watts mesurés : `P` = écart eau/air si la zone demande, `S` = PV.

---

<!-- _class: split -->

<!-- ![bg left:46%](../../pictures/presentations/photos/physique.png) -->

# [Ce qu'on isole.]

On retire [confondant]. Ce qui reste, c'est [cible].

Pas [ce qu'on ne prétend pas].

---

<!-- _class: dark -->

# Périmètre.

On identifie un RC sur capteurs réels. On teste le MPC sur un plant littérature.

On n'est pas une validation en maison réelle : le solaire du plant tape aussi les murs.

---

<!-- _class: chart -->

[Baseline / tendance / donnée brute — titre-phrase.]

<!-- ![w:920](../../pictures/presentations/baseline-fr.png) -->

---

<!-- _class: full -->

<!-- ![bg brightness:0.38](../../pictures/presentations/photos/physique.png) -->

# [Résultat principal]
# [métrique + n]

---

<!-- _class: chart -->

[Graphe de *ce* récit technique — pas forcément celui du deck RH.]

<!-- ![w:980](../../pictures/presentations/graphique-cle-fr.png) -->

---

<!-- _class: split -->

<!-- ![bg left:40%](../../pictures/presentations/photos/hero.png) -->

# [Robustesse.]

[Spatial / années / n]. [Ce qui n'est pas indépendant.]

<!-- ![w:480](../../pictures/presentations/carte-ou-robustesse-fr.png) -->

---

<!-- _class: chart -->

Pourquoi pas [modèle tape-à-l'œil] ? n = […]. [Modèle retenu + validation.]

<!-- ![w:640](../../pictures/presentations/validation-fr.png) -->

---

<!-- _class: dark -->

# Où ça casse.

[Limite 1.]

[Limite 2.]

[Limite 3. Corrélation ≠ cause si pertinent.]

---

<!-- _class: cta -->

<!-- ![bg brightness:0.30](../../pictures/presentations/photos/cta.png) -->

# Reproduire.

[Explorer en ligne](../explore-fr.html)

`python -m basic_mpc run`

`python -m basic_mpc dashboard`

<!-- Badges de stack ici, pas en slide 1 recruteur. -->
