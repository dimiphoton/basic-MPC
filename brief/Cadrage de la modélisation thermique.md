# Cadrage de la modélisation thermique

## 1. Objectif

L'objectif est de construire un modèle dynamique simplifié de la maison à partir de données temporelles issues de capteurs, puis d'utiliser ce modèle pour mettre en place un **Model Predictive Control (MPC)** du chauffage.

Le modèle doit donc satisfaire deux objectifs :

1. **Représenter suffisamment bien la dynamique thermique** de la maison pour prédire son évolution sur plusieurs heures.
2. **Être suffisamment simple et identifiable** pour pouvoir être estimé à partir des données disponibles et utilisé ensuite dans un MPC.

Le problème n'est donc pas de construire une représentation exhaustive de la maison, mais de trouver le meilleur compromis entre **fidélité physique, complexité, identifiabilité et utilisabilité pour le contrôle**.

---

## 2. Nature des données

Les données disponibles sont temporelles et comprennent notamment :

- températures intérieures, avec un capteur par pièce ;
- température extérieure ;
- réglage ou puissance des radiateurs ;
- ensoleillement ;
- éventuellement d'autres variables disponibles.

Les températures intérieures sont des **observations provenant de capteurs** et non directement les températures thermiques réelles du bâtiment.

Cette distinction est importante : une température mesurée peut être affectée par le bruit, la quantification, un biais ou des erreurs ponctuelles.

On introduit donc deux niveaux :

\[
\text{état thermique réel}
\quad\longrightarrow\quad
\text{capteur}
\quad\longrightarrow\quad
\text{observation}
\]

---

# 3. Modèle thermique : état, entrées et observations

On distingue trois catégories de variables.

### États thermiques

Les états représentent l'énergie thermique stockée dans le bâtiment.

Par exemple :

\[
x_t =
\begin{bmatrix}
T_{\mathrm{air},t}\\
T_{\mathrm{masse},t}
\end{bmatrix}
\]

où :

- \(T_{\mathrm{air}}\) représente la température de l'air intérieur ;
- \(T_{\mathrm{masse}}\) représente une température thermique équivalente des murs, dalles, mobilier, etc.

Le deuxième état n'est généralement pas directement mesuré : il constitue une **mémoire thermique cachée**.

### Entrées

Les entrées sont les variables qui influencent le système mais qui ne sont pas elles-mêmes prédites par le modèle thermique :

\[
u_t =
\begin{bmatrix}
T_{\mathrm{ext},t}\\
S_t\\
P_t
\end{bmatrix}
\]

avec :

- \(T_{\mathrm{ext}}\) : température extérieure ;
- \(S\) : ensoleillement ;
- \(P\) : apport du chauffage.

D'autres variables peuvent également agir sur la température : présence des occupants, ouverture des fenêtres, appareils électriques, ventilation, etc.

Lorsqu'elles ne sont pas disponibles ou suffisamment fiables, elles sont considérées comme des **perturbations non modélisées** et absorbées par le bruit du modèle.

### Observations

Les températures issues des capteurs sont notées :

\[
y_t
\]

et constituent les sorties observées du système.

---

# 4. Pourquoi introduire une mémoire thermique ?

Une maison possède plusieurs échelles de temps thermiques.

L'air intérieur peut réagir relativement rapidement, tandis que les murs, les sols et les autres éléments du bâtiment peuvent stocker et restituer de la chaleur pendant plusieurs heures.

Ainsi, deux situations peuvent avoir la même température d'air :

\[
T_{\mathrm{air}}=20^\circ C
\]

mais des états thermiques différents :

\[
T_{\mathrm{masse}}=17^\circ C
\]

ou

\[
T_{\mathrm{masse}}=22^\circ C.
\]

Ces deux situations n'auront pas nécessairement la même évolution future.

La température mesurée seule ne contient donc pas nécessairement toute l'information nécessaire pour prédire la température à long horizon.

Une représentation avec état caché permet de représenter cette information passée sous une forme compacte.

---

# 5. Comparaison R1C / R2C

Deux niveaux de complexité seront considérés.

## 5.1 Modèle R1C

Le modèle R1C utilise un seul état thermique :

\[
x_t=T_{\mathrm{air},t}.
\]

Une forme discrète peut être écrite :

\[
T_{t+1}
=
aT_t
+bT_{\mathrm{ext},t}
+cS_t
+dP_t
+w_t.
\]

Le modèle possède une seule échelle temporelle dominante.

Il constitue une **baseline importante** car il est :

- simple ;
- facilement identifiable ;
- peu coûteux ;
- utile pour vérifier si une mémoire thermique supplémentaire est réellement nécessaire.

Cependant, il ne possède qu'un seul état et ne permet donc pas de représenter explicitement plusieurs dynamiques thermiques.

---

## 5.2 Modèle R2C

Le modèle R2C introduit un second état représentant l'inertie thermique du bâtiment :

\[
x_t =
\begin{bmatrix}
T_{\mathrm{air},t}\\
T_{\mathrm{masse},t}
\end{bmatrix}.
\]

Une représentation physique possible est :

\[
C_a\frac{dT_a}{dt}
=
\frac{T_m-T_a}{R_{am}}
+
\frac{T_e-T_a}{R_{ae}}
+
\alpha_sS
+
\alpha_hP
\]

\[
C_m\frac{dT_m}{dt}
=
\frac{T_a-T_m}{R_{am}}.
\]

Ce modèle possède plusieurs échelles temporelles et permet donc de représenter une partie de l'inertie et du déphasage thermique.

---

# 6. Pourquoi ne pas choisir directement le modèle le plus complexe ?

Un modèle plus complexe n'est pas nécessairement un meilleur modèle.

Chaque état ou paramètre supplémentaire doit être **identifiable à partir des données**.

Un modèle R2C peut par exemple être théoriquement plus réaliste mais devenir difficile à identifier si :

- les données sont trop courtes ;
- les entrées sont fortement corrélées ;
- le chauffage varie peu ;
- l'ensoleillement est insuffisant ;
- les capteurs sont trop bruités.

Le choix du modèle sera donc effectué expérimentalement.

Le R1C servira de référence et le R2C sera retenu si l'état supplémentaire permet d'améliorer significativement les prédictions, notamment à moyen et long horizon.

---

# 7. Critère de comparaison

Les modèles ne seront pas uniquement comparés sur leur prédiction à \(t+1\).

Un modèle destiné à un MPC doit être capable de reproduire correctement l'évolution de la température sur plusieurs heures.

On évaluera donc les modèles sur plusieurs horizons :

\[
1h,\quad 3h,\quad 6h,\quad 12h,\quad 24h.
\]

Les critères pourront inclure :

- RMSE ;
- MAE ;
- erreur de prédiction à différents horizons ;
- stabilité des paramètres ;
- comportement lors des phases chauffage ON/OFF ;
- capacité à reproduire les variations jour/nuit ;
- cohérence physique des paramètres.

Les données utilisées pour l'évaluation devront être distinctes des données utilisées pour l'identification.

---

# 8. Modèle de capteur

Le modèle thermique ne doit pas être confondu avec le modèle de mesure.

On considère que le système possède un état thermique réel \(x_t\), mais que le capteur fournit une observation imparfaite :

\[
y_t = h(x_t) + v_t.
\]

Pour une mesure directe de la température de l'air :

\[
y_t = T_{\mathrm{air},t}+v_t.
\]

Le terme \(v_t\) peut représenter :

- le bruit de mesure ;
- la quantification de la température ;
- une précision limitée du capteur ;
- éventuellement des erreurs ponctuelles.

Si la température est enregistrée sous forme entière, cette quantification doit notamment être distinguée d'une véritable dynamique thermique.

Le modèle de capteur est particulièrement important pour l'estimation de l'état caché \(T_{\mathrm{masse}}\).

---

# 9. Modèle d'état complet

Le cadre retenu est donc un modèle de type state-space :

\[
x_{t+1}=f(x_t,u_t,\theta)+w_t
\]

\[
y_t=h(x_t,\theta_s)+v_t.
\]

où :

- \(x_t\) : états thermiques ;
- \(u_t\) : entrées mesurées ou contrôlées ;
- \(y_t\) : observations des capteurs ;
- \(\theta\) : paramètres thermiques ;
- \(\theta_s\) : paramètres du capteur ;
- \(w_t\) : perturbations thermiques non modélisées ;
- \(v_t\) : bruit de mesure.

Cette formulation permet de séparer explicitement :

\[
\boxed{
\text{physique thermique}
\neq
\text{mesure du capteur}
}
\]

---

# 10. Stratégie d'identification

L'identification sera réalisée sur des trajectoires temporelles suffisamment longues pour que le modèle puisse être contraint par plusieurs régimes thermiques.

L'objectif n'est pas uniquement de minimiser l'erreur de prédiction à un pas :

\[
y_{t+1}-\hat y_{t+1},
\]

mais de vérifier que les paramètres permettent de reproduire correctement une trajectoire sur plusieurs heures.

Une première estimation locale pourra éventuellement servir à initialiser les paramètres, mais la validation finale devra porter sur des trajectoires longues.

Les paramètres et les états cachés pourront ensuite être estimés dans un cadre state-space, notamment à l'aide d'un filtre de Kalman ou d'une méthode d'estimation équivalente.

---

# 11. Lien avec le MPC

Le modèle identifié constitue ensuite le modèle interne du MPC.

À chaque instant, le MPC dispose :

- de l'état thermique estimé ;
- des prévisions de température extérieure ;
- des prévisions d'ensoleillement ;
- de la consigne de température ;
- des contraintes sur le chauffage.

Il prédit alors l'évolution future :

\[
x_{t+1},x_{t+2},\ldots,x_{t+N}
\]

pour différentes séquences de chauffage et choisit la commande optimale.

La présence d'un état thermique caché permet au MPC de tenir compte de l'énergie stockée dans le bâtiment.

---

# 12. Cadrage final

Le projet sera donc organisé selon le pipeline suivant :

```text
                 DONNÉES CAPTEURS
                       │
                       ▼
              Prétraitement / qualité
                       │
                       ▼
              ┌───────────────────┐
              │ Identification     │
              │                   │
              │ R1C      R2C      │
              └───────────────────┘
                       │
                       ▼
                  Validation
             court + moyen + long
                       │
                       ▼
                Modèle retenu
                       │
                       ▼
              Estimation d'état
             (état thermique caché)
                       │
                       ▼
                     MPC
                       │
                       ▼
               Commande chauffage
```

Le **R1C est donc utilisé comme baseline**, tandis que le **R2C constitue le modèle cible** si les données montrent qu'une seconde dynamique thermique est nécessaire.

Le choix final ne sera pas fondé uniquement sur la fidélité physique théorique, mais sur le compromis :

\[
\boxed{
\text{simplicité}
+
\text{identifiabilité}
+
\text{qualité prédictive}
+
\text{cohérence physique}
+
\text{performance pour le MPC}
}
\]

L'objectif est ainsi de construire **le modèle le plus simple capable de capturer les dynamiques thermiques pertinentes pour le contrôle prédictif**.