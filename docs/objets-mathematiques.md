# Objets mathématiques du projet

Registre des équations et concepts, calé sur le code
(`src/basic_mpc/`), pas sur le brief phare. Quatre RC distincts
partagent des lettres ; ils ne sont pas le même objet.

Maille \(\Delta t = 300\,\mathrm{s}\) (5 min), fuseau
`Europe/Brussels`.

---

## 0. Vocabulaire : ce qu'il ne faut pas coller

| Lettre | Identification (salon 2020) | Contrôle (plant simulé) |
|---|---|---|
| \(P\) | Proxy \(\max(T_{\mathrm{eau}}-T_{\mathrm{air}},0)\) si appel de zone, en **K** | Apport produit par la bande \(n\), unités proxy du plant |
| \(S\) | Somme des phases PV, pas des W/m² | Perturbation thermique oracle (chauffe, ne réduit pas la facture) |
| \(R,C\) | Non séparables en SI ; \(C_a=1\) | Plant littérature en J/K et K/W |
| \(\theta\) | Fit PEM sur le salon | **Pas** réutilisé : modèle interne calé sur le plant |
| \(y\) | Capteur salon, pas \(0,1\,^\circ\mathrm{C}\) | Capteur simulé, même quantification |

**Grey-box** : structure physique (réseau RC) + paramètres inférés
statistiquement. Ni boîte noire, ni modèle SI calibré en labo.

**Non-circularité** : on n'évalue pas le MPC sur le RC qu'il a appris.
Le plant a un solaire sur la masse (\(\alpha_{s,\mathrm{mass}}\)) que
le contrôleur n'a pas.

---

## 1. Données et capteur

### 1.1 Observation

L'état thermique \(x\) n'est pas la mesure.

\[
y_k = T_{\mathrm{air},k} + v_k
\]

\(v\) regroupe bruit et **quantification**. Ce n'est pas une dynamique
de mur.

Salon : pas \(0,1\,^\circ\mathrm{C}\). Extérieur : \(1\,^\circ\mathrm{C}\).
Sur le plant simulé :

\[
\tilde y = T_{\mathrm{air}} + \varepsilon,\quad
\varepsilon\sim\mathcal N(0,\sigma_{\mathrm{cap}}^2),\quad
\sigma_{\mathrm{cap}}=0{,}02\,^\circ\mathrm{C}
\]

\[
y = \Delta\,\mathrm{round}(\tilde y / \Delta),\quad \Delta = 0{,}1\,^\circ\mathrm{C}.
\]

À l'identification, \(\sigma_v\) du Kalman est **fixé** à
\(0{,}05\,^\circ\mathrm{C}\) (un peu au-dessus du pas salon). On
n'identifie pas le capteur en même temps que le bâtiment.

### 1.2 Trous

Interpolation linéaire au plus **2 pas** (10 min) sur la maille 5 min.
Un trou plus long reste `NaN` : le Kalman **prédit sans update** ; si
\(u\) est `NaN`, on **n'avance pas** la dynamique (on n'invente pas
12 h de physique).

### 1.3 Entrées construites (identification seulement)

\[
P_k =
\max(T_{\mathrm{eau},k}-T_{\mathrm{air},k},\,0)
\cdot
\mathbf{1}_{\{T_{\mathrm{air},k} < T_{\mathrm{consigne},k}\}}
\quad\text{(kelvin, pas des watts)}
\]

\[
S_k = \sum_{i=1}^{3} \max(\mathrm{PV}_{L_i,k},\,0)
\quad\text{(proxy, pas des W/m}^2\text{)}
\]

\[
u_k = \begin{bmatrix} T_{\mathrm{ext},k} \\ S_k \\ P_k \end{bmatrix}
\]

Occupants, fenêtres, gains internes non mesurés → bruit de process
\(w\), pas un second jeu de \(R,C\).

---

## 2. Famille RC : quatre dynamiques

### 2.1 R1C1 (identification, baseline)

Un nœud air. Forme continue (commentaire du code) :

\[
\frac{\mathrm{d}T}{\mathrm{d}t}
=
\frac{T_{\mathrm{ext}}-T}{\tau}
+ g_s S + g_h P
\]

On identifie la forme **discrète** grey-box : le gain sur
\(T_{\mathrm{ext}}\) vaut \(1-a\).

\[
T_{k+1}
=
a\,T_k
+ (1-a)\,T_{\mathrm{ext},k}
+ g_S S_k
+ g_P P_k
+ w_k
\]

\[
\tau = -\frac{\Delta t}{\log a}
\quad\text{(en secondes ; en heures : }\tau/3600\text{)}
\]

\(\theta_{\mathrm{R1C1}} = (a,\,g_S,\,g_P,\,\sigma_w)\).
\(R\) et \(C\) ne sont pas séparables (\(P\) n'est pas en watts).
Bornes : \(a\in(0{,}80,\,0{,}9998)\), gains \(>0\).

### 2.2 R2C2 d'identification (solaire sur l'air seulement)

Deux états \(x = (T_a,\,T_m)\). \(C_a = 1\) (échelle absorbée par les
\(R\) et les gains). \(R_{ae}\), \(R_{am}\) sont alors des **temps** (s).

\[
C_a \dot T_a
=
\frac{T_m-T_a}{R_{am}}
+
\frac{T_e-T_a}{R_{ae}}
+
\alpha_s S
+
\alpha_h P
\]

\[
C_m \dot T_m
=
\frac{T_a-T_m}{R_{am}}
\]

Matrices continues, \(u = [T_{\mathrm{ext}},\,S,\,P]\) :

\[
A =
\begin{bmatrix}
-(1/(R_{am}C_a)+1/(R_{ae}C_a)) & 1/(R_{am}C_a) \\
1/(R_{am}C_m) & -1/(R_{am}C_m)
\end{bmatrix}
,\quad
B =
\begin{bmatrix}
1/(R_{ae}C_a) & g_S & g_P \\
0 & 0 & 0
\end{bmatrix}
\]

Échelles :

\[
\tau_{\mathrm{air}} = R_{ae} C_a,\qquad
\tau_{\mathrm{masse}} = R_{am} C_m.
\]

\(\theta_{\mathrm{R2C2}} = (R_{ae},\,R_{am},\,C_m,\,g_S,\,g_P,\,\sigma_w)\)
avec \(\sigma_{w,\mathrm{masse}}=\sigma_w\) (un seul bruit identifié).
Plafond : \(R_{ae}\le 5\cdot 10^5\,\mathrm{s}\) ⇒
\(\tau_{\mathrm{air}}\le 139\,\mathrm{h}\). \(C_m\in[1{,}2,\,200]\).

Résultat observé : les deux \(\tau\) saturent au lent (\(\sim 140\,\mathrm{h}\)).
Ce n'est **pas** un air rapide + une masse.

### 2.3 Plant littérature (simulateur)

Mêmes \(A\) 2×2, mais le solaire chauffe **aussi** la masse.

\[
B_{\mathrm{plant}} =
\begin{bmatrix}
1/(R_{ae}C_a) & \alpha_{s,\mathrm{air}}/C_a & \alpha_h/C_a \\
0 & \alpha_{s,\mathrm{mass}}/C_m & 0
\end{bmatrix}
\]

Paramètres **non fittés** (ordres de grandeur maison) :

| Symbole | Valeur | Unité |
|---|---|---|
| \(C_a\) | \(5\cdot 10^5\) | J/K |
| \(C_m\) | \(8\cdot 10^6\) | J/K |
| \(R_{am}\) | \(5\cdot 10^{-3}\) | K/W |
| \(R_{ae}\) | \(2\cdot 10^{-2}\) | K/W |
| \(\alpha_{s,\mathrm{air}}\) | \(8\cdot 10^{-3}\) | — |
| \(\alpha_{s,\mathrm{mass}}\) | \(4\cdot 10^{-3}\) | — |
| \(\alpha_h\) | \(4\) | — |

D'où \(\tau_{\mathrm{air}}\approx 2{,}8\,\mathrm{h}\),
\(\tau_{\mathrm{masse}}\approx 11\,\mathrm{h}\).
Bruit de process plant : \(\sigma=0{,}01\,^\circ\mathrm{C}\) (deux composantes).

### 2.4 Modèle interne du MPC

Structure du R2C2 d'identification (solaire **air seulement**), gains
**calés sur le plant**, pas sur le salon :

\[
R_{ae}^{\mathrm{int}} = R_{ae}^{\mathrm{plant}} C_a,\quad
R_{am}^{\mathrm{int}} = R_{am}^{\mathrm{plant}} C_a,\quad
C_m^{\mathrm{int}} = C_m^{\mathrm{plant}}/C_a
\]

\[
g_S = \alpha_{s,\mathrm{air}}/C_a,\quad
g_P = \alpha_h/C_a,\quad
C_a^{\mathrm{int}}=1.
\]

Pas de \(\alpha_{s,\mathrm{mass}}\). Le \(\hat\theta\) salon (\(\tau\sim 140\,\mathrm{h}\))
ne commande pas ce plant.

---

## 3. Discrétisation exacte (R2C2 et plant)

\[
A_d = \mathrm{e}^{A\Delta t},\qquad
B_d = A^{-1}(A_d-I)B
\quad\text{(\(A\) inversible : fuite vers l'extérieur).}
\]

Pas d'Euler. Le R1C1 est déjà posé en discret.

Convention d'entrée : \(u_{k-1}\) agit **après** \(y_k\) pour aller
de \(k-1\) à \(k\).

\[
x_{k+1} = A_d x_k + B_d u_k + w_k
\]

---

## 4. Filtre de Kalman (linéaire, à la main)

### 4.1 Hypothèses

\[
x_{k+1} = A_d x_k + B_d u_k + w_k,\quad
w_k\sim\mathcal N(0,Q)
\]

\[
y_k = C x_k + v_k,\quad
v_k\sim\mathcal N(0,R),\quad
C = [1]\ \text{ou}\ [1\ 0]
\]

R1C1 : \(Q = \sigma_w^2\), \(R=\sigma_v^2\).
R2C2 : \(Q=\mathrm{diag}(\sigma_w^2,\sigma_w^2)\).

Prior : \(x_0\) ancré sur le premier \(y\) fini ;
\(P_0 = 4\) (R1C1) ou \(\mathrm{diag}(4,\,16)\) (R2C2).

### 4.2 Un pas (prédiction puis update)

Si \(u_{k-1}\) est fini :

\[
\hat x_{k|k-1} = A_d \hat x_{k-1|k-1} + B_d u_{k-1}
\]

\[
P_{k|k-1} = A_d P_{k-1|k-1} A_d^\top + Q
\]

Innovation et variance (mesure = air) :

\[
e_k = y_k - C\hat x_{k|k-1},\qquad
S_k = C P_{k|k-1} C^\top + R
\]

Si \(y_k\) est fini :

\[
K_k = P_{k|k-1} C^\top S_k^{-1}
\]

\[
\hat x_{k|k} = \hat x_{k|k-1} + K_k e_k
\]

\[
P_{k|k} = (I-K_k C)P_{k|k-1}
\quad\text{(forme Joseph sur le chemin matriciel)}
\]

Si \(y_k\) est `NaN` : pas d'update, pas de terme dans la vraisemblance.

En boucle fermée, le même filtre tourne **en ligne**
(`KalmanTracker`) : état caché \(T_m\) estimé à chaque pas de contrôle.

### 4.3 Vraisemblance gaussienne des innovations

\[
\log L
=
\sum_{k:\,y_k\ \mathrm{fini}}
-\frac12
\left(
\log(2\pi S_k) + \frac{e_k^2}{S_k}
\right)
\]

C'est le critère PEM. Ce n'est pas un RMSE à un pas.

---

## 5. Identification (PEM)

### 5.1 Programme d'estimation

\[
\hat\theta = \arg\min_\theta\ (-\log L(\theta;\,y,u))
\]

- R1C1 : espace non contraint
  \((\mathrm{logit}(a),\,\log g_S,\,\log g_P,\,\log\sigma_w)\),
  **BFGS**.
- R2C2 : \(\log\) des six paramètres libres, **Nelder-Mead**
  (BFGS partait dans des \(A\) singulières).
- \(\sigma_v\) hors \(\theta\).

Incertitudes R1C1 : diagonale de \(H^{-1}\) dans l'espace \(\theta\),
delta method vers \(a\), gains, \(\tau\).

### 5.2 Protocole (trajectoire, pas iid)

1. Série continue 5 min.
2. Split **chronologique** 70 / 30 (début = train, fin = test). Pas de
   shuffle de semaines : l'état masse serait faux à chaque raccord.
3. Fit = les **14 400** derniers pas du train (\(\approx 50\,\mathrm{j}\),
   16 déc. 2020 → 4 fév. 2021). Même fenêtre R1C1 et R2C2.
4. Métriques sur **tout** le test (fév. → mai 2021).

\(R\) et \(C\) ne changent pas le samedi. Un RC ouvré / week-end
identifierait deux maisons pour absorber l'occupation.

Trou du protocole : fit **hiver**, test **printemps** ; \(g_S\) quasi
non identifiable sur la fenêtre froide.

### 5.3 Identifiabilité (concepts)

- **Grey-box** : la structure \(A(\theta),B(\theta)\) est imposée ; on
  n'estime pas une matrice libre.
- **Échelle** : sans watts, \(R\) et \(C\) ne se séparent pas → \(C_a=1\)
  ou forme \((a,g)\).
- **Excitation** : hors saison, \(P\approx 0\) ; le PEM sur toute
  l'année dilue le chauffage.
- **Overparamétrisation** : un R2C2 peut coller en NLL sans séparer
  deux échelles physiques (cas observé : plafond \(R_{ae}\)).

---

## 6. Prédiction multi-horizon (validation identification)

Un MPC se juge à plusieurs heures, pas à \(t+1\).

À partir de \(\hat x_{t|t}\), **prédiction libre** (pas de Kalman sur
l'horizon) avec les \(u\) **mesurés** :

\[
\hat x_{t+h|t}
=
A_d^h \hat x_{t|t}
+
\sum_{i=0}^{h-1} A_d^{h-1-i} B_d u_{t+i}
\]

\[
\hat y_{t+h|t} = C\hat x_{t+h|t}
\]

Horizons : \(1,3,6,12,24\,\mathrm{h}\). Un départ toutes les heures
(`stride = 12` pas).

\[
\mathrm{RMSE}(h) = \sqrt{\mathrm{moyenne}(\hat y_{t+h|t}-y_{t+h})^2},\quad
\mathrm{MAE}(h)=\mathrm{moyenne}|\hat y_{t+h|t}-y_{t+h}|
\]

Observé (test) : à 1 h, R1C1 = R2C2 (\(0{,}56\,^\circ\mathrm{C}\)) ;
à 24 h, \(1{,}93\) vs \(1{,}73\,^\circ\mathrm{C}\). L'ACF des
innovations R1C1 (~2 h) dit qu'il **manque** une dynamique ; le R2C2
ne l'a pas trouvée comme air rapide + masse.

---

## 7. Impédance \(Z(j\omega)\)

Outil de **lecture** de la dynamique, pas un estimateur.
\(T_{\mathrm{ext}}=0\), \(Z = T_{\mathrm{air}}/Q_{\mathrm{air}}\).

R1C1 (\(C=1\), \(R=\tau\)) :

\[
Z(j\omega) = \frac{\tau}{1+j\omega\tau}
\]

R2C2 :

\[
Y_a = \frac{1}{R_{am}}+\frac{1}{R_{ae}}+j\omega C_a,\quad
Y_m = \frac{1}{R_{am}}+j\omega C_m
\]

\[
Z(j\omega) = \frac{Y_m}{Y_a Y_m - (1/R_{am})^2}
\]

Le retard à 24 h se lit sur l'argument de \(Z(j\omega_{24\mathrm{h}})\).
C'est la leçon Pages (déphasage), distincte du QP.

---

## 8. Contrôle

Deux programmes d'optimisation **distincts** du PEM.

### 8.1 Actionneur (v1.1) : bande proportionnelle \(n\)

Décision métier : consigne thermostat \(T_{\mathrm{sp}}\), pas des watts.

\[
P_k
=
P_{\max}\,
\mathrm{sat}\!\left(\frac{T_{\mathrm{sp},k}-T_{\mathrm{air},k}}{n},\,0,\,1\right)
\]

\(n=1\,^\circ\mathrm{C}\) : à la consigne = arrêt ; \(1\,^\circ\mathrm{C}\)
en dessous = à fond. Inverse affine (pour le QP) :

\[
T_{\mathrm{sp},k}
=
T_{\mathrm{air},k} + n\,\frac{P_k}{P_{\max}},
\qquad
T_{\mathrm{sp}}\in[16,\,22]\,^\circ\mathrm{C}.
\]

\(T_{\mathrm{sp}}\) est **rafraîchi chaque pas** pour réaliser le \(P\)
du bloc de 30 min (sinon la bande coupe dès que l'air monte).

\(P_{\max}\) : marge \(1{,}4\) × puissance de maintien au
\(T_{\mathrm{ext}}\) le plus froid du scénario, \(S=0\), \(T_m=T_a\) :

\[
P_{\mathrm{hold}}
=
\max\left(0,\,\frac{T_{\mathrm{set}}-T_{\mathrm{ext}}}{R_{ae}\,\alpha_h}\right)
\]

(unités proxy du plant).

### 8.2 Calendriers (connus, pas des prévisions)

Heure locale Bruxelles. Occupé = \([7\mathrm{h},\,22\mathrm{h})\).

\[
T_{\mathrm{conf}}(t)
=
\begin{cases}
20\,^\circ\mathrm{C} & \text{occupé} \\
17\,^\circ\mathrm{C} & \text{sinon}
\end{cases}
\qquad
\pi(t)
=
\begin{cases}
0{,}40\ \mathrm{€/kWh} & \text{heures pleines (occupé)} \\
0{,}20\ \mathrm{€/kWh} & \text{heures creuses}
\end{cases}
\]

Météo \(T_{\mathrm{ext}},S\) : **oracle** sur l'horizon (on mesure le
gain du modèle, pas d'un prévisionniste). \(S\) chauffe ; \(S\) n'entre
pas dans \(\pi\).

### 8.3 Conversion proxy → kWh et poids confort

Chaleur qui entre dans le nœud air du plant (\(\alpha_h=4\)), **pas un
compteur** :

\[
E_{\mathrm{kWh}} = \beta\, P\, \Delta t_{\mathrm{s}},
\qquad
\beta = \frac{\alpha_h}{3{,}6\cdot 10^6}
= \frac{4}{3{,}6\cdot 10^6}
\]

\[
\lambda = 1{,}0\ \mathrm{€{\cdot}K^{-2}{\cdot}h^{-1}}
\]

(1 K sous \(T_{\mathrm{conf}}\) pendant 1 h ≈ 1 € d'inconfort).

### 8.4 Coût v1.1 (celui du livrable)

Horizon \(N\) pas, move blocking : \(P\) constant par blocs de 30 min
(6 pas). Horizon de prédiction : 6 h (\(N=72\)).

\[
J
=
\sum_{k=0}^{N-1}
\left[
\pi_k\,\beta\,P_k\,\Delta t_{\mathrm{s}}
+
\lambda\,
\max(T_{\mathrm{conf},k}-T_{\mathrm{air},k},\,0)^2
\,\Delta t_{\mathrm{h}}
\right]
\]

Pas de pénalité « trop chaud » (le plafond \(T_{\mathrm{sp}}\le 22\)
suffit). Pas de contrainte dure sur \(T_{\mathrm{air}}\) (sinon
infaisable au démarrage à 18 °C).

Forme condensée : \(T_{\mathrm{air}} = \mathrm{free} + \Gamma P\)
(réponse impulsionnelle du chauffage + trajectoire libre \(P=0\)).

\[
P_{\mathrm{full}} = S_{\mathrm{block}}\, p_{\mathrm{moves}}
\]

Solveur : **SLSQP** (contraintes linéaires sur \(T_{\mathrm{sp}\)),
repli **L-BFGS-B** borné si le domaine \(T_{\mathrm{sp}}\) est vide.
On décide \(p_{\mathrm{moves}}\), on applique le **premier** bloc
(horizon glissant).

### 8.5 Coût v1 (historique, encore dans le code)

\[
J_{\mathrm{v1}}
=
q_{\mathrm{track}}\|T-20\|^2
+
q_{\mathrm{band}}(\mathrm{sous}^2+\mathrm{sur}^2)
+
r_u \|P\|^2
\]

Poids magiques \(q_{\mathrm{track}}=1\), \(q_{\mathrm{band}}=20\),
\(r_{\mathrm{rel}}=0{,}04\). Remplacé pour le récit v1.1 : un
recruteur ne lit pas \(r_{\mathrm{rel}}\).

### 8.6 Baseline : hystérésis, pas un second QP

Même \(T_{\mathrm{conf}}(t)\). Consigne **égale** au calendrier (pas
d'anticipation). Sur la mesure \(y\) :

- allume si \(y \le T_{\mathrm{conf}}-n/2\) ;
- éteint si \(y \ge T_{\mathrm{conf}}+n/2\) ;
- sinon garde l'état.

Avec \(n=1\,^\circ\mathrm{C}\) : bande \(19{,}5\)–\(20{,}5\) le jour
si \(T_{\mathrm{conf}}=20\).

Le MPC a le droit de monter \(T_{\mathrm{sp}}\) **avant** une heure
pleine ou une nuit froide.

---

## 9. Évaluation en boucle fermée

Chaîne :

```text
T_conf(t), π(t)          calendriers
T_ext, S                 oracle
        │
        ▼
   MPC  →  T_sp  →  bande n  →  P  →  plant (α_s,mass)
        ▲                              │
        └──────── Kalman(y) ←──────────┘
```

Métriques (à la place d'un `P_hours` opaque) :

\[
\mathrm{facture}
=
\sum_k \pi_k\,\beta\,P_k\,\Delta t_{\mathrm{s}}
\]

\[
\text{heures sous confort}
=
\Delta t_{\mathrm{h}}
\sum_k
\mathbf{1}_{\{T_{\mathrm{conf},k}-T_{\mathrm{air},k} > 0{,}1\,^\circ\mathrm{C}\}}
\]

Le seuil \(0{,}1\,^\circ\mathrm{C}\) est la résolution du capteur :
\(0{,}05\,^\circ\mathrm{C}\) sous \(T_{\mathrm{conf}}\) n'est pas du
froid. Énergie kWh proxy : \(\sum \beta P\Delta t_{\mathrm{s}}\).

Scénario type README : 48 h dès 18 h locale. Les PNG de la boucle
fermée sont ceux des notebooks (`03-mpc-cout-consigne`).

Résultat figé (48 h, 18 h) : MPC **1,8 h** sous confort vs hystérésis
**19,8 h** ; facture \(16{,}19\) vs \(16{,}35\) € (\(\approx -1\,\%\)).
Le gain n'est pas la facture, c'est de ne pas rester froid.

---

## 10. Ce que chaque couche *est*

| Couche | Objet | Programme | Diagnostic |
|---|---|---|---|
| Modèle | ODE / \(A,B,C\) | — | \(\tau\), \(Z(j\omega)\), graphe (où entre \(S\)) |
| Stats | \(\theta\) sur une trajectoire | \(\min(-\log L)\) PEM | RMSE 1–24 h, ACF \(e_k\), plafond \(R_{ae}\) |
| Commande | \(T_{\mathrm{sp}}\) en ligne | \(\min J\) QP condensé | facture €, heures \(>0{,}1\,^\circ\mathrm{C}\) sous \(T_{\mathrm{conf}}\) |

Le Nelder-Mead du fit n'est pas le SLSQP du MPC. Le RMSE d'identification
n'est pas \(J\). Le R2C2 salon n'est pas le plant.
