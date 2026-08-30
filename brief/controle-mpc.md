# Problème de contrôle (MPC chauffage)

Cadrage du **quoi** : décision, actionneur, coût, contraintes.
Pas du **comment** (OSQP, Streamlit) — ces moyens viennent après.

Les nombres ci-dessous sont des **ordres de grandeur figés** pour
pouvoir coder plus tard sans réouvrir le métier. Ce ne sont pas des
tarifs de fournisseur ni un essai en maison réelle.

---

## 1. Ce que v1 fait, et ce qu'on remplace

Le MPC actuel ([`src/basic_mpc/control/mpc.py`](../src/basic_mpc/control/mpc.py))
décide un `P` borné et minimise

\[
\|T - 20\|^2 + \text{hors-bande}^2 + r_u \|P\|^2
\]

avec trois poids magiques (`q_track`, `q_band`, `r_rel`). On commande
comme si `P` était un actionneur en watts. Un thermostat ne reçoit pas
des watts : il reçoit une **consigne**. Un recruteur ne lit pas `r_rel`.

**On ne touche pas** au `P` d'identification
(`P = \max(T_{\mathrm{eau}} - T_{\mathrm{air}}, 0)` si appel de zone).
C'est une reconstruction du passé, pas une commande. Les deux `P`
n'ont pas le même statut :

| | Identification | Contrôle (ce brief) |
|---|---|---|
| Sens | Proxy de ce qui a chauffé le salon en 2020 | Apport produit par la loi locale |
| Origine | Eau du circuit + consigne mesurée | Écart consigne thermostat − air |
| Unités | Kelvin d'écart, pas des W | Unités proxy du plant, converties en kWh par \(\beta\) |

---

## 2. Décision

À chaque pas de contrôle, le MPC choisit une **consigne thermostat**
\(T_{\mathrm{sp},k}\) (°C), pas une puissance.

Bornes :

\[
T_{\mathrm{sp}} \in [16,\ 22]\ \mathrm{°C}.
\]

Horizon et move blocking **inchangés** par rapport à v1 : 6 h d'avance,
commande constante par blocs de 30 min. Kalman \((T_{\mathrm{air}},
T_{\mathrm{masse}})\) inchangé.

---

## 3. Actionneur : écart consigne − mesure de \(n\) degrés

Loi locale, **bande proportionnelle** \(n\) (défaut **\(n = 1\) °C**) :

\[
P_k = P_{\max}\ \mathrm{sat}\!\left(\frac{T_{\mathrm{sp},k} - T_{\mathrm{air},k}}{n},\ 0,\ 1\right).
\]

- à la consigne : arrêt ;
- \(n\) °C en dessous : à fond ;
- entre les deux : linéaire.

\(P_{\max}\) reste celui du plant (marge × puissance de maintien au
\(T_{\mathrm{ext}}\) le plus froid du scénario), comme en v1.

Formulation équivalente pour un QP : décider \(P_k\) borné, avec

\[
T_{\mathrm{sp},k} = T_{\mathrm{air},k} + n\,\frac{P_k}{P_{\max}},
\qquad
16 \le T_{\mathrm{sp},k} \le 22.
\]

Ce sont des contraintes linéaires. Pas besoin de Gurobi pour les poser.

### Baseline (à comparer, pas à optimiser)

Même programmation de confort \(T_{\mathrm{conf}}(t)\), **hystérésis**
sur la mesure, consigne thermostat **égale** à \(T_{\mathrm{conf}}(t)\)
(pas d'anticipation). Allume sous \(T_{\mathrm{conf}} - n/2\), éteint
au-dessus de \(T_{\mathrm{conf}} + n/2\) (avec \(n = 1\) °C : bande
19,5–20,5 le jour si \(T_{\mathrm{conf}} = 20\)).

Le MPC, lui, a le droit de monter \(T_{\mathrm{sp}}\) **avant** une
heure pleine ou une nuit froide.

---

## 4. Ce que le solveur minimise

Horizon \(N\), pas \(\Delta t\) (5 min du plant).

\[
J = \sum_{k=0}^{N-1}
\Bigl[
\pi_k \cdot \beta \cdot P_k \cdot \Delta t
+
\lambda \cdot \max(T_{\mathrm{conf},k} - T_{\mathrm{air},k},\ 0)^2 \cdot \Delta t
\Bigr].
\]

Deux termes, deux unités qui se rencontrent en **euros** :

1. **Facture** — \(\pi_k\) en €/kWh, \(\beta P_k \Delta t\) en kWh.
2. **Inconfort** — sous la consigne de confort seulement (pas de pénalité
   « trop chaud » : le plafond \(T_{\mathrm{sp}} \le 22\) suffit).
   \(\lambda\) convertit des K²·h en euros.

Pas de contrainte dure sur \(T_{\mathrm{air}}\) : au démarrage à 18 °C
le problème resterait infaisable. Le confort est dans \(J\).

---

## 5. Programmation de confort \(T_{\mathrm{conf}}(t)\)

Calendrier, **pas une prévision**. Heure locale `Europe/Brussels`.

| Fenêtre | \(T_{\mathrm{conf}}\) |
|---|---|
| 07:00 ≤ t < 22:00 | 20 °C (occupé) |
| sinon | 17 °C (réduit) |

C'est le « à l'avance » : comme un thermostat programmable. Le MPC
connaît ce calendrier sur tout l'horizon ; le bang-bang le suit sans
anticiper le prochain changement.

---

## 6. Prix de l'énergie \(\pi_k\)

**Électricité, tarif bi-horaire, horloge.** Pas de fioul en v1.1.
Pas de PV dans la facture (le solaire des données reste une
perturbation **thermique** \(S\), déjà dans le plant).

| Fenêtre (Europe/Brussels) | Nom | \(\pi\) |
|---|---|---|
| 22:00 ≤ t < 07:00 | Heures creuses | 0,20 €/kWh |
| sinon | Heures pleines | 0,40 €/kWh |

Ratio 1:2, nombres ronds, **pas** un scraping de fournisseur. Ça suffit
pour que « anticiper » veuille dire **préchauffer en heures creuses**.

Fioul (prix plat) et PV dans l'euro : hors v1.1. Le fioul n'incite pas
à décaler ; le PV dans la facture impose des prévisions de production
et de l'autoconsommation — un second problème.

---

## 7. Conversion proxy → kWh (\(\beta\)) et poids confort (\(\lambda\))

Le plant a \(C\) en J/K. L'apport à l'air est \(\alpha_h P\) (watts si
\(P\) est dans les unités proxy du plant, \(\alpha_h = 4\)).

\[
E_{\mathrm{kWh}} = \beta\, P\, \Delta t_{\mathrm{s}},
\qquad
\beta = \frac{\alpha_h}{3{,}6\times 10^6} = \frac{4}{3{,}6\times 10^6}.
\]

C'est la chaleur qui entre dans le nœud air du plant littérature, **pas
un compteur**. On le dit dans le README et les slides.

\(\lambda = 1{,}0\) €·K⁻²·h⁻¹ : 1 K sous \(T_{\mathrm{conf}}\) pendant
1 h coûte 1 € d'inconfort, soit l'ordre de quelques heures de chauffage
pleines au tarif HP. Un seul nombre, à la place des trois poids SciPy.

Dans \(J\), \(\Delta t\) est en **heures** pour le terme confort (pour
que \(\lambda\) soit bien en €·K⁻²·h⁻¹) et en **secondes** pour
\(\beta P \Delta t_{\mathrm{s}}\) ; ou on convertit une fois et on
reste homogène. L'implémentation choisit une convention et la documente
dans le code.

---

## 8. Perturbations, métriques, hors-scope

**Perturbations.** \(T_{\mathrm{ext}}\) et \(S\) connus sur l'horizon
(oracle météo). Limite déjà dite en v1. \(S\) chauffe la maison ; \(S\)
ne réduit pas la facture.

**Métriques de comparaison** (à la place du `P_hours` opaque) :

- facture proxy € ;
- heures sous \(T_{\mathrm{conf}}(t)\) ;
- énergie kWh proxy.

**Hors v1.1 (ne pas coder, ne pas glisser dans \(J\))** :

- COP de pompe à chaleur ;
- prix spot EPEX ;
- occupancy détectée (le calendrier suffit) ;
- multi-zone ;
- Gurobi ;
- PV dans la facture ;
- fioul.

**Moyens, après que \(J\) existe** : reformuler le QP (OSQP si le
problème est bien posé), puis un dashboard où le recruteur joue avec
\(n\), le calendrier, et HP/HC.

---

## 9. Chaîne

```text
T_conf(t), π(t)          calendriers (connus)
T_ext, S                 oracle météo
        │
        ▼
   MPC  →  T_sp  →  bande n  →  P  →  plant
        ▲                              │
        └──────── Kalman(y) ←──────────┘
```

Baseline : \(T_{\mathrm{sp}} = T_{\mathrm{conf}}(t)\), hystérésis, pas
de modèle.
