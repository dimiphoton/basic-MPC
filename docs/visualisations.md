# Catalogue des visualisations

Menu complet. On **produit** les figures depuis les notebooks
(`notebooks/00`…`03`, `RECOMPUTE = True`) ou le CLI équivalent.
Sortie : `pictures/experiments/` (PNG, parfois PDF). On **sélectionne**
lesquelles vont dans les slides RH / technique (3 graphes max par deck,
fond crème / transparent). L'app Streamlit n'est **pas** le registre
des expériences.

Noms de fichiers en kebab-case.

Schémas de modèles (générés, pas dessinés à la main) :

| Fichier | Contenu |
|---|---|
| `schema-r1c1` | Circuit baseline |
| `schema-r2c2` | Identification (solaire sur l'air) |
| `schema-plant` | Littérature (`α_s` aussi sur la masse) |
| `schema-kalman` | Prédiction / innovation / mise à jour |
| `schema-famille-rc` | Les trois circuits empilés |

Notebook `00-schemas-impedance` / CLI `draw-schemas` (PNG + PDF).
MPC v1.1 (€, T_sp) : notebook `03` / CLI `mpc-cout-consigne` (S4–S6).

---

## Impédance complexe (R, C) — toujours dans le menu

Un RC se lit comme \(Z(j\omega)\) : R sur l'axe réel, \(1/j\omega C\)
dans l'imaginaire.

| Id | Figure | Quand | Public probable |
|---|---|---|---|
| Z1 | Vecteur \(Z(j\omega_{24h})\) : une flèche R1C1, une R2C2 | Après identification | RH + technique |
| Z2 | Lieu de Nyquist \(Z(j\omega)\) (5 min → plusieurs jours) | Après R1C1 vs R2C2 | Technique |
| Z3 | Bode phase vs période (déphasage) | `export-simulator` | Technique + leçon Pages |

Superposer le plant littérature sur Z2 si ça clarifie l'écart volontaire
(`alpha_s_mass`).

---

## Résultats d'identification (données réelles)

| Id | Figure | Quand | Public probable |
|---|---|---|---|
| I1 | RMSE / MAE vs horizon (1, 3, 6, 12, 24 h), R1C1 vs R2C2 | Comparaison | RH (graphe clé) |
| I2 | 48 h : \(y\) vs \(\hat y\) R1C1 vs \(\hat y\) R2C2 (hiver + jour de soleil) | Comparaison | Les deux |
| I3 | Innovations Kalman : histogramme + ACF | R1C1 puis R2C2 | Technique |
| I4 | Paramètres \(R,C\) (et R2C2) + incertitude | Après chaque ident. | Technique |
| I5 | Portrait de phase \(T_\mathrm{air}\) vs \(\hat T_\mathrm{masse}\) (ou \(T\) vs \(\mathrm{d}T/\mathrm{d}t\)) | Après Kalman | Technique (option) |

---

## Simulations (plant, Kalman, MPC)

Ici l'état vrai existe.

| Id | Figure | Quand | Public probable |
|---|---|---|---|
| S1 | \(T_\mathrm{air}\) vraie vs \(y\) quantifiée (0,1 °C), une journée | Plant / Kalman | Technique |
| S2 | \(\hat T_\mathrm{masse}\) Kalman vs \(T_\mathrm{masse}\) plant | Identification sur plant | Technique |
| S3 | Même \(u\) : \(y_\mathrm{plant}\) vs \(y\) du R2C2 identifié | Après ident. | Technique (anti-circularité) |
| S4 | MPC vs hystérésis, 48 h : \(T_\mathrm{air}\) + \(T_{\mathrm{conf}}(t)\) + HP/HC | MPC € | RH + technique |
| S5 | Même 48 h : consigne \(T_{\mathrm{sp}}\) et \(P\) via bande \(n\) | MPC € | Technique |
| S6 | Facture proxy € + heures sous \(T_{\mathrm{conf}}\) (> 0,1 °C) | MPC € | RH (graphe clé) |
| Lab | Streamlit (optionnel) : météo tirée, une stratégie | hors livrable principal | Visiteur |

---

## Devoir approfondi (H1–H10)

Notebook `04-devoir-approfondi` / CLI `homework-pngs`. Pas de refit PEM.

| Id | Figure | Question que ça pose |
|---|---|---|
| H1 | Loi bande \(n\) : \(P/P_{\max}\) vs \(T_{\mathrm{sp}}-y\) | Qu'est-ce qu'on commande ? |
| H2 | Réponse indicielle plant : air vs masse | Pourquoi un 2e état ? |
| H3 | Bode \(\lvert Z/Z(0)\rvert\) plant vs fits | Les échelles sont-elles les mêmes ? |
| H4 | QQ-plot des innovations Kalman (plant) | Le bruit est-il gaussien ? |
| H5 | Trou de 4 h : \(\hat T_{\mathrm{masse}}\) sans \(y\) | Kalman ≠ interpolation |
| H6 | Split train / fenêtre PEM / test | Pourquoi pas de shuffle ? |
| H7 | MAE vs horizon R1C1 / R2C2 | Le RMSE I1 n'est pas tout |
| H8 | Pareto \(\lambda\) : facture vs heures sous \(T_{\mathrm{conf}}\) | Que minimise \(J\) ? |
| H9 | MPC oracle vs \(T_{\mathrm{ext}}\) retardé 6 h | Limite de l'oracle |
| H10 | Quantification 0,1 °C (plant, 24 h) | Capteur ≠ état |

---

## Hors menu

Pas de matrice \(A\) en slide. Pas d'interpolation d'un trou de 12 h
pour le rendu.

---

## Fichiers

- Travail : `pictures/experiments/<id>-<nom>.png`
- Slides : copie choisie dans `pictures/presentations/` (suffixe `-fr` / `-en` seulement si le titre est *dans* l'image)

Choix v1 (3 graphes max par récit, titres Marp au-dessus) :

| Deck | Graphes |
|---|---|
| Recruteur | S6 confort/conso, S4 \(T_\mathrm{air}\) 48 h |
| Technique | schéma Kalman, I1 RMSE vs horizon, S5 commande \(P\) |
