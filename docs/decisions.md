# Décisions

| Date | Décision | Alternative envisagée | Raison |
|---|---|---|---|
| 2026-08-30 | Métier Machine learning, domaine Bâtiment, stack Python / NumPy / SciPy | Domaine Énergie ; stack avec cvxpy / MLflow | Le recruteur coche un poste ML (baseline, validation). L'objet est la maison. SciPy suffit en v1. |
| 2026-08-30 | Noms **R1C1** (baseline) et **R2C2** (cible) | R1C / R2C comme dans le cadrage manuscrit | Les équations du brief sont 1R1C et 2R2C. Un seul vocabulaire dans le code et les slides. |
| 2026-08-30 | v1 mono-zone salon | Moyenne toutes pièces ; multi-zone d'emblée | Un capteur, une consigne, moins de fuites d'identifiabilité. Multi-zone hors v1. |
| 2026-08-30 | Construire `P` à partir du circuit d'eau et des consignes | Attendre une puissance mesurée (absente) | Sans entrée chauffage, ni identification ni MPC. Documenter que ce n'est pas des watts. |
| 2026-08-30 | Plant **codé**, distinct du modèle identifié | Rejouer le MPC sur le RC appris | Circularité : le contrôleur se prédit lui-même. Le brief phare l'interdit comme validation. |
| 2026-08-30 | Kalman à la main | `filterpy` / `pykalman` | Preuve de compréhension ; test unitaire sur cas synthétique. |
| 2026-08-30 | Hors v1 : Airflow, S3, MLflow | Suivre le brief phare à la lettre | Dilue le récit Kalman / RC / MPC. Log d'expériences : `experiments/runs.jsonl`. |
| 2026-08-30 | Interpolation limitée à 10 min sur maille 5 min | Interpoler tous les trous ; ou ne jamais interpoler | Un dt ≈ 600 s est un échantillon sauté. Un trou de 12 h n'est pas de la physique à inventer. |
| 2026-08-30 | Quantification observée : 0,1 °C salon, 1 °C extérieur | Traiter y comme T_air continue | Les pas du capteur ne sont pas une dynamique RC. |
| 2026-08-30 | `P = max(T_eau-T_air,0)` seulement si `T_air < consigne` | Écart eau/air seul ; ou consigne seule | Sans appel de zone, l'eau chaude du circuit n'est pas un apport au salon. Pas des watts : le RC absorbe l'échelle. |
| 2026-08-30 | `S` = somme PV, négatifs à 0 | Load électrique comme gains internes | Le brief demande un proxy solaire. Le load est un autre phénomène (occupants). |
| 2026-08-30 | Plant R2C2 littérature avec `alpha_s_mass` > 0 | Même équations que l'identification ; ou R3C3 | Distingue le plant du modèle interne du MPC. Solaire sur les murs = misspecification volontaire. |
| 2026-08-30 | Catalogue visuels complet (`docs/visualisations.md`) ; sélection slides plus tard | Couper Bode / phase dès maintenant | Le menu sert à produire ; RH vs technique se choisit au polish. |
| 2026-08-30 | R1C1 discret : `a`, gain `T_ext = 1-a`, `g_S`, `g_P` ; `σ_v` fixé à 0,05 °C | Identifier R,C séparés ; estimer aussi R de mesure | R et C ne sont pas séparables (P n'est pas en watts). La quantification donne l'ordre de grandeur de `v`. |
| 2026-08-30 | PEM sur les 50 derniers jours du train, pas toute l'année | Fit sur les 73 k pas de train | Boucle Python du Kalman : 50 jours suffisent et tombent en saison de chauffage. Les RMSE sont sur tout le test. |
| 2026-08-30 | Schémas RC générés (matplotlib, PNG+PDF) | TikZ / draw.io à la main | Reproductibles, même palette que les graphes, PDF Type 42 pour la publication. |
| 2026-08-30 | R2C2 : C_a=1, Nelder-Mead, plafond R_ae → τ_air ≤ 139 h | BFGS ; R,C libres | BFGS partait dans des A singulières. Le plafond évite un R_ae infini ; les deux τ restent lentes — le R2C2 n'est pas un air rapide + masse. |
| 2026-08-30 | Modèle interne MPC = R2C2 plant **sans** `α_s,mass`, pas le fit maison | Réutiliser le R2C2 identifié sur le salon | τ ~ 140 h et l'échelle de P du salon ne commandent pas le plant (~3 h / ~11 h). La non-circularité tient à `α_s,mass`. |
| 2026-08-30 | MPC : QP condensé SciPy, horizon 6 h, blocs 30 min ; météo oracle | cvxpy ; prévisions imparfaites | SciPy déjà dans la stack. L'oracle est une limite à dire : on mesure le gain du modèle, pas d'un prévisionniste. |
| 2026-08-30 | Slides RH : S6+S4 ; technique : Kalman+I1+S5 | Reprendre I1 côté RH ; Bode / Nyquist | RMSE brut décroche un RH. Le technique a besoin du Kalman et de la commande, pas du même barplot confort. |
| 2026-08-30 | Contrôle v1.1 : consigne \(T_{\mathrm{sp}}\), bande \(n=1\) °C, \(J\) = facture HP/HC + inconfort ; pas de PV dans l'euro | Commander \(P\) ; fioul ; PV dans la facture ; trois poids SciPy | Un thermostat reçoit une consigne. L'euro et le calendrier se lisent. Le PV dans \(J\) imposerait des prévisions de production. Détail : `brief/controle-mpc.md`. |
