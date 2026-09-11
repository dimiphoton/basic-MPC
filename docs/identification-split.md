# Split d'identification : pas de semaines mélangées

Note de méthode (2026-09-12). L'état du code : `temporal_split` et
`PEM_MAX_STEPS` dans `src/basic_mpc/identification/run.py`.

## Ce qui est en place

Les séries d'identification restent une **trajectoire continue** à 5 min
(`identification_5min.csv`). Le Kalman PEM n'est pas un apprentissage
sur des lignes iid.

1. **Prétraitement** — interpolation au plus 10 min ; les longs trous
   restent `NaN` (prédiction seule, pas d'update).
2. **Entrées construites** — `u = [T_ext, S, P]` avec
   `P = max(T_eau − T_air, 0)` seulement si l'air est sous la consigne ;
   `S` = somme des phases PV (pas des watts).
3. **Split 70 / 30 chronologique** — train = début, test = fin.
   Aucun mélange. Le test
   `test_split_temporel_ne_melange_pas` l'impose.
4. **PEM** — les 14 400 derniers pas du train (~50 j de chauffage),
   16 déc. 2020 → 4 fév. 2021. Même fenêtre pour R1C1 et R2C2.
5. **Métriques** — RMSE 1 / 3 / 6 / 12 / 24 h sur **tout** le test
   (4 fév. → 24 mai 2021).

Le 70 % de train n'est donc pas entièrement fitté : trop long pour la
boucle Kalman Python, et hors saison `P` est peu excité.

## Pourquoi on ne mélange pas les semaines

Découper des semaines, les tirer au sort, puis recoller une série, casse
l'objet du PEM :

- l'état (surtout la masse) est faux à chaque raccord ;
- les sauts artificiels se font passer pour de la dynamique ;
- \(R\), \(C\) ou le bruit de processus absorbent un artefact.

Le réflexe ML « diversifier train et test » suppose des échantillons
échangeables. Ici la généralisation utile est **dans le temps**, pas
« une autre semaine du même hiver déjà vue à côté ».

L'équivalent correct, si on manquait de régimes **dans** la saison de
chauffage, serait du **multi-batch** : plusieurs trajectoires continues
(semaine 3, semaine 7, …), chacune avec son burn-in, **même** \(\theta\).
Ce n'est pas un shuffle. Hors v1.

Mélanger l'été dans le même fit dilue l'excitation de `P` et suppose des
paramètres constants alors que ventilation et ouvertures changent.
Pour un MPC hiver, le bloc froid continu reste le bon objet.

## Jours travaillés / non travaillés

\(R\) et \(C\) sont des propriétés du bâtiment. Ils ne changent pas le
samedi. Ce qui change, c'est le **perturbateur non mesuré** : occupants,
cuisine, consigne, parfois aération.

Une partie est déjà dans `u` : `P` ne s'allume que si l'air est sous la
consigne. Le reste (gains internes) part dans le bruit de processus.
Deux RC séparés (ouvré / week-end) identifieraient **deux maisons** pour
absorber l'occupation — ce n'est pas grey-box.

Ordre si on rouvre le sujet :

1. **Diagnostic** — innovations et RMSE 1 h / 24 h par jour de la
   semaine, et chauffage ON/OFF. Pas un nouveau split d'emblée.
2. Biais **systématique** le week-end → une entrée d'occupation (dummy,
   calendrier ; éventuellement le load électrique, écarté comme proxy
   solaire). Pas un second jeu de \(R\), \(C\).
3. Le PEM reste sur des **jours calendaires d'affilée**, week-ends
   compris : c'est le régime que le filtre doit vivre.

Sans occupation mesurée, stratifier le fit appauvrit la trajectoire plus
qu'il n'identifie la physique.

## Le vrai trou du split actuel

Ce n'est pas l'absence de mélange. C'est **hiver pour le fit, printemps
pour le test**, plus un gain solaire quasi non identifiable (fenêtre
décembre–février).

Si on itère :

- garder une (ou quelques) **fenêtre(s) continue(s) de chauffage** pour
  le PEM ;
- tenir à l'écart **une ou deux semaines de la même saison** pour juger
  le modèle là où le MPC s'en sert ;
- conserver fév.–mai comme **stress saisonnier**, pas comme seul juge.

Rien de tout ça n'est dans le code v1. La décision est de **ne pas**
mélanger pour « diversifier », et de **ne pas** fitter un RC par type de
jour.
