# Modélisation thermique grey-box et contrôle prédictif d'un bâtiment

*Projet phare — profil machine learning / contrôle*

## Contexte et problématique

Le pilotage intelligent du chauffage est un des leviers les plus directs pour réduire la consommation énergétique des bâtiments, mais il suppose de disposer d'un modèle thermique à la fois fidèle et exploitable en temps réel. Les modèles purement physiques (RC) sont interprétables mais rarement calibrés avec précision sans identification par les données ; les modèles purement data-driven sont précis sur les données d'entraînement mais peu interprétables et fragiles hors distribution. L'approche grey-box — un modèle physique dont les paramètres sont inférés statistiquement — combine les deux.

Ce projet répond à la question :

> Peut-on inférer un modèle thermique RC crédible à partir de données de capteurs réelles, puis l'utiliser comme socle d'un contrôleur prédictif (MPC) capable d'anticiper les besoins de chauffage ?

## Objectif

Construire un modèle thermique grey-box (RC inféré par les données) avec modélisation explicite du bruit de capteur, utiliser un filtre de Kalman à la fois pour l'identification des paramètres et pour l'estimation d'état en temps réel, puis implémenter un contrôleur MPC utilisant ce modèle calibré comme modèle interne de prédiction.

## Compétences démontrées

- Modélisation physique (réseau RC, représentation espace d'état)
- Identification de système grey-box (estimation de paramètres par méthode de l'erreur de prédiction / maximum de vraisemblance)
- Filtrage bayésien (filtre de Kalman, implémentation et diagnostic)
- Modélisation de capteurs (bruit de mesure, équation d'observation)
- Contrôle optimal à horizon glissant (formulation MPC, contraintes, fonction de coût)
- Rigueur scientifique : validation, quantification de l'incertitude, limites explicitées
- MLOps léger : suivi d'expériences (MLflow), conteneurisation (Docker), orchestration illustrée (Airflow), stockage cloud (AWS S3)

## Approche et choix techniques

- **Modèle** : réseau RC mono-zone dans un premier temps (extension multi-zone en bonus), formulé en espace d'état continu puis discrétisé.
- **Identification** : les paramètres RC sont estimés par méthode de l'erreur de prédiction — la vraisemblance des innovations produites par le filtre de Kalman guide l'optimisation des paramètres (identification grey-box classique).
- **Modèle de capteur** : bruit de mesure explicite dans l'équation d'observation (`y = Cx + v`), distinguant température physique réelle et température mesurée.
- **Contrôle** : MPC à horizon glissant, utilisant le modèle RC calibré comme modèle de prédiction interne et le filtre de Kalman comme estimateur d'état à chaque pas de contrôle.
- **Validation en boucle fermée** : en l'absence de générateur/simulateur externe, le modèle RC calibré sert lui-même de "plant" de substitution pour tester le contrôleur. **Cette limite doit être explicitement documentée** : il s'agit d'un test de cohérence interne, pas d'une validation indépendante. Une piste bonus consiste à confronter le contrôleur à un modèle RC de référence issu de la littérature, distinct du modèle appris, pour une validation moins circulaire.

## Source de données

Dataset fourni dans le cadre académique (relevés de température et de chauffage). Le README doit documenter précisément : les variables disponibles, la fréquence d'échantillonnage, la période couverte, et toute transformation appliquée avant modélisation.

## Livrables attendus

1. Prétraitement des données : nettoyage, gestion des valeurs manquantes, vérification de la fréquence d'échantillonnage.
2. Formulation mathématique du modèle RC (équations d'état et d'observation) documentée dans le README.
3. Implémentation du filtre de Kalman (recommandé : codé à la main, pour la valeur pédagogique et la preuve de compréhension — sinon, justifier le choix d'une librairie).
4. Script d'identification des paramètres RC, avec chaque run tracé dans MLflow (paramètres, log-vraisemblance, métriques de validation).
5. Validation du modèle : comparaison prédictions vs mesures réelles, métriques d'erreur, analyse des résidus.
6. Implémentation du contrôleur MPC (horizon, contraintes, fonction de coût explicitées et justifiées).
7. Simulation en boucle fermée modèle + contrôleur, avec la limite de validation documentée en toutes lettres.
8. Conteneurisation Docker de l'environnement complet (une seule commande pour reproduire l'ensemble).
9. DAG Airflow illustrant un pipeline de ré-entraînement périodique (déploiement réel non requis, la démonstration suffit).
10. README complet : contexte, équations, méthodologie, résultats, limites, perspectives.

## Structure de repo attendue

```
projet-thermique-grey-box-mpc/
├── README.md
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── preprocessing.py
│   ├── rc_model.py
│   ├── kalman_filter.py
│   ├── identification.py
│   ├── mpc_controller.py
│   └── simulate_closed_loop.py
├── notebooks/
├── docker/
│   └── Dockerfile
├── airflow/
│   └── dags/
│       └── retrain_pipeline.py
├── tests/
│   └── test_kalman_filter.py
└── requirements.txt
```

## Règles strictes de professionnalisme

- Environnement figé et reproductible (`requirements.txt` avec versions épinglées, ou `pyproject.toml` + lock file) ; le projet doit s'installer et tourner via une seule commande documentée dans le README.
- Aucune donnée brute, secret, clé API ou identifiant AWS commité dans le repo — `.gitignore` strict, variables d'environnement pour toute configuration sensible.
- Commits atomiques avec messages conventionnels (`feat:`, `fix:`, `docs:`, `test:`...).
- Toute fonction de calcul non triviale (filtre de Kalman, coût MPC, identification) documentée par une docstring précisant entrées, sorties et hypothèses.
- Au moins un test unitaire sur le filtre de Kalman (par exemple : vérifier la convergence sur un cas synthétique où l'état vrai est connu).
- Toute limite méthodologique explicitement mentionnée dans le README — en particulier la circularité du test en boucle fermée décrite plus haut. Pas de survente des résultats.
- Le README doit permettre à quelqu'un d'extérieur au domaine de comprendre l'essentiel en 5 minutes, avant d'entrer dans le détail mathématique pour qui veut creuser.

## Pour aller plus loin (optionnel)

- Extension à un modèle multi-zone.
- Filtre de Kalman étendu ou unscented si des non-linéarités sont introduites (ex. gains solaires).
- Comparaison quantifiée du MPC avec un contrôleur simple (seuils, bang-bang) pour chiffrer le gain apporté par la formulation optimale.
