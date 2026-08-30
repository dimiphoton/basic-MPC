# Objectif du projet

- **But** : pratiquer l'inférence statistique, puis le model predictive
  control (MPC), sur le comportement thermique d'une maison (températures
  par pièce, commande de chauffage, production PV).
- **Origine** : ancien projet ULG. Le code d'origine est archivé sur la
  branche `old` (notebooks, classes trop grosses avec arrays ou NetworkX).
  Reprise pour le portfolio : code modulaire, documenté, pas des notebooks.
- **Contraintes de départ** : jeu de données imposé (séries de températures
  par pièce, chauffage, extérieur, PV). Pipeline dès le départ.

Attendu (formulation d'origine, à affiner en roadmap) :

- modèle de réponse thermique du bâtiment (base, une pièce / toutes les
  pièces, modèle de lecture des capteurs) ;
- pipeline d'inférence statistique (modèle, optimiseur, bayésien) ;
- visualisation simple + petit dashboard ;
- simulateur MPC (prix du combustible, confort, prévision 24 h,
  incertitude, comparaison à une commande binaire).

Nice to have : générateur de données à la place du jeu brut, avec un
modèle distinct de celui inféré ensuite.

Métier, domaine et stack se remplissent dans `brief/identite.md` (source
du bandeau README / covers / topics GitHub), pas ici.
