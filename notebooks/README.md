# Notebooks d'expérience

Ils **appellent** `src/basic_mpc/` : pas de Kalman recopié, pas de RC
réécrit. Jetables pour explorer ; dès qu'une logique sert ailleurs, elle
reste dans `src/`.

Ordre suggéré (scolaire) :

| Fichier | Question |
|---|---|
| `01-modeles-rc.ipynb` | R1C1 vs R2C2 vs plant (schémas) |
| `02-modele-capteur.ipynb` | \(y = T + v\), quantification 0,1 °C |
| `03-kalman-r1c1.ipynb` | Le filtre bat-il la mesure brute ? |
| `04-kalman-masse-cachee.ipynb` | Inférer les murs jamais mesurés |
| `05-plant-vs-identifie.ipynb` | Pourquoi pas de circularité |
| `06-innovations-et-nll.ipynb` | Bruit blanc ? PEM |
| `07-impedance-dephasage.ipynb` | Vecteur \(Z\), retard 24 h |
| `08-rapports-cli.ipynb` | Lire les JSON déjà calculés |

Texte : [`docs/lecon-rc-kalman.md`](../docs/lecon-rc-kalman.md).
Leçon Pages : schémas + capteur + Kalman.
Labo stratégies : `streamlit run webapp/app.py`.

Prérequis : `pip install -e ".[dev]"` (et un kernel Jupyter).
