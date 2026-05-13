# 🌞 SolarAI Diagnostic System

Système de diagnostic intelligent pour panneaux photovoltaïques basé sur l'intelligence artificielle.  
Projet de Fin d'Année — Génie Maintenance & Instrumentation — INSAT Tunis (2025–2026).

---

## Description

**SolarAI Diagnostic** est une application web complète qui permet de diagnostiquer automatiquement l'état de santé d'un panneau solaire photovoltaïque en temps réel. À partir de **16 paramètres électriques et environnementaux**, le système :

- **Identifie la classe de panne** parmi 8 catégories (ombrage, court-circuit, PID, encrassement…)
- **Détecte les comportements anormaux** via un score d'anomalie de 0 à 100
- **Estime la durée de vie restante** (RUL — Remaining Useful Life) du panneau
- **Génère un rapport PDF professionnel** de 5 pages téléchargeable

Le projet combine un **backend Flask** exposant une API REST, trois modèles d'IA entraînés sur 6 000 échantillons synthétiques, et un **frontend HTML/CSS/JS** au design industriel sombre.

---

## Architecture Technique

```
┌─────────────────────────────────────────────────────────────────┐
│                     FRONTEND (HTML/CSS/JS)                      │
│  index.html    diagnostic.html    history.html    metrics.html  │
│  Dashboard KPI  16 sliders + SVG  Tableau + CSV   Accuracy F1  │
└──────────────────────────────┬──────────────────────────────────┘
                               │ HTTP REST (fetch / CORS)
                               │ localhost:5001/api
┌──────────────────────────────▼──────────────────────────────────┐
│                       API FLASK (app.py)                        │
│  /health  /diagnose  /batch-diagnose  /history  /demo-data      │
│  /generate-report  /download-report  /model-metrics             │
└──────┬───────────────────┬───────────────────┬──────────────────┘
       │                   │                   │
┌──────▼──────┐   ┌────────▼──────┐   ┌────────▼───────┐
│  XGBoost    │   │  Isolation    │   │  Régression    │
│ Classifieur │   │  Forest       │   │  Polynomiale   │
│ 8 classes   │   │  Anomalie     │   │  RUL (années)  │
│ F1 = 99.92% │   │  Score 0–100  │   │  25 ans max    │
└──────┬──────┘   └────────┬──────┘   └────────┬───────┘
       │                   │                   │
┌──────▼───────────────────▼───────────────────▼───────────────────┐
│              Feature Engineering (22 features)                   │
│  16 brutes (irradiance, VOC, ISC, MPP…)                         │
│   + 6 dérivées (ratio_vmpp_voc, perf_ratio, delta_temp…)        │
└──────────────────────────────────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                    DATASET (6 000 lignes)                        │
│  750 échantillons × 8 classes · Split 80/20 · random_state=42  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Technologies Utilisées

| Composant            | Technologie           | Version  |
|----------------------|-----------------------|----------|
| API REST             | Flask + Flask-CORS    | 3.0.3    |
| Classifieur IA       | XGBoost               | 2.0.3    |
| Détection anomalie   | Scikit-learn (IForest)| 1.5.0    |
| Traitement données   | NumPy + Pandas        | 1.26 / 2.2 |
| Génération PDF       | ReportLab             | 4.2.0    |
| Graphiques PDF       | Matplotlib            | 3.9.0    |
| Base de données      | SQLite (stdlib)       | —        |
| Frontend             | HTML5 / CSS3 / JS ES6 | —        |
| Graphiques frontend  | Chart.js              | 4.4.0    |
| Typographies         | Rajdhani, JetBrains Mono, Inter | — |

---

## Installation et Lancement

### Prérequis
- Python 3.10+
- Git

### Démarrage rapide (Linux / macOS)

```bash
git clone https://github.com/votre-repo/solar-ai-diagnostic.git
cd solar-ai-diagnostic
bash start.sh
```

### Démarrage rapide (Windows)

```bat
git clone https://github.com/votre-repo/solar-ai-diagnostic.git
cd solar-ai-diagnostic
start.bat
```

Le script :
1. Crée un environnement virtuel Python isolé (`.venv`)
2. Installe toutes les dépendances automatiquement
3. Entraîne les modèles IA si nécessaire (première exécution ~30 s)
4. Choisit automatiquement un port libre (à partir de 5001)
5. Démarre le serveur — frontend **et** API sur la même URL

Ouvrir ensuite **http://localhost:5001** dans le navigateur.

> Le frontend et l'API sont servis par le même serveur Flask — il n'y a **pas** de second serveur HTTP à lancer.

### Démarrage manuel (sans script)

```bash
# 1. Créer et activer l'environnement virtuel
python3 -m venv .venv
source .venv/bin/activate      # Windows : .venv\Scripts\activate.bat

# 2. Installer les dépendances
pip install -r backend/requirements.txt

# 3. Lancer le serveur (frontend + API)
cd backend
python app.py
```

Ouvrir **http://localhost:5001** dans le navigateur.

### Tests automatiques

```bash
python backend/tests/test_api.py
```

---

## Résultats Obtenus

| Métrique              | Valeur       | Détail                              |
|-----------------------|--------------|-------------------------------------|
| Accuracy globale      | **99.92 %**  | Jeu de test isolé (1 200 lignes)    |
| F1-Score Macro        | **99.92 %**  | Moyenne non pondérée — 8 classes    |
| F1-Score Pondéré      | **99.92 %**  | Pondéré par le support de chaque classe |
| Temps de réponse API  | **< 200 ms** | Diagnostic complet (local)          |
| Taille du dataset     | **6 000**    | 750 × 8 classes · synthétique       |
| Features utilisées    | **22**       | 16 brutes + 6 ingénierie            |

### F1-Score par classe

| Classe                  | F1-Score |
|-------------------------|----------|
| Normal                  | 100.00 % |
| Ombrage partiel         | 100.00 % |
| Court-circuit           | 100.00 % |
| Circuit ouvert          | 100.00 % |
| Dégradation PID         | 100.00 % |
| Encrassement            | 100.00 % |
| Défaut connexion        | 99.67 %  |
| Vieillissement accéléré | 99.33 %  |

---

## Description des Algorithmes IA

### XGBoost — Classifieur de pannes

XGBoost (Extreme Gradient Boosting) est un algorithme d'apprentissage par ensemble basé sur le gradient boosting. Il construit séquentiellement des arbres de décision, chaque nouvel arbre corrigeant les erreurs du précédent. Sa régularisation intégrée (L1/L2) le rend robuste au surapprentissage. Dans ce projet, il classifie les pannes parmi 8 catégories en s'appuyant sur les 22 features physiques du panneau. La feature la plus discriminante est le **ratio Vmpp/Voc** (33.6 % d'importance), qui résume la qualité de la courbe courant-tension (IV).

### Isolation Forest — Détection d'anomalies

L'Isolation Forest isole les anomalies en construisant des arbres aléatoires et en mesurant la profondeur à laquelle un point est isolé. Plus un point est facile à isoler (profondeur faible), plus il est anormal. L'algorithme est non supervisé : il n'a pas besoin de labels pour détecter des comportements atypiques. Il retourne un **score d'anomalie de 0 à 100** — les valeurs > 60 déclenchent une alerte, > 80 une alerte critique. Il est particulièrement adapté aux anomalies rares.

### Régression Polynomiale — Prédiction RUL

Le prédicteur de durée de vie restante (Remaining Useful Life) utilise une régression polynomiale de degré 2 ajustée sur la relation entre les paramètres de dégradation observés (efficacité, résistance série, fill factor) et le temps de fonctionnement restant estimé. Le modèle est entraîné sur les cas normaux et de vieillissement pour apprendre la courbe de dégradation. Il retourne une estimation en **heures** et en **années** (base 25 ans nominale), avec un statut Bon / Surveiller / Critique.

---

## Structure du Projet

```
solar-ai-diagnostic/
│
├── backend/                        # API Flask + modèles IA
│   ├── app.py                      # Application Flask — 10 endpoints REST
│   ├── requirements.txt            # Dépendances Python
│   │
│   ├── models/                     # Algorithmes IA
│   │   ├── fault_classifier.py     # Classifieur XGBoost — 8 classes de pannes
│   │   ├── anomaly_detector.py     # Isolation Forest — score d'anomalie 0–100
│   │   ├── predictor.py            # Régression polynomiale — estimation RUL
│   │   ├── train_all.py            # Entraînement complet des 3 modèles
│   │   └── saved/                  # Modèles sérialisés (.joblib)
│   │
│   ├── utils/                      # Utilitaires
│   │   ├── feature_engineering.py  # 22 features + validation + normalisation UI
│   │   ├── report_generator.py     # PDF 5 pages (ReportLab + Matplotlib)
│   │   └── database.py             # Persistance SQLite des diagnostics
│   │
│   ├── data/
│   │   └── reports/                # Rapports PDF générés
│   │
│   └── tests/
│       └── test_api.py             # 12 tests automatiques tous endpoints
│
├── frontend/                       # Interface web
│   ├── index.html                  # Dashboard — KPI, production 24h, radar santé
│   ├── diagnostic.html             # Diagnostic — 16 sliders, jauge SVG, résultats
│   ├── history.html                # Historique — filtres, tableau paginé, CSV
│   ├── metrics.html                # Métriques — accuracy, F1, feature importance
│   │
│   ├── css/
│   │   ├── style.css               # Thème industriel sombre — 40+ variables CSS
│   │   ├── diagnostic.css          # Sliders, jauge SVG, barres probabilités
│   │   ├── history.css             # Tableau, filtres, modals, pagination
│   │   └── metrics.css             # KPI métriques, rapport de classification
│   │
│   └── js/
│       ├── api.js                  # Couche HTTP — toutes les requêtes vers l'API
│       ├── main.js                 # Dashboard — horloge, KPI countUp, graphiques
│       ├── charts.js               # Chart.js — production 24h, radar santé
│       ├── diagnostic.js           # Sliders sync, jauge SVG animée, PDF
│       ├── history.js              # Filtres, pagination, CSV export, modals
│       └── metrics.js              # Feature importance, rapport 8 classes
│
├── notebook/                       # Notebooks Jupyter d'exploration
└── README.md                       # Ce fichier
```

---

## Endpoints API

| Méthode | Endpoint                     | Description                               |
|---------|------------------------------|-------------------------------------------|
| GET     | `/api/health`                | Statut API + modèles chargés + nom modèle |
| POST    | `/api/diagnose`              | Diagnostic complet (classif + anomalie + RUL) |
| POST    | `/api/batch-diagnose`        | Diagnostic de plusieurs mesures en lot    |
| GET     | `/api/model-metrics`         | Accuracy, F1, rapport de classification   |
| GET     | `/api/feature-importance`    | Features triées par importance            |
| GET     | `/api/demo-data`             | 24 points de simulation journalière       |
| GET     | `/api/history`               | Historique SQLite paginé                  |
| DELETE  | `/api/history`               | Effacement complet de l'historique        |
| POST    | `/api/generate-report`       | Génération rapport PDF 5 pages            |
| GET     | `/api/download-report/<nom>` | Téléchargement du PDF généré              |

---

## Auteur

**Barhoumi Montassar**  
Étudiant en Génie Maintenance & Instrumentation (3ème année)  
Institut National des Sciences Appliquées et de Technologie — **INSAT Tunis**  
Université de Carthage — Année universitaire **2025–2026**

---

*Projet académique — dataset synthétique généré à des fins pédagogiques.*
