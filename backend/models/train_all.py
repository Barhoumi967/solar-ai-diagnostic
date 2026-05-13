"""
Script d'entraînement complet — Solar AI Diagnostic.

Entraîne dans l'ordre :
  1. FaultClassifier  (RF vs XGBoost → meilleur sauvegardé)
  2. AnomalyDetector  (Isolation Forest sur classe 0)
  3. LifetimePredictor (Régression polynomiale)

Sauvegarde tous les fichiers .pkl dans backend/models/saved/
"""

import sys
import time
from pathlib import Path

# Chemin vers le répertoire backend
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.fault_classifier import FaultClassifier
from models.anomaly_detector import AnomalyDetector
from models.predictor        import LifetimePredictor


def main():
    debut_total = time.time()

    print("\n" + "█" * 62)
    print("  SOLAR AI DIAGNOSTIC — Entraînement de tous les modèles")
    print("█" * 62)

    resultats = {}

    # ── 1. Classificateur de défauts ──────────────────────────────────
    print("\n▶ Étape 1/3 — Fault Classifier")
    t0  = time.time()
    clf = FaultClassifier()
    clf.entrainer()
    resultats["FaultClassifier"] = {
        "modele":  clf.best_name,
        "duree":   round(time.time() - t0, 1),
        "fichiers": ["best_model.pkl", "scaler.pkl",
                     "confusion_matrix.png", "feature_importance.png"],
    }

    # ── 2. Détecteur d'anomalies ──────────────────────────────────────
    print("\n▶ Étape 2/3 — Anomaly Detector")
    t0  = time.time()
    det = AnomalyDetector()
    det.entrainer()
    resultats["AnomalyDetector"] = {
        "modele":  "Isolation Forest",
        "duree":   round(time.time() - t0, 1),
        "fichiers": ["anomaly_detector.pkl"],
    }

    # ── 3. Prédicteur de durée de vie ─────────────────────────────────
    print("\n▶ Étape 3/3 — Lifetime Predictor")
    t0   = time.time()
    pred = LifetimePredictor(degre=3)
    pred.entrainer()
    resultats["LifetimePredictor"] = {
        "modele":  f"Régression polynomiale deg={pred.degre}",
        "duree":   round(time.time() - t0, 1),
        "fichiers": ["lifetime_predictor.pkl", "courbe_degradation.png"],
    }

    # ── Test rapide de chaque modèle chargé depuis disque ─────────────
    print("\n" + "─" * 62)
    print("  Vérification — chargement depuis disque")
    print("─" * 62)

    exemple = {
        "irradiance": 950, "temperature_panneau": 45, "temperature_ambiante": 28,
        "tension_voc": 41.5, "courant_isc": 10.2, "puissance_mpp": 360,
        "tension_mpp": 35.8, "courant_mpp": 10.1, "resistance_serie": 0.3,
        "resistance_shunt": 7500, "fill_factor": 76, "efficacite": 19.5,
        "temps_fonctionnement": 8000, "humidite": 35, "vitesse_vent": 5,
        "facteur_idealite": 1.2,
    }

    # Test FaultClassifier
    clf2  = FaultClassifier.charger()
    pred_clf = clf2.predict(exemple)
    print(f"  FaultClassifier  → {pred_clf['label_panne']} ({pred_clf['confiance']} %)")

    # Test AnomalyDetector
    det2     = AnomalyDetector.charger()
    pred_det = det2.get_anomaly_score(exemple)
    print(f"  AnomalyDetector  → Score {pred_det['score']}/100 — {pred_det['niveau']}")

    # Test LifetimePredictor
    pred2    = LifetimePredictor.charger()
    pred_rul = pred2.predict_rul(temps_actuel=8000, efficacite_actuelle=19.5)
    print(f"  LifetimePredictor → RUL {pred_rul['rul_annees']} ans — Statut : {pred_rul['statut']}")

    # ── Résumé final ──────────────────────────────────────────────────
    duree_totale = round(time.time() - debut_total, 1)

    print("\n" + "█" * 62)
    print("  RÉSUMÉ FINAL")
    print("█" * 62)
    for nom, info in resultats.items():
        print(f"\n  {nom}")
        print(f"    Algorithme  : {info['modele']}")
        print(f"    Durée       : {info['duree']} s")
        print(f"    Fichiers    : {', '.join(info['fichiers'])}")

    print(f"\n  Durée totale : {duree_totale} s")
    print("\n  ✅ Tous les modèles entraînés et sauvegardés")
    print("█" * 62 + "\n")


if __name__ == "__main__":
    main()
