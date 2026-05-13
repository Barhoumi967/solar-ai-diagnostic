"""
Entraînement complet du pipeline de diagnostic PV.

Lance : python train.py
Sauvegarde les modèles dans backend/models/saved/
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Chemin racine backend
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from models.fault_classifier import FaultClassifier, FAULT_LABELS
from models.anomaly_detector import AnomalyDetector
from utils.feature_engineering import engineer_features

SAVE_DIR = ROOT / "models" / "saved"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = ROOT / "data" / "solar_faults.csv"

FEATURE_COLS = [
    "irradiance", "temperature_panneau", "temperature_ambiante",
    "tension_voc", "courant_isc", "puissance_mpp", "tension_mpp",
    "courant_mpp", "resistance_serie", "resistance_shunt", "fill_factor",
    "efficacite", "temps_fonctionnement", "humidite", "vitesse_vent",
    "facteur_idealite",
    # Features dérivées (ajoutées par engineer_features)
    "performance_ratio", "delta_temperature", "ratio_impp_isc",
    "ratio_vmpp_voc", "rs_normalise", "score_vieillissement",
]


def barre(label: str, val: float, width: int = 30) -> str:
    filled = int(val / 100 * width)
    return f"{label:<28} [{'█' * filled}{'░' * (width - filled)}] {val:5.1f}%"


# ─────────────────────────────────────────────
# 1. Chargement & enrichissement des features
# ─────────────────────────────────────────────
print("\n" + "=" * 62)
print("  ENTRAÎNEMENT — Solar AI Diagnostic")
print("=" * 62)

print("\n[1/5] Chargement des données…")
df = pd.read_csv(CSV_PATH)
df = engineer_features(df)

X = df[FEATURE_COLS]
y = df["classe"]

print(f"      Dataset : {len(df)} lignes × {len(FEATURE_COLS)} features")

# ─────────────────────────────────────────────
# 2. Split train / test
# ─────────────────────────────────────────────
print("\n[2/5] Split train/test (80/20, stratifié)…")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"      Entraînement : {len(X_train)}  |  Test : {len(X_test)}")

# ─────────────────────────────────────────────
# 3. Entraînement FaultClassifier
# ─────────────────────────────────────────────
print("\n[3/5] Entraînement FaultClassifier (Random Forest)…")
t0 = time.time()

clf = FaultClassifier(n_estimators=300, random_state=42)
clf.fit(X_train, y_train)

duree = time.time() - t0
print(f"      Terminé en {duree:.1f}s")

# ── Évaluation ──
y_pred = clf.predict(X_test)
report = classification_report(
    y_test, y_pred,
    target_names=list(FAULT_LABELS.values()),
    output_dict=True,
)

print("\n      ── Performances par classe (jeu de test) ──")
for cls_name, metrics in report.items():
    if isinstance(metrics, dict) and "f1-score" in metrics:
        f1_pct = metrics["f1-score"] * 100
        print("      " + barre(cls_name, f1_pct))

accuracy = report["accuracy"] * 100
print(f"\n      Accuracy globale : {accuracy:.2f}%")

# ── Cross-validation 5-fold ──
print("\n      Cross-validation 5-fold en cours…")
cv_scores = cross_val_score(clf.model, X_train, y_train, cv=5, scoring="accuracy")
print(f"      CV Accuracy : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

# ── Importance des features ──
importances = pd.Series(
    clf.model.feature_importances_, index=FEATURE_COLS
).sort_values(ascending=False)

print("\n      ── Top 8 features importantes ──")
for feat, imp in importances.head(8).items():
    print("      " + barre(feat, imp * 100, width=25))

# ─────────────────────────────────────────────
# 4. Entraînement AnomalyDetector
# ─────────────────────────────────────────────
print("\n[4/5] Entraînement AnomalyDetector (Isolation Forest)…")
# Entraîné uniquement sur les données normales (classe 0)
X_normal = X_train[y_train == 0]
detector = AnomalyDetector(contamination=0.05)
detector.fit(X_normal)

# Évaluation : les classes 1-7 doivent être détectées comme anomalies (-1)
scores = detector.predict(X_test)
vrai_anomalie = (y_test != 0).astype(int)
detecte_anomalie = (scores == -1).astype(int)
recall_anomalie = (
    (detecte_anomalie & vrai_anomalie).sum() / vrai_anomalie.sum() * 100
)
print(f"      Recall anomalies : {recall_anomalie:.1f}%")

# ─────────────────────────────────────────────
# 5. Sauvegarde
# ─────────────────────────────────────────────
print("\n[5/5] Sauvegarde des modèles…")

clf_path = SAVE_DIR / "fault_classifier.joblib"
det_path = SAVE_DIR / "anomaly_detector.joblib"
scaler_path = SAVE_DIR / "feature_list.joblib"

clf.save(str(clf_path))
detector.save(str(det_path))
joblib.dump(FEATURE_COLS, str(scaler_path))

print(f"      fault_classifier.joblib  → {clf_path}")
print(f"      anomaly_detector.joblib  → {det_path}")
print(f"      feature_list.joblib      → {scaler_path}")

print("\n" + "=" * 62)
print(f"  Entraînement terminé — Accuracy : {accuracy:.2f}%")
print("  Démarrer l'API : python app.py")
print("=" * 62 + "\n")
