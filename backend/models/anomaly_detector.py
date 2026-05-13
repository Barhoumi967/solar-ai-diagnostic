"""
Détecteur d'anomalies basé sur Isolation Forest.

Entraîné uniquement sur les données normales (classe 0).
Retourne un score normalisé 0–100 avec seuils d'alerte.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from utils.feature_engineering import engineer_features

SAVE_DIR = Path(__file__).parent / "saved"
CSV_PATH = ROOT / "data" / "solar_faults.csv"

FEATURE_COLS = [
    "irradiance", "temperature_panneau", "temperature_ambiante",
    "tension_voc", "courant_isc", "puissance_mpp", "tension_mpp",
    "courant_mpp", "resistance_serie", "resistance_shunt", "fill_factor",
    "efficacite", "temps_fonctionnement", "humidite", "vitesse_vent",
    "facteur_idealite",
    "performance_ratio", "delta_temperature", "ratio_impp_isc",
    "ratio_vmpp_voc", "rs_normalise", "score_vieillissement",
]

# ── Seuils d'alerte ───────────────────────────────────────────────────
SEUILS = {
    "Normal":    (0,  30),
    "Attention": (30, 60),
    "Alerte":    (60, 80),
    "Critique":  (80, 100),
}


def _niveau(score: float) -> str:
    """Retourne le niveau d'alerte correspondant au score 0–100."""
    for niveau, (bas, haut) in SEUILS.items():
        if bas <= score < haut:
            return niveau
    return "Critique"


class AnomalyDetector:
    """
    Détecteur d'anomalies Isolation Forest.

    Entraîné sur la classe 0 (Normal) uniquement.
    Le score brut sklearn (négatif, plus bas = plus anormal) est
    renormalisé vers [0, 100] pour une lecture intuitive.
    """

    def __init__(self, contamination: float = 0.02, random_state: int = 42):
        self.model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        # Limites du score brut observées sur les données d'entraînement
        # (remplies lors du fit pour la normalisation)
        self._score_min: float = -1.0
        self._score_max: float =  0.0

    # ── Entraînement ─────────────────────────────────────────────────
    def entrainer(self) -> None:
        """Entraîne sur les données normales et calibre la normalisation."""
        print("\n" + "=" * 62)
        print("  ANOMALY DETECTOR — Entraînement")
        print("=" * 62)

        df  = pd.read_csv(CSV_PATH)
        df  = engineer_features(df)

        # Entraînement sur classe 0 uniquement
        X_normal = df[df["classe"] == 0][FEATURE_COLS]
        self.model.fit(X_normal)
        print(f"  Données normales utilisées : {len(X_normal)} échantillons")

        # Calibration des bornes sur l'ensemble complet (toutes classes)
        X_all          = df[FEATURE_COLS]
        scores_bruts   = self.model.score_samples(X_all)
        self._score_min = float(scores_bruts.min())
        self._score_max = float(scores_bruts.max())
        print(f"  Score brut min : {self._score_min:.4f}")
        print(f"  Score brut max : {self._score_max:.4f}")

        # Évaluation : recall des anomalies (classes 1-7)
        y_all   = df["classe"].values
        preds   = self.model.predict(X_all)           # -1 = anomalie, 1 = normal
        anomalies_reelles  = (y_all != 0)
        anomalies_detectees = (preds == -1)
        recall = anomalies_detectees[anomalies_reelles].mean() * 100
        fpr    = anomalies_detectees[~anomalies_reelles].mean() * 100
        print(f"  Recall anomalies : {recall:.1f} %")
        print(f"  Faux positifs    : {fpr:.1f} %")

        self.sauvegarder()

    # ── Normalisation score brut → [0, 100] ──────────────────────────
    def _normaliser(self, score_brut: float) -> float:
        """
        Convertit le score Isolation Forest (négatif) en score 0–100.
        score_brut proche de 0   → score normalisé proche de 0   (normal)
        score_brut très négatif  → score normalisé proche de 100 (anomalie sévère)
        """
        plage = self._score_max - self._score_min
        if plage == 0:
            return 0.0
        # Inversion : un score brut élevé (proche de 0) donne un score anomalie bas
        normalise = (self._score_max - score_brut) / plage * 100
        return float(np.clip(normalise, 0, 100))

    # ── API principale ────────────────────────────────────────────────
    def get_anomaly_score(self, donnees_dict: dict) -> dict:
        """
        Calcule le score d'anomalie pour une mesure PV.

        Paramètres :
            donnees_dict : dict des 16 features brutes du panneau

        Retourne :
            score   : float [0–100]
            niveau  : str  "Normal" | "Attention" | "Alerte" | "Critique"
            detecte : bool True si Isolation Forest classe comme anomalie
        """
        df_raw = pd.DataFrame([donnees_dict])
        df_eng = engineer_features(df_raw)
        X      = df_eng[FEATURE_COLS]

        score_brut  = float(self.model.score_samples(X)[0])
        score_norm  = self._normaliser(score_brut)
        est_anomalie = bool(self.model.predict(X)[0] == -1)

        return {
            "score":    round(score_norm, 1),
            "niveau":   _niveau(score_norm),
            "detecte":  est_anomalie,
            "score_brut": round(score_brut, 4),
        }

    # ── Sauvegarde / chargement ───────────────────────────────────────
    def sauvegarder(self) -> None:
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, SAVE_DIR / "anomaly_detector.pkl")
        print(f"  anomaly_detector.pkl → {SAVE_DIR / 'anomaly_detector.pkl'}")

    @classmethod
    def charger(cls) -> "AnomalyDetector":
        return joblib.load(SAVE_DIR / "anomaly_detector.pkl")
