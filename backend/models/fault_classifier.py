"""
Classificateur de défauts photovoltaïques.

Compare RandomForestClassifier et XGBClassifier, sélectionne automatiquement
le meilleur selon le F1-score macro, affiche les métriques complètes et
sauvegarde le modèle gagnant + scaler.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # rendu sans interface graphique
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
from xgboost import XGBClassifier

# Ajout du répertoire backend au path pour les imports internes
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from utils.feature_engineering import engineer_features

# ── Labels des 8 classes ─────────────────────────────────────────────
FAULT_LABELS = {
    0: "Normal",
    1: "Ombrage partiel",
    2: "Court-circuit",
    3: "Circuit ouvert",
    4: "Dégradation PID",
    5: "Encrassement",
    6: "Défaut connexion",
    7: "Vieillissement accéléré",
}

SAVE_DIR   = Path(__file__).parent / "saved"
CSV_PATH   = ROOT / "data" / "solar_faults.csv"

FEATURE_COLS = [
    "irradiance", "temperature_panneau", "temperature_ambiante",
    "tension_voc", "courant_isc", "puissance_mpp", "tension_mpp",
    "courant_mpp", "resistance_serie", "resistance_shunt", "fill_factor",
    "efficacite", "temps_fonctionnement", "humidite", "vitesse_vent",
    "facteur_idealite",
    # Features dérivées calculées par engineer_features
    "performance_ratio", "delta_temperature", "ratio_impp_isc",
    "ratio_vmpp_voc", "rs_normalise", "score_vieillissement",
]


class FaultClassifier:
    """
    Classificateur de défauts PV.

    Entraîne RF et XGBoost, garde le meilleur selon F1 macro,
    expose predict() pour l'API Flask.
    """

    def __init__(self):
        self.scaler        = StandardScaler()
        self.best_model    = None        # modèle sélectionné après comparaison
        self.best_name     = ""          # "RandomForest" ou "XGBoost"
        self.feature_names = FEATURE_COLS
        self.X_test        = None        # conservé pour les graphiques
        self.y_test        = None

    # ── Chargement & prétraitement ────────────────────────────────────
    def _charger_donnees(self) -> tuple:
        """Charge le CSV, enrichit les features, applique le StandardScaler."""
        df = pd.read_csv(CSV_PATH)
        df = engineer_features(df)

        X = df[FEATURE_COLS]
        y = df["classe"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        # Fit du scaler uniquement sur le train pour éviter la fuite de données
        X_train_sc = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=FEATURE_COLS,
        )
        X_test_sc = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=FEATURE_COLS,
        )

        return X_train_sc, X_test_sc, y_train.reset_index(drop=True), y_test.reset_index(drop=True)

    # ── Entraînement et comparaison ───────────────────────────────────
    def entrainer(self) -> None:
        """Entraîne RF et XGBoost, sélectionne le meilleur et affiche les métriques."""
        print("\n" + "=" * 62)
        print("  FAULT CLASSIFIER — Entraînement")
        print("=" * 62)

        X_train, X_test, y_train, y_test = self._charger_donnees()
        self.X_test = X_test
        self.y_test = y_test

        # ── Définition des deux modèles ──
        modeles = {
            "RandomForest": RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
            "XGBoost": XGBClassifier(
                n_estimators=300,
                learning_rate=0.1,
                max_depth=6,
                use_label_encoder=False,
                eval_metric="mlogloss",
                random_state=42,
                n_jobs=-1,
            ),
        }

        resultats = {}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for nom, modele in modeles.items():
            print(f"\n[{nom}] Entraînement en cours…")
            modele.fit(X_train, y_train)
            y_pred = modele.predict(X_test)

            # Métriques sur le jeu de test
            acc  = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
            rec  = recall_score(y_test, y_pred, average="macro", zero_division=0)
            f1   = f1_score(y_test, y_pred, average="macro", zero_division=0)

            # Validation croisée 5-fold (F1 macro)
            cv_scores = cross_val_score(modele, X_train, y_train, cv=cv,
                                        scoring="f1_macro", n_jobs=-1)

            resultats[nom] = {"modele": modele, "f1": f1, "cv": cv_scores}

            print(f"  Accuracy         : {acc*100:.2f} %")
            print(f"  Precision macro  : {prec*100:.2f} %")
            print(f"  Recall macro     : {rec*100:.2f} %")
            print(f"  F1 macro (test)  : {f1*100:.2f} %")
            print(f"  CV F1 5-fold     : {cv_scores.mean()*100:.2f} % ± {cv_scores.std()*100:.2f} %")
            print()
            print(classification_report(
                y_test, y_pred,
                target_names=list(FAULT_LABELS.values()),
            ))

        # ── Sélection du meilleur selon F1 macro ──
        self.best_name  = max(resultats, key=lambda k: resultats[k]["f1"])
        self.best_model = resultats[self.best_name]["modele"]

        print("\n" + "─" * 62)
        print(f"  Meilleur modèle : {self.best_name}")
        print(f"  F1 macro        : {resultats[self.best_name]['f1']*100:.2f} %")
        print("─" * 62)

        # ── Graphiques ──
        self._afficher_confusion(self.best_model, X_test, y_test)
        self._afficher_feature_importance(self.best_model)

        # ── Sauvegarde ──
        self.sauvegarder()

    # ── Matrice de confusion ──────────────────────────────────────────
    def _afficher_confusion(self, modele, X_test, y_test) -> None:
        y_pred = modele.predict(X_test)
        cm     = confusion_matrix(y_test, y_pred)
        noms   = list(FAULT_LABELS.values())

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f"Matrice de confusion — {self.best_name}", fontsize=13)

        # Valeurs brutes
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=noms, yticklabels=noms, ax=axes[0])
        axes[0].set_title("Comptes bruts")
        axes[0].set_xlabel("Prédit"); axes[0].set_ylabel("Réel")
        axes[0].tick_params(axis="x", rotation=35)

        # Normalisée
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=noms, yticklabels=noms, vmin=0, vmax=1, ax=axes[1])
        axes[1].set_title("Recall par classe (normalisée)")
        axes[1].set_xlabel("Prédit"); axes[1].set_ylabel("Réel")
        axes[1].tick_params(axis="x", rotation=35)

        plt.tight_layout()
        out = SAVE_DIR / "confusion_matrix.png"
        plt.savefig(out, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  Matrice de confusion sauvegardée → {out}")

    # ── Feature importance ────────────────────────────────────────────
    def _afficher_feature_importance(self, modele) -> None:
        # XGBoost et RandomForest exposent tous deux feature_importances_
        importances = pd.Series(
            modele.feature_importances_, index=FEATURE_COLS
        ).sort_values(ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(10, 6))
        importances[::-1].plot(kind="barh", ax=ax, color="#f5a623", edgecolor="#252d3f")
        ax.set_title(f"Top 10 features — {self.best_name}")
        ax.set_xlabel("Importance (réduction d'impureté Gini)")
        ax.grid(axis="x", alpha=0.4)

        for bar, val in zip(ax.patches, importances[::-1].values):
            ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{val*100:.1f}%", va="center", fontsize=9)

        plt.tight_layout()
        out = SAVE_DIR / "feature_importance.png"
        plt.savefig(out, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  Feature importance sauvegardée → {out}")

    # ── Sauvegarde ────────────────────────────────────────────────────
    def sauvegarder(self) -> None:
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.best_model, SAVE_DIR / "best_model.pkl")
        joblib.dump(self.scaler,     SAVE_DIR / "scaler.pkl")
        joblib.dump(FEATURE_COLS,    SAVE_DIR / "feature_list.joblib")
        print(f"  best_model.pkl  → {SAVE_DIR / 'best_model.pkl'}")
        print(f"  scaler.pkl      → {SAVE_DIR / 'scaler.pkl'}")

    # ── Prédiction via l'API ──────────────────────────────────────────
    def predict(self, donnees_dict: dict) -> dict:
        """
        Prédit la classe de défaut à partir d'un dictionnaire de mesures brutes.

        Retourne :
            classe         : int (0-7)
            label_panne    : str
            confiance      : float (%)
            probabilites   : dict {label: proba%}
        """
        if self.best_model is None:
            raise RuntimeError("Modèle non entraîné — appeler entrainer() d'abord.")

        # Reconstruction DataFrame avec enrichissement
        df_raw = pd.DataFrame([donnees_dict])
        df_eng = engineer_features(df_raw)
        X      = df_eng[FEATURE_COLS]
        X_sc   = pd.DataFrame(self.scaler.transform(X), columns=FEATURE_COLS)

        classe = int(self.best_model.predict(X_sc)[0])
        probas = self.best_model.predict_proba(X_sc)[0]

        return {
            "classe":       classe,
            "label_panne":  FAULT_LABELS[classe],
            "confiance":    round(float(probas[classe]) * 100, 2),
            "probabilites": {
                FAULT_LABELS[i]: round(float(p) * 100, 2)
                for i, p in enumerate(probas)
            },
        }

    # ── Chargement depuis disque ──────────────────────────────────────
    @classmethod
    def charger(cls) -> "FaultClassifier":
        fc = cls()
        fc.best_model = joblib.load(SAVE_DIR / "best_model.pkl")
        fc.scaler     = joblib.load(SAVE_DIR / "scaler.pkl")
        fc.best_name  = "chargé depuis disque"
        return fc
