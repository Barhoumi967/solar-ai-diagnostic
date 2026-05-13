"""
Prédicteur de durée de vie restante (RUL) des panneaux PV.

Régression polynomiale sur efficacite vs temps_fonctionnement.
Seuil critique : 80 % de l'efficacité initiale.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error

ROOT     = Path(__file__).resolve().parent.parent
SAVE_DIR = Path(__file__).parent / "saved"
CSV_PATH = ROOT / "data" / "solar_faults.csv"

# Durée de vie maximale considérée (25 ans × 8760 h/an)
DUREE_VIE_MAX_H  = 25 * 8760          # 219 000 h
SEUIL_DEGRADATION = 0.80               # 80 % de l'efficacité initiale


class LifetimePredictor:
    """
    Prédit la durée de vie restante (RUL) d'un panneau PV.

    Modèle : régression polynomiale (degré 3) entre
    temps_fonctionnement (h) et efficacite (%).
    """

    def __init__(self, degre: int = 3):
        self.degre   = degre
        self.pipeline: Pipeline | None = None
        self.eff_initiale: float = 19.5   # efficacité nominale par défaut (%)
        self.r2:  float = 0.0
        self.mae: float = 0.0

    # ── Entraînement ─────────────────────────────────────────────────
    def entrainer(self) -> None:
        """Ajuste la régression polynomiale sur les données de vieillissement."""
        print("\n" + "=" * 62)
        print("  LIFETIME PREDICTOR — Entraînement")
        print("=" * 62)

        df = pd.read_csv(CSV_PATH)

        # Classe 7 uniquement (vieillissement) : seule classe avec une tendance
        # claire de dégradation de l'efficacité dans le temps
        df_reg = df[(df["classe"] == 7) & (df["efficacite"] > 0)].copy()
        # Compléter avec quelques données normales pour ancrer la courbe à t faible
        df_normal = df[(df["classe"] == 0) & (df["efficacite"] > 0)].copy()
        df_reg = pd.concat([df_normal, df_reg], ignore_index=True)

        X = df_reg["temps_fonctionnement"].values.reshape(-1, 1)
        y = df_reg["efficacite"].values

        # Pipeline : PolynomialFeatures + régression linéaire
        self.pipeline = Pipeline([
            ("poly",    PolynomialFeatures(degree=self.degre, include_bias=False)),
            ("lineaire", LinearRegression()),
        ])
        self.pipeline.fit(X, y)

        # Métriques
        y_pred    = self.pipeline.predict(X)
        self.r2   = float(r2_score(y, y_pred))
        self.mae  = float(mean_absolute_error(y, y_pred))

        # Efficacité initiale = prédiction à t=0
        self.eff_initiale = float(self.pipeline.predict([[0]])[0])

        print(f"  Degré polynôme   : {self.degre}")
        print(f"  Échantillons     : {len(df_reg)}")
        print(f"  R²               : {self.r2:.4f}")
        print(f"  MAE              : {self.mae:.3f} %")
        print(f"  Efficacité à t=0 : {self.eff_initiale:.2f} %")
        print(f"  Seuil critique   : {self.eff_initiale * SEUIL_DEGRADATION:.2f} % ({int(SEUIL_DEGRADATION*100)} % de l'init.)")

        self._afficher_courbe_degradation()
        self.sauvegarder()

    # ── Prédiction RUL ────────────────────────────────────────────────
    def predict_rul(self, temps_actuel: float, efficacite_actuelle: float) -> dict:
        """
        Estime la durée de vie restante (RUL).

        Paramètres :
            temps_actuel       : heures de fonctionnement déjà effectuées
            efficacite_actuelle : efficacité mesurée actuellement (%)

        Retourne :
            rul_heures : float — heures restantes estimées
            rul_annees : float — années restantes
            eff_critique : float — seuil d'efficacité critique (%)
            statut      : str   — "Bon" | "Surveiller" | "Critique"
        """
        if self.pipeline is None:
            raise RuntimeError("Modèle non entraîné — appeler entrainer() d'abord.")

        eff_critique = self.eff_initiale * SEUIL_DEGRADATION

        # Recherche du temps t* où efficacite(t*) = eff_critique
        # par balayage sur une grille temporelle fine
        t_grid  = np.linspace(temps_actuel, DUREE_VIE_MAX_H, 50_000).reshape(-1, 1)
        eff_pred = self.pipeline.predict(t_grid)

        # Premier temps où l'efficacité passe sous le seuil
        indices_sous_seuil = np.where(eff_pred < eff_critique)[0]

        if len(indices_sous_seuil) == 0:
            # Le modèle ne prédit pas de franchissement dans les 25 ans
            t_fin      = DUREE_VIE_MAX_H
            rul_heures = max(0.0, t_fin - temps_actuel)
        else:
            # t_grid est de shape (N, 1) — on aplatit pour extraire un scalaire
            t_fin      = float(t_grid.flatten()[indices_sous_seuil[0]])
            rul_heures = max(0.0, t_fin - temps_actuel)

        rul_annees = rul_heures / 8760

        # Statut
        pct_duree = (temps_actuel / DUREE_VIE_MAX_H) * 100
        ratio_eff = efficacite_actuelle / self.eff_initiale
        if ratio_eff >= 0.90 and pct_duree < 60:
            statut = "Bon"
        elif ratio_eff >= SEUIL_DEGRADATION:
            statut = "Surveiller"
        else:
            statut = "Critique"

        return {
            "rul_heures":   round(rul_heures, 0),
            "rul_annees":   round(rul_annees, 2),
            "eff_critique": round(eff_critique, 2),
            "eff_actuelle": round(efficacite_actuelle, 2),
            "eff_initiale": round(self.eff_initiale, 2),
            "statut":       statut,
            "pct_vie_ecoulee": round(pct_duree, 1),
        }

    # ── Courbe de dégradation sur 25 ans ──────────────────────────────
    def _afficher_courbe_degradation(self) -> None:
        """Génère et sauvegarde la courbe de dégradation prévue sur 25 ans."""
        t_annees  = np.linspace(0, 25, 500)
        t_heures  = t_annees * 8760
        eff_pred  = self.pipeline.predict(t_heures.reshape(-1, 1))
        eff_crit  = self.eff_initiale * SEUIL_DEGRADATION

        fig, ax = plt.subplots(figsize=(11, 5))
        fig.patch.set_facecolor('#0d0f14')
        ax.set_facecolor('#161b27')

        # Courbe principale
        ax.plot(t_annees, eff_pred, color='#f5a623', lw=2.5, label='Efficacité prédite')

        # Zone critique
        ax.axhline(eff_crit, color='#e05252', lw=1.5, ls='--',
                   label=f'Seuil critique ({eff_crit:.1f} %)')
        ax.fill_between(t_annees, eff_pred, eff_crit,
                        where=eff_pred < eff_crit,
                        color='#e05252', alpha=0.15, label='Zone hors service')

        # Marqueur fin de vie
        indices = np.where(eff_pred < eff_crit)[0]
        if len(indices):
            t_fin = t_annees[indices[0]]
            ax.axvline(t_fin, color='#e05252', lw=1, ls=':')
            ax.annotate(f'Fin de vie prévue\n~{t_fin:.1f} ans',
                        xy=(t_fin, eff_crit),
                        xytext=(t_fin + 1.5, eff_crit + 1.5),
                        color='#e05252', fontsize=9,
                        arrowprops=dict(arrowstyle='->', color='#e05252'))

        ax.set_xlabel('Durée de fonctionnement (années)', color='#e8eaf0')
        ax.set_ylabel('Efficacité (%)', color='#e8eaf0')
        ax.set_title('Courbe de dégradation prévue — 25 ans', color='#e8eaf0', fontsize=13)
        ax.tick_params(colors='#7a8299')
        ax.grid(color='#252d3f', linestyle='--', alpha=0.5)
        ax.legend(fontsize=9, framealpha=0.2)
        ax.set_xlim(0, 25)

        plt.tight_layout()
        out = SAVE_DIR / "courbe_degradation.png"
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  Courbe dégradation sauvegardée → {out}")

    # ── Sauvegarde / chargement ───────────────────────────────────────
    def sauvegarder(self) -> None:
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, SAVE_DIR / "lifetime_predictor.pkl")
        print(f"  lifetime_predictor.pkl → {SAVE_DIR / 'lifetime_predictor.pkl'}")

    @classmethod
    def charger(cls) -> "LifetimePredictor":
        return joblib.load(SAVE_DIR / "lifetime_predictor.pkl")
