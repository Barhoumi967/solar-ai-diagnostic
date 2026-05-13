"""
Ingénierie de features pour le diagnostic photovoltaïque.

Fournit :
  - engineer_features()        : ajoute les 6 features dérivées à un DataFrame
  - validate_input()           : vérifie la présence et les plages des 16 features
  - compute_derived_features() : calcule les features dérivées depuis un dict brut
  - normalize_for_display()    : normalise une valeur pour l'affichage UI (0–100)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── Définition des 16 features brutes ──────────────────────────────────
# Chaque feature : (valeur_min, valeur_max, unité, description)
FEATURE_SPECS: dict[str, tuple[float, float, str, str]] = {
    "irradiance":           (0,      1200,  "W/m²", "Irradiance solaire reçue"),
    "temperature_panneau":  (-10,    100,   "°C",   "Température de la cellule"),
    "temperature_ambiante": (-20,    60,    "°C",   "Température de l'air ambiant"),
    "tension_voc":          (0,      60,    "V",    "Tension à circuit ouvert"),
    "courant_isc":          (0,      15,    "A",    "Courant de court-circuit"),
    "puissance_mpp":        (0,      500,   "W",    "Puissance au point MPP"),
    "tension_mpp":          (0,      55,    "V",    "Tension au point MPP"),
    "courant_mpp":          (0,      15,    "A",    "Courant au point MPP"),
    "resistance_serie":     (0,      10,    "Ω",    "Résistance série interne"),
    "resistance_shunt":     (1,      50000, "Ω",    "Résistance shunt (parallèle)"),
    "fill_factor":          (0,      100,   "%",    "Fill Factor de la courbe IV"),
    "efficacite":           (0,      30,    "%",    "Efficacité de conversion"),
    "temps_fonctionnement": (0,      100000,"h",    "Heures de fonctionnement cumulées"),
    "humidite":             (0,      100,   "%",    "Humidité relative ambiante"),
    "vitesse_vent":         (0,      50,    "m/s",  "Vitesse du vent"),
    "facteur_idealite":     (1.0,    2.5,   "",     "Facteur d'idéalité de la diode"),
}

# Valeurs nominales de référence pour la normalisation d'affichage
VALEURS_NOMINALES: dict[str, float] = {
    "irradiance":           1000.0,
    "temperature_panneau":  25.0,
    "temperature_ambiante": 20.0,
    "tension_voc":          41.5,
    "courant_isc":          10.2,
    "puissance_mpp":        360.0,
    "tension_mpp":          35.8,
    "courant_mpp":          10.1,
    "resistance_serie":     0.3,
    "resistance_shunt":     7500.0,
    "fill_factor":          76.0,
    "efficacite":           19.5,
    "temps_fonctionnement": 8000.0,
    "humidite":             35.0,
    "vitesse_vent":         5.0,
    "facteur_idealite":     1.2,
}

# Surface estimée du panneau de référence (m²) pour le calcul du performance_ratio
SURFACE_PANNEAU_M2 = 1.87   # panneau 360 W nominaux

# Efficacité nominale de référence (%)
EFFICACITE_NOMINALE = 19.5


# ═══════════════════════════════════════════════════════════════════════
# 1. engineer_features — transformation DataFrame (utilisée à l'entraînement)
# ═══════════════════════════════════════════════════════════════════════
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute 6 features physiques dérivées à un DataFrame de mesures brutes.

    Features ajoutées :
      - performance_ratio    : rendement réel vs irradiance reçue
      - delta_temperature    : stress thermique panneau / ambiant
      - ratio_impp_isc       : qualité de la courbe IV côté courant
      - ratio_vmpp_voc       : qualité de la courbe IV côté tension
      - rs_normalise         : résistance série adimensionnelle
      - score_vieillissement : indice composite durée + dégradation
    """
    df = df.copy()

    # Performance ratio : puissance réelle / (irradiance × surface × η_STC)
    irr_safe = df["irradiance"].replace(0, np.nan)
    df["performance_ratio"] = df["puissance_mpp"] / (irr_safe * SURFACE_PANNEAU_M2)
    df["performance_ratio"] = df["performance_ratio"].fillna(0).clip(0, 1)

    # Différentiel thermique panneau - ambiant
    df["delta_temperature"] = df["temperature_panneau"] - df["temperature_ambiante"]

    # Rapport Impp / Isc (indicateur qualité IV côté courant)
    isc_safe = df["courant_isc"].replace(0, np.nan)
    df["ratio_impp_isc"] = df["courant_mpp"] / isc_safe
    df["ratio_impp_isc"] = df["ratio_impp_isc"].fillna(0).clip(0, 1)

    # Rapport Vmpp / Voc (indicateur qualité IV côté tension)
    voc_safe = df["tension_voc"].replace(0, np.nan)
    df["ratio_vmpp_voc"] = df["tension_mpp"] / voc_safe
    df["ratio_vmpp_voc"] = df["ratio_vmpp_voc"].fillna(0).clip(0, 1)

    # Résistance série normalisée sur 5 Ω max
    df["rs_normalise"] = df["resistance_serie"] / 5.0

    # Score de vieillissement : combinaison durée de vie consommée + perte d'efficacité
    df["score_vieillissement"] = (
        (df["temps_fonctionnement"] / 50_000) *
        (1 - df["efficacite"] / EFFICACITE_NOMINALE)
    )

    return df


# ═══════════════════════════════════════════════════════════════════════
# 2. validate_input — vérification d'un dict de mesures brutes
# ═══════════════════════════════════════════════════════════════════════
def validate_input(data: dict) -> dict[str, list[str]]:
    """
    Vérifie qu'un dictionnaire de mesures contient toutes les features
    requises et que leurs valeurs sont dans les plages physiques valides.

    Paramètres :
        data : dict {nom_feature: valeur_numérique}

    Retourne :
        dict avec deux clés :
          "erreurs"     : liste de messages d'erreur bloquants
          "avertissements" : liste d'avertissements non bloquants
    """
    erreurs:         list[str] = []
    avertissements:  list[str] = []

    for nom, (vmin, vmax, unite, desc) in FEATURE_SPECS.items():
        # Vérification de présence
        if nom not in data:
            erreurs.append(f"Feature manquante : '{nom}' ({desc})")
            continue

        # Conversion sécurisée
        try:
            valeur = float(data[nom])
        except (TypeError, ValueError):
            erreurs.append(f"'{nom}' : valeur non numérique ({data[nom]!r})")
            continue

        # Vérification de plage
        if valeur < vmin or valeur > vmax:
            erreurs.append(
                f"'{nom}' = {valeur} hors plage [{vmin}, {vmax}] {unite}"
            )

        # Avertissements physiques spécifiques
        if nom == "tension_mpp" and "tension_voc" in data:
            try:
                if float(data["tension_mpp"]) >= float(data["tension_voc"]):
                    avertissements.append(
                        "Vmpp ≥ Voc : incohérence physique — Vmpp doit être < Voc"
                    )
            except (TypeError, ValueError):
                pass

        if nom == "courant_mpp" and "courant_isc" in data:
            try:
                if float(data["courant_mpp"]) > float(data["courant_isc"]):
                    avertissements.append(
                        "Impp > Isc : incohérence physique — Impp doit être ≤ Isc"
                    )
            except (TypeError, ValueError):
                pass

        if nom == "fill_factor":
            try:
                ff = float(data[nom])
                if ff < 30:
                    avertissements.append(
                        f"Fill Factor très faible ({ff:.1f} %) — "
                        "possible court-circuit ou vieillissement sévère"
                    )
            except (TypeError, ValueError):
                pass

    return {"erreurs": erreurs, "avertissements": avertissements}


# ═══════════════════════════════════════════════════════════════════════
# 3. compute_derived_features — calcul des features dérivées depuis un dict
# ═══════════════════════════════════════════════════════════════════════
def compute_derived_features(data: dict) -> dict:
    """
    Calcule les 6 features physiques dérivées depuis un dict de mesures brutes.

    Paramètres :
        data : dict {nom_feature: valeur_numérique} (16 features brutes)

    Retourne :
        dict contenant les 6 features dérivées :
          - performance_ratio    (0–1)
          - delta_temperature    (°C)
          - ratio_impp_isc       (0–1)
          - ratio_vmpp_voc       (0–1)
          - rs_normalise         (0–2, sans unité)
          - score_vieillissement (≥ 0)
    """
    def _get(cle: str, defaut: float = 0.0) -> float:
        try:
            return float(data.get(cle, defaut))
        except (TypeError, ValueError):
            return defaut

    irr      = _get("irradiance")
    puiss    = _get("puissance_mpp")
    t_pan    = _get("temperature_panneau")
    t_amb    = _get("temperature_ambiante")
    i_mpp    = _get("courant_mpp")
    i_sc     = _get("courant_isc")
    v_mpp    = _get("tension_mpp")
    v_oc     = _get("tension_voc")
    r_serie  = _get("resistance_serie")
    duree    = _get("temps_fonctionnement")
    eff      = _get("efficacite")

    # Performance ratio : puissance réelle / (irradiance × surface)
    denom_pr = irr * SURFACE_PANNEAU_M2
    perf_ratio = min(1.0, max(0.0, puiss / denom_pr)) if denom_pr > 0 else 0.0

    # Différentiel thermique
    delta_temp = t_pan - t_amb

    # Rapport courant MPP / ISC
    ratio_i = min(1.0, max(0.0, i_mpp / i_sc)) if i_sc > 0 else 0.0

    # Rapport tension MPP / VOC
    ratio_v = min(1.0, max(0.0, v_mpp / v_oc)) if v_oc > 0 else 0.0

    # Résistance série normalisée
    rs_norm = r_serie / 5.0

    # Score de vieillissement
    score_vieill = (duree / 50_000) * (1 - eff / EFFICACITE_NOMINALE)

    return {
        "performance_ratio":    round(perf_ratio,    4),
        "delta_temperature":    round(delta_temp,     2),
        "ratio_impp_isc":       round(ratio_i,        4),
        "ratio_vmpp_voc":       round(ratio_v,        4),
        "rs_normalise":         round(rs_norm,        4),
        "score_vieillissement": round(score_vieill,   4),
    }


# ═══════════════════════════════════════════════════════════════════════
# 4. normalize_for_display — normalisation 0–100 pour l'interface UI
# ═══════════════════════════════════════════════════════════════════════
def normalize_for_display(valeur: float, nom_feature: str) -> float:
    """
    Normalise une valeur mesurée sur l'échelle 0–100 pour l'affichage UI.

    La normalisation est linéaire entre vmin et vmax de la feature.
    Une valeur hors plage est clampée à [0, 100].

    Paramètres :
        valeur      : valeur mesurée (float)
        nom_feature : clé dans FEATURE_SPECS

    Retourne :
        Pourcentage 0–100 représentant la position dans la plage valide.
    """
    if nom_feature not in FEATURE_SPECS:
        return 50.0     # valeur neutre pour une feature inconnue

    vmin, vmax, _, _ = FEATURE_SPECS[nom_feature]
    plage = vmax - vmin

    if plage <= 0:
        return 50.0

    normalise = (valeur - vmin) / plage * 100.0
    return round(max(0.0, min(100.0, normalise)), 1)
