"""
Solar AI Diagnostic — API Flask complète.

Endpoints :
    GET  /api/health             — statut des modèles
    POST /api/diagnose           — diagnostic complet (classification + anomalie + RUL)
    POST /api/batch-diagnose     — diagnostic de plusieurs mesures
    GET  /api/model-metrics      — métriques du modèle (accuracy, F1, rapport)
    GET  /api/feature-importance — importance des features triée
    GET  /api/demo-data          — 24 points simulés (journée complète)
    GET  /api/history            — 50 derniers diagnostics (mémoire)
    POST /api/generate-report    — génère un rapport PDF
"""

import logging
import os
import sys
import random
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS

# ── Chemins ───────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent
FRONTEND_DIR = ROOT.parent / "frontend"
sys.path.insert(0, str(ROOT))

from models.fault_classifier import FaultClassifier, FAULT_LABELS
from models.anomaly_detector import AnomalyDetector
from models.predictor        import LifetimePredictor
from utils.database          import initialiser_db, sauvegarder_diagnostic, lire_historique, supprimer_historique
from utils.report_generator  import generate_report

# ── Logging avec timestamps ───────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Application Flask ─────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # CORS activé pour toutes les origines

# ── Initialisation SQLite ─────────────────────────────────────────────
initialiser_db()

# ── Historique en mémoire (50 derniers diagnostics) ───────────────────
historique_memoire: deque = deque(maxlen=50)

# ── Features brutes attendues dans chaque requête ─────────────────────
RAW_FEATURES = [
    "irradiance", "temperature_panneau", "temperature_ambiante",
    "tension_voc", "courant_isc", "puissance_mpp", "tension_mpp",
    "courant_mpp", "resistance_serie", "resistance_shunt", "fill_factor",
    "efficacite", "temps_fonctionnement", "humidite", "vitesse_vent",
    "facteur_idealite",
]

# ── Recommandations par classe ────────────────────────────────────────
RECOMMENDATIONS = {
    0: ["Système fonctionnel — Maintenance préventive standard"],
    1: [
        "Inspecter les panneaux pour objets obstruants",
        "Nettoyer la végétation environnante",
        "Vérifier l'orientation des panneaux",
    ],
    2: [
        "Arrêt immédiat recommandé",
        "Vérifier l'isolation électrique",
        "Contacter un technicien certifié",
    ],
    3: [
        "Vérifier les connexions de câblage",
        "Tester la continuité électrique",
        "Inspecter les boîtes de jonction",
    ],
    4: [
        "Tester la tension de nuit",
        "Vérifier la mise à la terre",
        "Envisager le remplacement des modules affectés",
    ],
    5: [
        "Nettoyage immédiat des panneaux",
        "Programmer un nettoyage régulier",
        "Vérifier le système anti-poussière",
    ],
    6: [
        "Inspecter tous les connecteurs MC4",
        "Vérifier la résistance de contact",
        "Resserrer les connexions",
    ],
    7: [
        "Évaluation complète de l'installation",
        "Planifier le remplacement des modules",
        "Audit de performance recommandé",
    ],
}


# ── Chargement des modèles au démarrage ───────────────────────────────
def _charger_modeles():
    """Charge les 3 modèles depuis les fichiers .pkl sauvegardés."""
    try:
        clf  = FaultClassifier.charger()
        det  = AnomalyDetector.charger()
        pred = LifetimePredictor.charger()
        logger.info("FaultClassifier   chargé ✓")
        logger.info("AnomalyDetector   chargé ✓")
        logger.info("LifetimePredictor chargé ✓")
        return clf, det, pred
    except FileNotFoundError as e:
        logger.warning(f"Modèles introuvables : {e}")
        logger.warning("Lancer backend/models/train_all.py pour entraîner les modèles.")
        return None, None, None


classifier, anomaly_detector, lifetime_predictor = _charger_modeles()


# ── Utilitaire : validation des features ─────────────────────────────
def _valider_features(data: dict) -> list[str]:
    """Retourne la liste des features manquantes dans la requête."""
    return [f for f in RAW_FEATURES if f not in data]


def _construire_diagnostic(mesures: dict) -> dict:
    """
    Exécute le pipeline complet sur un jeu de mesures :
      1. FaultClassifier  → classe + confiance + probabilités
      2. AnomalyDetector  → score 0–100 + niveau d'alerte
      3. LifetimePredictor → RUL en heures et en années
    """
    res_clf = classifier.predict(mesures)
    res_det = anomaly_detector.get_anomaly_score(mesures)
    res_rul = lifetime_predictor.predict_rul(
        temps_actuel        = mesures["temps_fonctionnement"],
        efficacite_actuelle = mesures["efficacite"],
    )

    classe = res_clf["classe"]
    return {
        "panne_detectee": res_clf["label_panne"],
        "classe":         classe,
        "confiance":      res_clf["confiance"],
        "score_anomalie": res_det["score"],
        "niveau_alerte":  res_det["niveau"],
        "probabilites":   res_clf["probabilites"],
        "rul_heures":     res_rul["rul_heures"],
        "rul_annees":     res_rul["rul_annees"],
        "rul_statut":     res_rul["statut"],
        "recommendations": RECOMMENDATIONS.get(classe, []),
        "timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ════════════════════════════════════════════════════════════════════
#  ENDPOINTS
# ════════════════════════════════════════════════════════════════════

# ── 1. Health ─────────────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    """Statut de l'API et des modèles chargés."""
    return jsonify({
        "status":        "ok",
        "models_loaded": classifier is not None,
        "modele_actif":  classifier.best_name if classifier else None,
        "modeles": {
            "fault_classifier":   classifier is not None,
            "anomaly_detector":   anomaly_detector is not None,
            "lifetime_predictor": lifetime_predictor is not None,
        },
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })


# ── 2. Diagnose ───────────────────────────────────────────────────────
@app.route("/api/diagnose", methods=["POST"])
def diagnose():
    """
    Diagnostic complet d'un panneau PV.
    Body : les 16 features brutes.
    Retourne : classification + anomalie + RUL + recommandations.
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Corps JSON manquant"}), 400

    manquantes = _valider_features(data)
    if manquantes:
        return jsonify({"error": f"Features manquantes : {manquantes}"}), 400

    if classifier is None:
        return jsonify({"error": "Modèles non chargés — lancer train_all.py"}), 503

    mesures = {k: float(data[k]) for k in RAW_FEATURES}

    resultat = _construire_diagnostic(mesures)
    logger.info(f"Diagnostic → {resultat['panne_detectee']} ({resultat['confiance']} %)")

    # Persistance SQLite
    sauvegarder_diagnostic(
        classe          = resultat["classe"],
        libelle         = resultat["panne_detectee"],
        confiance       = resultat["confiance"],
        anomalie_score  = resultat["score_anomalie"],
        anomalie_niveau = resultat["niveau_alerte"],
        rul_annees      = resultat["rul_annees"],
        rul_statut      = resultat["rul_statut"],
        mesures         = mesures,
    )

    # Historique mémoire
    historique_memoire.appendleft({**resultat, "mesures": mesures})

    return jsonify(resultat)


# ── 3. Batch-diagnose ────────────────────────────────────────────────
@app.route("/api/batch-diagnose", methods=["POST"])
def batch_diagnose():
    """
    Diagnostique une liste de mesures en une seule requête.
    Body : {"mesures": [...liste de dicts...]}
    Retourne : liste de résultats + statistiques globales.
    """
    data = request.get_json(silent=True)
    if not data or "mesures" not in data:
        return jsonify({"error": "Clé 'mesures' manquante dans le body"}), 400

    if classifier is None:
        return jsonify({"error": "Modèles non chargés"}), 503

    liste_mesures = data["mesures"]
    if not isinstance(liste_mesures, list) or len(liste_mesures) == 0:
        return jsonify({"error": "'mesures' doit être une liste non vide"}), 400

    resultats = []
    erreurs   = []

    for i, m in enumerate(liste_mesures):
        manquantes = _valider_features(m)
        if manquantes:
            erreurs.append({"index": i, "erreur": f"Features manquantes : {manquantes}"})
            continue
        mesures  = {k: float(m[k]) for k in RAW_FEATURES}
        resultat = _construire_diagnostic(mesures)
        resultats.append({**resultat, "index": i})

    # Statistiques globales sur le batch
    classes_detectees = [r["classe"] for r in resultats]
    defauts = [r for r in resultats if r["classe"] != 0]
    stats = {
        "total":           len(liste_mesures),
        "traites":         len(resultats),
        "erreurs":         len(erreurs),
        "panneaux_sains":  classes_detectees.count(0),
        "panneaux_defectueux": len(defauts),
        "taux_defauts":    round(len(defauts) / max(len(resultats), 1) * 100, 1),
        "classe_dominante": max(set(classes_detectees), key=classes_detectees.count)
                            if classes_detectees else None,
    }

    logger.info(f"Batch : {stats['traites']} mesures, {stats['panneaux_defectueux']} défauts")
    return jsonify({"resultats": resultats, "statistiques": stats, "erreurs": erreurs})


# ── 4. Model-metrics ─────────────────────────────────────────────────
@app.route("/api/model-metrics", methods=["GET"])
def model_metrics():
    """
    Calcule et retourne les métriques du modèle sur le dataset complet.
    Inclut : accuracy, F1-score macro, rapport de classification par classe.
    """
    if classifier is None:
        return jsonify({"error": "Modèle non chargé"}), 503

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    from utils.feature_engineering import engineer_features

    csv_path = ROOT / "data" / "solar_faults.csv"
    if not csv_path.exists():
        return jsonify({"error": "Dataset introuvable"}), 404

    df = pd.read_csv(csv_path)
    df = engineer_features(df)

    feature_cols = classifier.feature_names
    X = df[feature_cols]
    y = df["classe"]

    # Même split que l'entraînement pour comparabilité
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2,
                                             random_state=42, stratify=y)
    X_test_sc = pd.DataFrame(
        classifier.scaler.transform(X_test),
        columns=feature_cols,
    )
    y_pred = classifier.best_model.predict(X_test_sc)

    rapport_dict = {}
    rapport_str  = classification_report(y_test, y_pred,
                                          target_names=list(FAULT_LABELS.values()),
                                          output_dict=True)
    for label, metrics in rapport_str.items():
        if isinstance(metrics, dict):
            rapport_dict[label] = {
                "precision": round(metrics["precision"] * 100, 2),
                "recall":    round(metrics["recall"]    * 100, 2),
                "f1_score":  round(metrics["f1-score"]  * 100, 2),
                "support":   int(metrics["support"]),
            }

    return jsonify({
        "modele":       classifier.best_name,
        "accuracy":     round(accuracy_score(y_test, y_pred) * 100, 2),
        "f1_macro":     round(f1_score(y_test, y_pred, average="macro") * 100, 2),
        "f1_weighted":  round(f1_score(y_test, y_pred, average="weighted") * 100, 2),
        "n_test":       len(y_test),
        "rapport":      rapport_dict,
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })


# ── 5. Feature-importance ────────────────────────────────────────────
@app.route("/api/feature-importance", methods=["GET"])
def feature_importance():
    """
    Retourne les features triées par importance décroissante.
    Prêt à brancher sur un graphique barres côté frontend.
    """
    if classifier is None:
        return jsonify({"error": "Modèle non chargé"}), 503

    importances = classifier.best_model.feature_importances_
    features    = classifier.feature_names

    data = sorted(
        [{"feature": f, "importance": round(float(i) * 100, 3)}
         for f, i in zip(features, importances)],
        key=lambda x: -x["importance"],
    )

    return jsonify({
        "modele":   classifier.best_name,
        "features": data,
        "top_3":    [d["feature"] for d in data[:3]],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })


# ── 6. Demo-data ─────────────────────────────────────────────────────
@app.route("/api/demo-data", methods=["GET"])
def demo_data():
    """
    Génère 24 points de données simulées représentant une journée de production.
    Simule la courbe d'irradiance solaire + injections de défauts aléatoires.
    """
    random.seed(42)
    np.random.seed(42)

    maintenant = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    points = []

    for heure in range(24):
        ts = maintenant + timedelta(hours=heure)

        # Courbe d'irradiance solaire réaliste (cloche gaussienne centrée à 13h)
        irr_base = max(0, 1050 * np.exp(-0.5 * ((heure - 13) / 4) ** 2))
        irr = irr_base + np.random.normal(0, 20) if irr_base > 0 else 0

        # Température corrélée à l'irradiance
        t_amb = 18 + 0.015 * irr + np.random.normal(0, 1.5)
        t_pan = t_amb + 0.03 * irr + np.random.normal(0, 2)

        # Puissance corrélée à l'irradiance
        puissance = max(0, irr * 0.37 + np.random.normal(0, 8))

        # Efficacité légèrement dégradée en milieu de journée (chaleur)
        eff = max(0, 18.5 - 0.04 * (t_pan - 25) + np.random.normal(0, 0.3))

        # Injection d'une alerte aléatoire à ~14h et ~17h pour le démo
        alerte = None
        if heure == 14:
            alerte = {"type": "Encrassement", "classe": 5, "score": 61}
        elif heure == 17:
            alerte = {"type": "Ombrage partiel", "classe": 1, "score": 48}

        points.append({
            "timestamp":   ts.strftime("%Y-%m-%dT%H:%M:%S"),
            "heure":       heure,
            "irradiance":  round(float(irr),      1),
            "puissance":   round(float(puissance), 1),
            "temperature_panneau":  round(float(t_pan), 1),
            "temperature_ambiante": round(float(t_amb), 1),
            "efficacite":  round(float(eff),       2),
            "alerte":      alerte,
        })

    # Statistiques journalières
    pts_jour = [p for p in points if p["irradiance"] > 50]
    energie_kwh = sum(p["puissance"] for p in points) / 1000  # kWh approximatif

    return jsonify({
        "date":          maintenant.strftime("%Y-%m-%d"),
        "points":        points,
        "statistiques": {
            "irradiance_max":   max((p["irradiance"] for p in points), default=0),
            "puissance_max":    max((p["puissance"]  for p in points), default=0),
            "energie_journee":  round(energie_kwh, 2),
            "heures_production": len(pts_jour),
            "alertes":          [p for p in points if p["alerte"]],
        },
    })


# ── 7. History ────────────────────────────────────────────────────────
@app.route("/api/history", methods=["GET"])
def history_get():
    """
    Retourne les diagnostics avec pagination optionnelle.
    Paramètres :
      source   — "memory" (défaut) ou "db"
      limite   — nb max à charger depuis la DB (défaut 200)
      page     — numéro de page (1-indexé, défaut 1)
      par_page — lignes par page (défaut 0 = tout retourner)
    """
    source   = request.args.get("source",   "memory")
    limite   = int(request.args.get("limite",   200))
    page     = max(1, int(request.args.get("page",     1)))
    par_page = int(request.args.get("par_page", 0))

    if source == "db":
        tous = lire_historique(limite)
    else:
        tous = list(historique_memoire)

    total = len(tous)

    if par_page > 0:
        nb_pages = max(1, -(-total // par_page))   # division plafond
        page     = min(page, nb_pages)
        debut    = (page - 1) * par_page
        rows     = tous[debut: debut + par_page]
    else:
        nb_pages = 1
        rows     = tous

    return jsonify({
        "diagnostics": rows,
        "total":       total,
        "page":        page,
        "par_page":    par_page,
        "nb_pages":    nb_pages,
        "source":      "sqlite" if source == "db" else "memory",
    })


@app.route("/api/history", methods=["DELETE"])
def history_delete():
    """Efface l'historique mémoire et SQLite."""
    historique_memoire.clear()
    supprimer_historique()
    return jsonify({"status": "ok", "message": "Historique effacé"})


# ── 8. Generate-report ────────────────────────────────────────────────
@app.route("/api/generate-report", methods=["POST"])
def generate_report_endpoint():
    """
    Génère un rapport PDF à partir d'un résultat de diagnostic.
    Body : résultat complet de /api/diagnose + champ 'mesures'.
    Retourne : chemin du fichier PDF généré.
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Corps JSON manquant"}), 400

    mesures = data.get("mesures", {})
    if not mesures:
        return jsonify({"error": "Champ 'mesures' manquant"}), 400

    try:
        chemin_pdf = generate_report(diagnostic=data, mesures=mesures)
        logger.info(f"Rapport PDF généré : {chemin_pdf}")
        nom_fichier = Path(chemin_pdf).name
        return jsonify({
            "status":       "ok",
            "pdf_path":     chemin_pdf,
            "pdf_filename": nom_fichier,
            "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
    except Exception as e:
        logger.error(f"Erreur génération PDF : {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/download-report/<nom_fichier>", methods=["GET"])
def download_report(nom_fichier):
    """Sert le fichier PDF généré en téléchargement."""
    from utils.report_generator import REPORTS_DIR
    chemin = REPORTS_DIR / nom_fichier
    if not chemin.exists() or not chemin.suffix == '.pdf':
        return jsonify({"error": "Fichier introuvable"}), 404
    return send_file(str(chemin), mimetype='application/pdf',
                     as_attachment=True, download_name=nom_fichier)


# ── Frontend statique ─────────────────────────────────────────────────
@app.route("/")
def serve_index():
    return send_from_directory(str(FRONTEND_DIR), "index.html")


@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(str(FRONTEND_DIR), path)


# ── Point d'entrée ────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    logger.info(f"Démarrage de l'API Solar AI Diagnostic sur le port {port}")
    logger.info(f"Ouvrir : http://localhost:{port}")
    app.run(debug=False, host="0.0.0.0", port=port)
