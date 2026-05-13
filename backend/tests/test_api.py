"""
Tests automatiques de l'API Solar AI Diagnostic.

Lance tous les endpoints et affiche un rapport coloré :
  ✅ ou ❌ + temps de réponse + extrait du résultat

Utilisation :
    python backend/tests/test_api.py
    python backend/tests/test_api.py --base http://localhost:5001/api
"""

import sys
import json
import time
import urllib.request
import urllib.error
import argparse
from datetime import datetime

# ── Adresse de base de l'API ──────────────────────────────────────────
BASE_URL_DEFAUT = "http://localhost:5001/api"

# ── Codes ANSI pour la couleur console ────────────────────────────────
VERT  = "\033[92m"
ROUGE = "\033[91m"
JAUNE = "\033[93m"
CYAN  = "\033[96m"
GRIS  = "\033[90m"
BOLD  = "\033[1m"
RESET = "\033[0m"

# ── Données de test ────────────────────────────────────────────────────

# Scénario 1 : panneau en état normal
MESURES_NORMAL = {
    "irradiance": 950, "temperature_panneau": 45, "temperature_ambiante": 28,
    "tension_voc": 41.5, "courant_isc": 10.2, "puissance_mpp": 360,
    "tension_mpp": 35.8, "courant_mpp": 10.1, "resistance_serie": 0.3,
    "resistance_shunt": 7500, "fill_factor": 76, "efficacite": 19.5,
    "temps_fonctionnement": 8000, "humidite": 35, "vitesse_vent": 5,
    "facteur_idealite": 1.2,
}

# Scénario 2 : court-circuit (tension VOC effondrée)
MESURES_COURT_CIRCUIT = {
    "irradiance": 800, "temperature_panneau": 55, "temperature_ambiante": 30,
    "tension_voc": 12.0, "courant_isc": 1.5, "puissance_mpp": 18,
    "tension_mpp": 8.0,  "courant_mpp": 2.2, "resistance_serie": 0.4,
    "resistance_shunt": 95, "fill_factor": 44, "efficacite": 2.8,
    "temps_fonctionnement": 15000, "humidite": 70, "vitesse_vent": 8,
    "facteur_idealite": 1.85,
}

# Scénario 3 : ombrage partiel (courant réduit)
MESURES_OMBRAGE = {
    "irradiance": 600, "temperature_panneau": 38, "temperature_ambiante": 25,
    "tension_voc": 36.0, "courant_isc": 5.5, "puissance_mpp": 148,
    "tension_mpp": 28.5, "courant_mpp": 5.2, "resistance_serie": 0.9,
    "resistance_shunt": 900, "fill_factor": 48, "efficacite": 11.0,
    "temps_fonctionnement": 15000, "humidite": 55, "vitesse_vent": 7,
    "facteur_idealite": 1.5,
}

# Scénario 4 : dégradation PID (température élevée + Rsh faible)
MESURES_PID = {
    "irradiance": 850, "temperature_panneau": 72, "temperature_ambiante": 40,
    "tension_voc": 26.0, "courant_isc": 7.2, "puissance_mpp": 140,
    "tension_mpp": 21.0, "courant_mpp": 6.7, "resistance_serie": 1.2,
    "resistance_shunt": 350, "fill_factor": 55, "efficacite": 9.5,
    "temps_fonctionnement": 18000, "humidite": 88, "vitesse_vent": 4,
    "facteur_idealite": 1.9,
}

# Scénario 5 : encrassement (irradiance réduite, paramètres OK)
MESURES_ENCRASSEMENT = {
    "irradiance": 420, "temperature_panneau": 35, "temperature_ambiante": 22,
    "tension_voc": 40.0, "courant_isc": 4.8, "puissance_mpp": 140,
    "tension_mpp": 33.0, "courant_mpp": 4.2, "resistance_serie": 0.5,
    "resistance_shunt": 5000, "fill_factor": 72, "efficacite": 12.5,
    "temps_fonctionnement": 5000, "humidite": 60, "vitesse_vent": 3,
    "facteur_idealite": 1.25,
}


# ═══════════════════════════════════════════════════════════════════════
# Utilitaires HTTP
# ═══════════════════════════════════════════════════════════════════════

def _requete(methode: str, url: str, data: dict | None = None) -> tuple[dict, float]:
    """
    Exécute une requête HTTP et retourne (réponse_json, durée_ms).

    Lève urllib.error.URLError en cas d'échec réseau.
    """
    debut = time.perf_counter()

    if methode == "GET":
        req = urllib.request.Request(url)
    elif methode == "POST":
        corps = json.dumps(data).encode("utf-8")
        req   = urllib.request.Request(url, data=corps,
                headers={"Content-Type": "application/json"}, method="POST")
    elif methode == "DELETE":
        req = urllib.request.Request(url, method="DELETE")
    else:
        raise ValueError(f"Méthode HTTP non supportée : {methode}")

    with urllib.request.urlopen(req, timeout=15) as reponse:
        corps_reponse = json.loads(reponse.read())

    duree_ms = (time.perf_counter() - debut) * 1000
    return corps_reponse, duree_ms


def _extrait(data: dict, cles: list[str], max_chars: int = 80) -> str:
    """Extrait un résumé lisible des clés importantes d'une réponse."""
    parties = []
    for cle in cles:
        val = data.get(cle)
        if val is None:
            continue
        if isinstance(val, float):
            parties.append(f"{cle}={val:.2f}")
        elif isinstance(val, str) and len(val) > 30:
            parties.append(f"{cle}='{val[:30]}…'")
        else:
            parties.append(f"{cle}={val!r}")
    resume = " | ".join(parties)
    return resume[:max_chars] + "…" if len(resume) > max_chars else resume


# ═══════════════════════════════════════════════════════════════════════
# Suite de tests
# ═══════════════════════════════════════════════════════════════════════

def lancer_tests(base: str) -> int:
    """
    Exécute tous les tests et retourne le nombre d'échecs.

    Paramètres :
        base : URL de base de l'API (ex. http://localhost:5001/api)
    """
    resultats: list[dict] = []

    # ── Définition des cas de test ─────────────────────────────────────
    tests = [

        # ── Santé et modèles ────────────────────────────────────────────
        {
            "nom":       "GET /health — santé de l'API",
            "methode":   "GET",
            "url":       f"{base}/health",
            "data":      None,
            "verif":     lambda r: r.get("status") == "ok" and r.get("models_loaded") is True,
            "extrait":   ["status", "models_loaded", "modele_actif"],
        },

        # ── Diagnostic — scénario Normal ────────────────────────────────
        {
            "nom":       "POST /diagnose — scénario Normal",
            "methode":   "POST",
            "url":       f"{base}/diagnose",
            "data":      MESURES_NORMAL,
            "verif":     lambda r: "panne_detectee" in r and r.get("confiance", 0) > 0,
            "extrait":   ["panne_detectee", "confiance", "niveau_alerte"],
        },

        # ── Diagnostic — scénario Court-circuit ─────────────────────────
        {
            "nom":       "POST /diagnose — Court-circuit",
            "methode":   "POST",
            "url":       f"{base}/diagnose",
            "data":      MESURES_COURT_CIRCUIT,
            "verif":     lambda r: "panne_detectee" in r and r.get("confiance", 0) > 50,
            "extrait":   ["panne_detectee", "confiance", "score_anomalie"],
        },

        # ── Diagnostic — scénario Ombrage ───────────────────────────────
        {
            "nom":       "POST /diagnose — Ombrage partiel",
            "methode":   "POST",
            "url":       f"{base}/diagnose",
            "data":      MESURES_OMBRAGE,
            "verif":     lambda r: "panne_detectee" in r,
            "extrait":   ["panne_detectee", "confiance", "rul_annees"],
        },

        # ── Diagnostic — scénario PID ────────────────────────────────────
        {
            "nom":       "POST /diagnose — Dégradation PID",
            "methode":   "POST",
            "url":       f"{base}/diagnose",
            "data":      MESURES_PID,
            "verif":     lambda r: "panne_detectee" in r,
            "extrait":   ["panne_detectee", "confiance", "niveau_alerte"],
        },

        # ── Diagnostic — scénario Encrassement ──────────────────────────
        {
            "nom":       "POST /diagnose — Encrassement",
            "methode":   "POST",
            "url":       f"{base}/diagnose",
            "data":      MESURES_ENCRASSEMENT,
            "verif":     lambda r: "panne_detectee" in r,
            "extrait":   ["panne_detectee", "confiance", "score_anomalie"],
        },

        # ── Diagnostic batch ─────────────────────────────────────────────
        {
            "nom":       "POST /batch-diagnose — 3 mesures simultanées",
            "methode":   "POST",
            "url":       f"{base}/batch-diagnose",
            "data":      {"mesures": [MESURES_NORMAL, MESURES_COURT_CIRCUIT, MESURES_OMBRAGE]},
            "verif":     lambda r: isinstance(r.get("resultats"), list) and len(r["resultats"]) == 3,
            "extrait":   ["total", "reussis"],
        },

        # ── Métriques du modèle ──────────────────────────────────────────
        {
            "nom":       "GET /model-metrics — accuracy et F1",
            "methode":   "GET",
            "url":       f"{base}/model-metrics",
            "data":      None,
            "verif":     lambda r: r.get("accuracy", 0) > 90,
            "extrait":   ["accuracy", "f1_macro", "n_test"],
        },

        # ── Importance des features ──────────────────────────────────────
        {
            "nom":       "GET /feature-importance — top features",
            "methode":   "GET",
            "url":       f"{base}/feature-importance",
            "data":      None,
            "verif":     lambda r: isinstance(r.get("features"), list) and len(r["features"]) > 0,
            "extrait":   ["top_3"],
        },

        # ── Données de démo ──────────────────────────────────────────────
        {
            "nom":       "GET /demo-data — 24 points journaliers",
            "methode":   "GET",
            "url":       f"{base}/demo-data",
            "data":      None,
            "verif":     lambda r: isinstance(r.get("points"), list) and len(r["points"]) == 24,
            "extrait":   ["date"],
        },

        # ── Historique (source db) ───────────────────────────────────────
        {
            "nom":       "GET /history?source=db — lecture SQLite",
            "methode":   "GET",
            "url":       f"{base}/history?source=db",
            "data":      None,
            "verif":     lambda r: "diagnostics" in r and "total" in r,
            "extrait":   ["total", "source", "nb_pages"],
        },

        # ── Historique paginé ────────────────────────────────────────────
        {
            "nom":       "GET /history?source=db&page=1&par_page=3 — pagination",
            "methode":   "GET",
            "url":       f"{base}/history?source=db&page=1&par_page=3",
            "data":      None,
            "verif":     lambda r: r.get("par_page") == 3,
            "extrait":   ["total", "page", "nb_pages", "par_page"],
        },

        # ── Génération de rapport PDF ────────────────────────────────────
        {
            "nom":       "POST /generate-report — PDF 5 pages",
            "methode":   "POST",
            "url":       f"{base}/generate-report",
            "data":      {**_diagnostic_test(), "mesures": MESURES_NORMAL},
            "verif":     lambda r: r.get("status") == "ok" and "pdf_filename" in r,
            "extrait":   ["status", "pdf_filename"],
        },

    ]

    # ── Affichage de l'en-tête ─────────────────────────────────────────
    print(f"\n{BOLD}{CYAN}{'═' * 70}{RESET}")
    print(f"{BOLD}{CYAN}  Solar AI Diagnostic — Suite de tests automatiques{RESET}")
    print(f"{GRIS}  API : {base}")
    print(f"  Début : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}{RESET}")
    print(f"{BOLD}{CYAN}{'═' * 70}{RESET}\n")

    nb_ok = nb_err = 0

    for i, test in enumerate(tests, 1):
        try:
            reponse, duree = _requete(test["methode"], test["url"], test["data"])
            ok = test["verif"](reponse)
        except urllib.error.URLError as e:
            ok     = False
            reponse = {}
            duree   = 0.0
            _erreur_reseau = str(e)
        except Exception as e:
            ok     = False
            reponse = {}
            duree   = 0.0
            _erreur_reseau = str(e)
        else:
            _erreur_reseau = None

        # Icône et couleur
        icone  = f"{VERT}✅{RESET}" if ok else f"{ROUGE}❌{RESET}"
        col_t  = VERT if duree < 200 else (JAUNE if duree < 500 else ROUGE)
        temps  = f"{col_t}{duree:6.1f} ms{RESET}"

        # Résumé du résultat
        if _erreur_reseau:
            info = f"{ROUGE}{_erreur_reseau[:60]}{RESET}"
        elif ok:
            info = f"{GRIS}{_extrait(reponse, test['extrait'])}{RESET}"
        else:
            info = f"{JAUNE}Vérification échouée — {_extrait(reponse, list(reponse.keys())[:3])}{RESET}"

        print(f"  {icone}  [{i:02d}] {test['nom']}")
        print(f"        {temps}  {info}")

        resultats.append({"nom": test["nom"], "ok": ok, "duree_ms": duree})
        if ok:
            nb_ok  += 1
        else:
            nb_err += 1

    # ── Résumé final ────────────────────────────────────────────────────
    total = nb_ok + nb_err
    print(f"\n{BOLD}{CYAN}{'═' * 70}{RESET}")
    print(f"{BOLD}  Résultats : {VERT}{nb_ok}/{total} réussis{RESET}", end="")
    if nb_err:
        print(f"  {ROUGE}{nb_err} échec(s){RESET}")
    else:
        print(f"  {VERT}— Tous les tests passent ✅{RESET}")

    # Temps de réponse moyen
    durees_ok = [r["duree_ms"] for r in resultats if r["ok"] and r["duree_ms"] > 0]
    if durees_ok:
        moy = sum(durees_ok) / len(durees_ok)
        col = VERT if moy < 200 else JAUNE
        print(f"  Temps moyen : {col}{moy:.1f} ms{RESET}")

    print(f"{BOLD}{CYAN}{'═' * 70}{RESET}\n")

    return nb_err


def _diagnostic_test() -> dict:
    """
    Génère un résultat de diagnostic fictif pour le test du générateur PDF.
    Reproduit la structure retournée par /api/diagnose.
    """
    return {
        "panne_detectee":  "Normal",
        "classe":          0,
        "confiance":       99.95,
        "score_anomalie":  5.2,
        "niveau_alerte":   "Normal",
        "probabilites": {
            "Normal":              99.95,
            "Ombrage partiel":      0.02,
            "Court-circuit":        0.01,
            "Circuit ouvert":       0.01,
            "Dégradation PID":      0.00,
            "Encrassement":         0.00,
            "Défaut connexion":     0.00,
            "Vieillissement accéléré": 0.01,
        },
        "rul_heures":      21600.0,
        "rul_annees":      2.47,
        "rul_statut":      "Bon",
        "recommendations": [
            "Aucune intervention corrective requise.",
            "Nettoyage préventif de la surface selon le calendrier standard.",
            "Vérification annuelle des connexions et des boîtes de jonction.",
        ],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ═══════════════════════════════════════════════════════════════════════
# Point d'entrée
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tests API Solar AI Diagnostic")
    parser.add_argument("--base", default=BASE_URL_DEFAUT,
                        help=f"URL de base de l'API (défaut : {BASE_URL_DEFAUT})")
    args = parser.parse_args()

    nb_echecs = lancer_tests(args.base)
    sys.exit(0 if nb_echecs == 0 else 1)
