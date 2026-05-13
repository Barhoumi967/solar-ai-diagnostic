"""
Couche d'accès SQLite pour l'historique des diagnostics.

Table : diagnostics
  id, timestamp, classe, libelle, confiance, anomalie_score,
  anomalie_niveau, rul_annees, rul_statut, mesures_json
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "diagnostics.db"


def _connexion() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # accès par nom de colonne
    return conn


def initialiser_db() -> None:
    """Crée la table si elle n'existe pas encore."""
    with _connexion() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS diagnostics (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT    NOT NULL,
                classe          INTEGER NOT NULL,
                libelle         TEXT    NOT NULL,
                confiance       REAL    NOT NULL,
                anomalie_score  REAL,
                anomalie_niveau TEXT,
                rul_annees      REAL,
                rul_statut      TEXT,
                mesures_json    TEXT
            )
        """)
        conn.commit()


def sauvegarder_diagnostic(
    classe: int,
    libelle: str,
    confiance: float,
    anomalie_score: float,
    anomalie_niveau: str,
    rul_annees: float | None,
    rul_statut: str | None,
    mesures: dict,
) -> int:
    """Insère un diagnostic et retourne son id."""
    with _connexion() as conn:
        cur = conn.execute("""
            INSERT INTO diagnostics
              (timestamp, classe, libelle, confiance,
               anomalie_score, anomalie_niveau,
               rul_annees, rul_statut, mesures_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            classe, libelle, confiance,
            anomalie_score, anomalie_niveau,
            rul_annees, rul_statut,
            json.dumps(mesures, ensure_ascii=False),
        ))
        conn.commit()
        return cur.lastrowid


def lire_historique(limite: int = 50) -> list[dict]:
    """Retourne les `limite` derniers diagnostics, du plus récent au plus ancien."""
    import json as _json
    with _connexion() as conn:
        rows = conn.execute("""
            SELECT * FROM diagnostics
            ORDER BY id DESC
            LIMIT ?
        """, (limite,)).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        try:
            d['mesures'] = _json.loads(d['mesures_json']) if d.get('mesures_json') else {}
        except (ValueError, TypeError):
            d['mesures'] = {}
        result.append(d)
    return result


def supprimer_historique() -> None:
    """Efface tous les diagnostics."""
    with _connexion() as conn:
        conn.execute("DELETE FROM diagnostics")
        conn.commit()
