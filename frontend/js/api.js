/* ============================================================
   Solar AI Diagnostic — Couche d'accès à l'API Flask
   URL relative — fonctionne quel que soit le port
   ============================================================ */

const API_BASE = '/api';

/* ── Utilitaire : requête générique avec gestion d'erreur ─── */
async function _fetch(url, options = {}) {
  try {
    const res = await fetch(url, options);
    if (!res.ok) {
      /* Essaie d'extraire le message d'erreur du JSON */
      let msg = `Erreur HTTP ${res.status}`;
      try { const j = await res.json(); msg = j.error ?? msg; } catch {}
      throw new Error(msg);
    }
    return res.json();
  } catch (err) {
    /* Réseau inaccessible */
    if (err.name === 'TypeError' && err.message.includes('fetch')) {
      throw new Error('API hors ligne — vérifiez que le serveur Flask est bien démarré.');
    }
    throw err;
  }
}

const SolarAPI = {

  /* ── GET /api/health ──────────────────────────────────── */
  /* Vérifie que le serveur et les 3 modèles IA sont chargés */
  health() {
    return _fetch(`${API_BASE}/health`);
  },

  /* ── POST /api/diagnose ───────────────────────────────── */
  /* Lance le diagnostic complet sur un jeu de 16 mesures.
     Retourne : panne_detectee, classe, confiance,
                score_anomalie, niveau_alerte, probabilites,
                rul_heures, rul_annees, rul_statut,
                recommendations, timestamp               */
  diagnoseSystem(mesures) {
    return _fetch(`${API_BASE}/diagnose`, {
      method : 'POST',
      headers: { 'Content-Type': 'application/json' },
      body   : JSON.stringify(mesures),
    });
  },

  /* ── GET /api/history ─────────────────────────────────── */
  /* Lit l'historique depuis SQLite (source=db).
     Retourne : { diagnostics: [...], total, source }      */
  loadHistory(limite = 50) {
    return _fetch(`${API_BASE}/history?limite=${limite}&source=db`);
  },

  /* ── DELETE /api/history ──────────────────────────────── */
  /* Supprime tout l'historique (mémoire + SQLite)         */
  clearHistory() {
    return _fetch(`${API_BASE}/history`, { method: 'DELETE' });
  },

  /* ── GET /api/demo-data ───────────────────────────────── */
  /* Récupère les 24 points de simulation journalière      */
  loadDemoData() {
    return _fetch(`${API_BASE}/demo-data`);
  },

  /* ── POST /api/generate-report ───────────────────────── */
  /* Génère un rapport PDF pour un résultat de diagnostic.
     @param {Object} diagResult  — réponse complète de diagnoseSystem
     @param {Object} mesures     — les 16 features envoyées
     Retourne : { status, pdf_path, pdf_filename, timestamp }         */
  generateReport(diagResult, mesures) {
    const payload = { ...diagResult, mesures };
    return _fetch(`${API_BASE}/generate-report`, {
      method : 'POST',
      headers: { 'Content-Type': 'application/json' },
      body   : JSON.stringify(payload),
    });
  },

  /* ── GET /api/download-report/:nom ───────────────────── */
  /* Déclenche le téléchargement du PDF généré             */
  downloadReport(nomFichier) {
    const url = `${API_BASE}/download-report/${encodeURIComponent(nomFichier)}`;
    const a   = document.createElement('a');
    a.href     = url;
    a.download = nomFichier;
    a.click();
  },

  /* ── GET /api/model-metrics ───────────────────────────── */
  /* Retourne accuracy, F1, rapport de classification      */
  modelMetrics() {
    return _fetch(`${API_BASE}/model-metrics`);
  },

  /* ── GET /api/feature-importance ─────────────────────── */
  /* Features triées par importance décroissante           */
  featureImportance() {
    return _fetch(`${API_BASE}/feature-importance`);
  },

  /* ── POST /api/batch-diagnose ─────────────────────────── */
  /* Diagnostique une liste de mesures en une seule requête */
  batchDiagnose(listeMesures) {
    return _fetch(`${API_BASE}/batch-diagnose`, {
      method : 'POST',
      headers: { 'Content-Type': 'application/json' },
      body   : JSON.stringify({ mesures: listeMesures }),
    });
  },

};
