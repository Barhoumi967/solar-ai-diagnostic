/* ============================================================
   Solar AI Diagnostic — Logique page Diagnostic Intelligent
   Sliders ↔ inputs synchronisés, présets, appel API,
   jauge SVG animée, probabilités Chart.js, RUL, recommandations
   ============================================================ */

/* ── Noms des features dans l'ordre attendu par le backend ── */
const FEATURES = [
  'irradiance', 'temperature_panneau', 'temperature_ambiante',
  'tension_voc', 'courant_isc', 'puissance_mpp', 'tension_mpp',
  'courant_mpp', 'resistance_serie', 'resistance_shunt',
  'fill_factor', 'efficacite', 'temps_fonctionnement',
  'humidite', 'vitesse_vent', 'facteur_idealite',
];

/* ── Valeurs des 5 scénarios rapides ──────────────────────── */
const SCENARIOS = {
  'Normal': {
    irradiance:950, temperature_panneau:45, temperature_ambiante:28,
    tension_voc:41.5, courant_isc:10.2, puissance_mpp:360,
    tension_mpp:35.8, courant_mpp:10.1, resistance_serie:0.3,
    resistance_shunt:7500, fill_factor:76, efficacite:19.5,
    temps_fonctionnement:8000, humidite:35, vitesse_vent:5, facteur_idealite:1.2,
  },
  'Ombrage': {
    irradiance:600, temperature_panneau:38, temperature_ambiante:25,
    tension_voc:36.0, courant_isc:5.5, puissance_mpp:148,
    tension_mpp:28.5, courant_mpp:5.2, resistance_serie:0.9,
    resistance_shunt:900, fill_factor:48, efficacite:11.0,
    temps_fonctionnement:15000, humidite:55, vitesse_vent:7, facteur_idealite:1.5,
  },
  'Court-circuit': {
    irradiance:800, temperature_panneau:55, temperature_ambiante:30,
    tension_voc:12.0, courant_isc:1.5, puissance_mpp:18,
    tension_mpp:8.0, courant_mpp:2.2, resistance_serie:0.4,
    resistance_shunt:95, fill_factor:44, efficacite:2.8,
    temps_fonctionnement:15000, humidite:70, vitesse_vent:8, facteur_idealite:1.85,
  },
  'PID': {
    irradiance:850, temperature_panneau:72, temperature_ambiante:40,
    tension_voc:26.0, courant_isc:7.2, puissance_mpp:140,
    tension_mpp:21.0, courant_mpp:6.7, resistance_serie:1.2,
    resistance_shunt:350, fill_factor:55, efficacite:9.5,
    temps_fonctionnement:18000, humidite:88, vitesse_vent:4, facteur_idealite:1.9,
  },
  'Encrassement': {
    irradiance:420, temperature_panneau:35, temperature_ambiante:22,
    tension_voc:40.0, courant_isc:4.8, puissance_mpp:140,
    tension_mpp:33.0, courant_mpp:4.2, resistance_serie:0.5,
    resistance_shunt:5000, fill_factor:72, efficacite:12.5,
    temps_fonctionnement:5000, humidite:60, vitesse_vent:3, facteur_idealite:1.25,
  },
};

/* ── Urgence des recommandations selon la classe ─────────── */
const URGENCE_CLASSE = { 0:'ok', 1:'warn', 2:'critical', 3:'critical',
                          4:'warn', 5:'ok', 6:'warn', 7:'warn' };

/* ── Sévérité du badge selon la classe ───────────────────── */
function severite(classe) {
  if (classe === 0) return 'normal';
  if (classe === 2 || classe === 3) return 'danger';
  return 'warning';
}

/* ── Derniers résultat + mesures pour le rapport PDF ─────── */
let dernierResultat  = null;
let dernieresMesures = null;
let instanceProba    = null;

/* ══════════════════════════════════════════════════════════
   syncSliders — Lie chaque slider à son input numérique
   et vice-versa.
   ══════════════════════════════════════════════════════════ */
function syncSliders() {
  FEATURES.forEach(nom => {
    const input  = document.querySelector(`[name="${nom}"]`);
    const slider = document.querySelector(`[name="${nom}_sl"]`);
    if (!input || !slider) return;

    /* Slider → input */
    slider.addEventListener('input', () => {
      input.value = slider.value;
      coloriserSlider(slider);
    });

    /* Input → slider */
    input.addEventListener('input', () => {
      const v = parseFloat(input.value);
      if (!isNaN(v)) {
        slider.value = v;
        coloriserSlider(slider);
      }
    });

    /* Couleur initiale */
    coloriserSlider(slider);
  });
}

/* Colore la piste du slider selon la position (0 → 100 %) */
function coloriserSlider(slider) {
  const min = parseFloat(slider.min);
  const max = parseFloat(slider.max);
  const val = parseFloat(slider.value);
  const pct = ((val - min) / (max - min)) * 100;
  slider.style.background =
    `linear-gradient(to right, var(--accent-cyan) ${pct}%, var(--bordure) ${pct}%)`;
}

/* ══════════════════════════════════════════════════════════
   initScenarios — Boutons de scénarios rapides
   ══════════════════════════════════════════════════════════ */
function initScenarios() {
  document.querySelectorAll('.btn-scenario').forEach(btn => {
    btn.addEventListener('click', () => {
      const cle    = btn.dataset.preset;
      const valeurs = SCENARIOS[cle];
      if (!valeurs) return;

      /* Remplit tous les inputs et sliders */
      FEATURES.forEach(nom => {
        const input  = document.querySelector(`[name="${nom}"]`);
        const slider = document.querySelector(`[name="${nom}_sl"]`);
        if (valeurs[nom] !== undefined) {
          if (input)  input.value  = valeurs[nom];
          if (slider) { slider.value = valeurs[nom]; coloriserSlider(slider); }
        }
      });

      /* Marque le bouton actif */
      document.querySelectorAll('.btn-scenario').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
    });
  });
}

/* ══════════════════════════════════════════════════════════
   initForm — Gestion de la soumission
   ══════════════════════════════════════════════════════════ */
function initForm() {
  const form = document.getElementById('diag-form');
  if (!form) return;

  form.addEventListener('submit', async e => {
    e.preventDefault();

    /* Collecte les 16 valeurs */
    const mesures = {};
    FEATURES.forEach(nom => {
      const el = form.querySelector(`[name="${nom}"]`);
      if (el) mesures[nom] = parseFloat(el.value);
    });

    afficherChargement();
    btnLancer(true);

    try {
      const res = await SolarAPI.diagnoseSystem(mesures);
      dernierResultat  = res;
      dernieresMesures = mesures;
      afficherResultat(res, mesures);
    } catch (err) {
      afficherErreur(err.message);
    } finally {
      btnLancer(false);
    }
  });
}

/* ── Active/désactive l'état de chargement du bouton ─────── */
function btnLancer(enCours) {
  const btn = document.getElementById('btn-lancer');
  const txt = document.getElementById('btn-lancer-txt');
  if (!btn) return;
  if (enCours) {
    btn.classList.add('loading');
    btn.innerHTML = '<div class="btn-spinner"></div><span>Analyse en cours…</span>';
  } else {
    btn.classList.remove('loading');
    btn.innerHTML = `
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
           stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
        <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
      </svg>
      <span>⚡ LANCER LE DIAGNOSTIC</span>`;
  }
}

/* ── États de la colonne droite ───────────────────────────── */
function afficherChargement() {
  document.getElementById('res-placeholder')?.classList.add('hidden');
  document.getElementById('res-panel')?.classList.add('hidden');
  document.getElementById('res-loading')?.classList.remove('hidden');
}

function afficherErreur(msg) {
  document.getElementById('res-loading')?.classList.add('hidden');
  const panel = document.getElementById('res-panel');
  if (!panel) return;
  panel.classList.remove('hidden');
  panel.innerHTML = `
    <div class="alert-critical" style="animation:fadeInUp .3s ease">
      <strong>Erreur API :</strong> ${msg}
    </div>
    <button class="btn btn-ghost" style="margin-top:10px" onclick="reinitialiser()">Réessayer</button>`;
}

/* ══════════════════════════════════════════════════════════
   afficherResultat — Remplit tous les blocs du panneau droit
   ══════════════════════════════════════════════════════════ */
function afficherResultat(res, mesures) {
  document.getElementById('res-loading')?.classList.add('hidden');
  const panel = document.getElementById('res-panel');
  if (!panel) return;
  panel.classList.remove('hidden');

  /* Score de santé = 100 - score_anomalie (inversé) */
  const scoresSante = Math.max(0, Math.round(100 - res.score_anomalie));

  /* ① Jauge SVG */
  animerJauge(scoresSante);

  /* ② Badge panne */
  const sev = severite(res.classe);
  document.getElementById('res-badge-panne').innerHTML = `
    <div class="fault-badge-xl ${sev}">
      <span style="width:10px;height:10px;border-radius:50%;background:currentColor;
                   animation:pulse-dot 2s ease-in-out infinite;flex-shrink:0"></span>
      ${res.panne_detectee.toUpperCase()}
    </div>
    <div class="conf-sub">Confiance : <strong style="color:var(--accent-cyan)">${res.confiance} %</strong></div>`;

  /* ③ Score anomalie */
  const coulAnom = res.score_anomalie < 30
    ? 'var(--accent-succes)'
    : res.score_anomalie < 60
      ? 'var(--accent-jaune)'
      : res.score_anomalie < 80
        ? 'var(--accent-orange)'
        : 'var(--accent-danger)';

  const niveauCls = {
    Normal:'normal', Attention:'attention', Alerte:'alerte', Critique:'critique',
  }[res.niveau_alerte] || 'normal';

  document.getElementById('anom-niveau-badge').textContent = res.niveau_alerte.toUpperCase();
  document.getElementById('anom-niveau-badge').className   = `badge badge-${niveauCls}`;
  document.getElementById('anom-score-val').textContent    = res.score_anomalie + '/100';

  requestAnimationFrame(() => setTimeout(() => {
    const fill = document.getElementById('anom-fill');
    if (fill) {
      fill.style.width      = res.score_anomalie + '%';
      fill.style.background = coulAnom;
    }
  }, 60));

  /* ④ Probabilités */
  rendreProbas(res.probabilites, res.panne_detectee);

  /* ⑤ RUL */
  rendreRUL(res);

  /* ⑥ Recommandations */
  rendreRecommandations(res.recommendations, res.classe);
}

/* ══════════════════════════════════════════════════════════
   animerJauge — Anime l'arc SVG circulaire de 0 vers score
   Circonférence cercle r=80 : 2π×80 ≈ 502
   ══════════════════════════════════════════════════════════ */
function animerJauge(score) {
  const arc  = document.getElementById('gauge-arc');
  const val  = document.getElementById('gauge-val');
  if (!arc || !val) return;

  /* Couleur dynamique selon le score */
  const couleur = score >= 70
    ? '#10b981'
    : score >= 40
      ? '#f59e0b'
      : '#ef4444';

  arc.style.stroke = couleur;
  arc.style.filter = `drop-shadow(0 0 8px ${couleur}80)`;

  /* Calcule l'offset pour représenter le score sur 100% du cercle */
  const circonference = 502;
  const offset        = circonference - (score / 100) * circonference;

  /* Animation via transition CSS */
  arc.style.transition   = 'stroke-dashoffset 1.2s cubic-bezier(.25,1,.5,1), stroke .4s ease';
  arc.style.strokeDashoffset = String(offset);

  /* Compteur numérique animé dans le centre */
  const duree = 1100;
  const debut  = performance.now();
  function etape(t) {
    const pct  = Math.min((t - debut) / duree, 1);
    const ease = 1 - Math.pow(1 - pct, 3);
    val.textContent = Math.round(ease * score);
    val.style.fill  = couleur;
    if (pct < 1) requestAnimationFrame(etape);
  }
  requestAnimationFrame(etape);
}

/* ── Graphique barres horizontales des probabilités ──────── */
function rendreProbas(probabilites, labelPredit) {
  if (instanceProba) { instanceProba.destroy(); instanceProba = null; }
  const ctx = document.getElementById('proba-chart');
  if (!ctx) return;

  /* Tri décroissant */
  const entrees = Object.entries(probabilites).sort((a, b) => b[1] - a[1]);
  const labels  = entrees.map(([l]) => l);
  const valeurs = entrees.map(([, v]) => +v.toFixed(1));

  /* Couleur spéciale pour la classe prédite, gris pour les autres */
  const couleurs = labels.map((l, i) =>
    l === labelPredit
      ? '#00d4ff'
      : valeurs[i] >= 15
        ? '#f59e0b'
        : '#374151'
  );

  instanceProba = new Chart(ctx.getContext('2d'), {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        data            : valeurs,
        backgroundColor : couleurs,
        borderRadius    : 4,
        borderSkipped   : false,
      }],
    },
    options: {
      indexAxis           : 'y',
      responsive          : true,
      maintainAspectRatio : false,
      animation           : { duration: 700, easing: 'easeOutQuart' },
      plugins: {
        legend  : { display: false },
        tooltip : {
          backgroundColor : '#111827',
          borderColor     : '#1f2937',
          borderWidth     : 1,
          titleColor      : '#e2e8f0',
          bodyColor       : '#94a3b8',
          padding         : 8,
          cornerRadius    : 6,
          callbacks       : { label: c => ` ${c.raw.toFixed(1)} %` },
        },
      },
      scales: {
        x: {
          min  : 0, max: 100,
          grid : { color: 'rgba(31,41,55,.8)', drawBorder: false },
          ticks: { color: '#4b5563', font: { size: 10 }, callback: v => v + '%' },
        },
        y: {
          grid : { display: false },
          ticks: { color: '#94a3b8', font: { size: 10 } },
        },
      },
    },
  });
}

/* ── Panneau RUL ─────────────────────────────────────────── */
function rendreRUL(res) {
  const bloc = document.getElementById('rul-block');
  if (!bloc) return;

  const pct     = res.rul_heures > 0
    ? Math.min(100, Math.round((1 - res.rul_annees / 25) * 100))
    : 0;
  const coulBarre = pct < 50
    ? 'var(--accent-succes)'
    : pct < 80
      ? 'var(--accent-jaune)'
      : 'var(--accent-danger)';

  const coulVal = { Bon:'var(--accent-succes)', Surveiller:'var(--accent-jaune)',
                    Critique:'var(--accent-danger)' }[res.rul_statut] ?? 'var(--accent-jaune)';

  const statutCls = { Bon:'badge-succes', Surveiller:'badge-jaune',
                      Critique:'badge-danger' }[res.rul_statut] ?? 'badge-muted';

  bloc.innerHTML = `
    <div class="rul-title-row">
      <span class="rul-main-label">Durée de vie restante (RUL)</span>
      <span class="badge ${statutCls}">${res.rul_statut}</span>
    </div>
    <div class="rul-value-big" style="color:${coulVal}">
      RUL Estimé : ${res.rul_annees} ans
    </div>
    <div class="rul-heures-sub">${Number(res.rul_heures).toLocaleString('fr-FR')} heures restantes</div>
    <div class="rul-life-label">
      <span>Vie consommée</span>
      <span class="text-mono">${pct} %</span>
    </div>
    <div class="rul-life-bar">
      <div class="rul-life-fill" id="rul-life-fill" style="background:${coulBarre}"></div>
    </div>`;

  /* Animation de la barre */
  requestAnimationFrame(() => setTimeout(() => {
    const f = document.getElementById('rul-life-fill');
    if (f) f.style.width = pct + '%';
  }, 80));
}

/* ── Recommandations avec icônes colorées ─────────────────── */
function rendreRecommandations(recs, classe) {
  const bloc = document.getElementById('recs-block');
  if (!bloc) return;

  const urg = URGENCE_CLASSE[classe] ?? 'warn';
  const icone = { ok: '✓', warn: '⚠', critical: '!' }[urg];

  bloc.innerHTML = `
    <div class="res-section-label">Recommandations</div>
    ${recs.map(r => `
      <div class="rec-item">
        <div class="rec-icon ${urg}">${icone}</div>
        <span>${r}</span>
      </div>`).join('')}`;
}

/* ── Réinitialise la page pour une nouvelle analyse ──────── */
function reinitialiser() {
  document.querySelectorAll('.btn-scenario').forEach(b => b.classList.remove('active'));
  dernierResultat  = null;
  dernieresMesures = null;

  document.getElementById('res-placeholder')?.classList.remove('hidden');
  document.getElementById('res-loading')?.classList.add('hidden');
  document.getElementById('res-panel')?.classList.add('hidden');

  /* Remet la jauge à zéro */
  const arc = document.getElementById('gauge-arc');
  if (arc) {
    arc.style.transition       = 'none';
    arc.style.strokeDashoffset = '502';
    arc.style.stroke           = '#00d4ff';
  }
  const val = document.getElementById('gauge-val');
  if (val) { val.textContent = '—'; val.style.fill = '#e2e8f0'; }
}

/* ── Génère le rapport PDF via l'API ─────────────────────── */
async function genererRapportPDF() {
  if (!dernierResultat || !dernieresMesures) return;
  const btn = document.getElementById('btn-pdf');
  if (btn) { btn.disabled = true; btn.textContent = 'Génération…'; }

  try {
    const res = await SolarAPI.generateReport(dernierResultat, dernieresMesures);
    /* Déclenche le téléchargement via le nouvel endpoint */
    if (res.pdf_filename) {
      SolarAPI.downloadReport(res.pdf_filename);
      afficherToast('Rapport PDF téléchargé.', 'succes');
    } else {
      afficherToast('Rapport généré (fichier : ' + (res.pdf_path ?? '?') + ').', 'succes');
    }
  } catch (err) {
    afficherToast('Erreur PDF : ' + err.message, 'erreur');
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = `
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor"
             stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
          <polyline points="14 2 14 8 20 8"/>
        </svg>
        Générer Rapport PDF`;
    }
  }
}

/* ── Toast de notification ───────────────────────────────── */
function afficherToast(msg, type = 'info') {
  let zone = document.getElementById('toast-zone');
  if (!zone) {
    zone = document.createElement('div');
    zone.id = 'toast-zone';
    zone.style.cssText =
      'position:fixed;top:20px;right:24px;z-index:9999;display:flex;flex-direction:column;gap:8px';
    document.body.appendChild(zone);
  }
  const pal = {
    succes: { bg:'var(--accent-succes-dim)', border:'rgba(16,185,129,.4)', color:'var(--accent-succes)' },
    erreur: { bg:'var(--accent-danger-dim)', border:'rgba(239,68,68,.4)',  color:'var(--accent-danger)' },
    info  : { bg:'var(--accent-cyan-dim)',   border:'rgba(0,212,255,.3)',  color:'var(--accent-cyan)'   },
  }[type] || {};
  const t = document.createElement('div');
  t.style.cssText = `background:${pal.bg};border:1px solid ${pal.border};color:${pal.color};
    font-size:.82rem;font-weight:500;padding:10px 16px;border-radius:8px;
    box-shadow:0 4px 20px rgba(0,0,0,.4);animation:fadeInUp .3s ease;max-width:340px`;
  t.textContent = msg;
  zone.appendChild(t);
  setTimeout(() => t.remove(), 5000);
}

/* ── Horloge ──────────────────────────────────────────────── */
function updateClock() {
  const el = document.getElementById('horloge');
  if (el) el.textContent = new Date().toLocaleTimeString('fr-FR');
}

/* ── Statut API sidebar ───────────────────────────────────── */
async function checkAPIStatus() {
  const dot   = document.getElementById('sidebar-dot');
  const label = document.getElementById('sidebar-label');
  try {
    const d  = await SolarAPI.health();
    const ok = d.models_loaded === true;
    if (dot)   dot.className    = `status-dot ${ok ? 'online' : 'offline'}`;
    if (label) label.textContent = ok ? 'API Connectée' : 'Modèles absents';
    const bm = document.getElementById('badge-modele');
    if (bm && d.modele_actif) bm.textContent = d.modele_actif;
  } catch {
    if (dot)   dot.className    = 'status-dot offline';
    if (label) label.textContent = 'API hors ligne';
  }
}

/* ══════════════════════════════════════════════════════════
   Initialisation
   ══════════════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
  updateClock();
  setInterval(updateClock, 1000);

  checkAPIStatus();
  setInterval(checkAPIStatus, 30_000);

  syncSliders();
  initScenarios();
  initForm();

  /* Coloration initiale de tous les sliders */
  document.querySelectorAll('.param-slider').forEach(coloriserSlider);
});
