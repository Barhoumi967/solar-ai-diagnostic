/* ============================================================
   Solar AI Diagnostic — Logique page Métriques IA
   Chargement métriques XGBoost, feature importance, rapport
   ============================================================ */

let instanceImportance = null;
let instanceF1Classes  = null;

/* ── Couleurs par classe (même ordre que LABELS_CL history.js) */
const COULEURS_CLASSES = [
  '#10b981','#f59e0b','#ef4444','#dc2626',
  '#8b5cf6','#00d4ff','#ff6b35','#94a3b8',
];

/* ── Labels courts pour le graphique barres ───────────────── */
const LABELS_COURTS = {
  'Normal'                 : 'Normal',
  'Ombrage partiel'        : 'Ombrage',
  'Court-circuit'          : 'Court-circ.',
  'Circuit ouvert'         : 'Circ. ouvert',
  'Dégradation PID'        : 'PID',
  'Encrassement'           : 'Encrassement',
  'Défaut connexion'       : 'Défaut conn.',
  'Vieillissement accéléré': 'Vieillissement',
};

/* ── Labels lisibles des features ─────────────────────────── */
const LABELS_FEAT = {
  ratio_vmpp_voc        : 'Ratio Vmpp/Voc',
  efficacite            : 'Efficacité',
  fill_factor           : 'Fill Factor',
  resistance_shunt      : 'Résist. shunt',
  tension_voc           : 'Tension VOC',
  resistance_serie      : 'Résist. série',
  tension_mpp           : 'Tension MPP',
  temps_fonctionnement  : 'Temps fonct.',
  courant_isc           : 'Courant ISC',
  puissance_mpp         : 'Puissance MPP',
  courant_mpp           : 'Courant MPP',
  vitesse_vent          : 'Vitesse vent',
  performance_ratio     : 'Perf. ratio',
  irradiance            : 'Irradiance',
  score_vieillissement  : 'Score vieilliss.',
  ratio_impp_isc        : 'Ratio Impp/Isc',
  humidite              : 'Humidité',
  temperature_panneau   : 'Temp. panneau',
  temperature_ambiante  : 'Temp. ambiante',
  facteur_idealite      : 'Fact. idéalité',
  delta_temperature     : 'ΔTempérature',
  rs_normalise          : 'Rs normalisé',
};

/* ══════════════════════════════════════════════════════════
   chargerMetriques — Charge les deux endpoints en parallèle
   ══════════════════════════════════════════════════════════ */
async function chargerMetriques() {
  try {
    const [met, feat] = await Promise.all([
      SolarAPI.modelMetrics(),
      SolarAPI.featureImportance(),
    ]);

    rendreKPI(met, feat);
    rendreImportance(feat.features ?? []);
    rendreRapport(met.rapport ?? {});
    rendreF1Classes(met.rapport ?? {});

  } catch (err) {
    document.getElementById('kpi-zone').innerHTML =
      `<div style="grid-column:1/-1;color:var(--accent-danger);font-size:.85rem;padding:20px">
         Impossible de charger les métriques — ${err.message}
       </div>`;
  }
}

/* ── KPI cards ─────────────────────────────────────────────── */
function rendreKPI(met, feat) {
  /* Accuracy */
  animerKPIVal('m-accuracy', met.accuracy, 1, '%', 1200);
  /* F1 Macro */
  animerKPIVal('m-f1macro',  met.f1_macro,  1, '%', 1200);
  /* N test */
  animerKPIVal('m-ntest', met.n_test, 0, '', 1000);

  /* Badge modèle */
  const bm = document.getElementById('badge-modele');
  if (bm && met.modele) bm.textContent = met.modele;

  /* Top feature */
  const topEl = document.getElementById('m-top-feat');
  const topKey = feat.top_3?.[0] ?? (feat.features?.[0]?.feature ?? '—');
  if (topEl) topEl.textContent = LABELS_FEAT[topKey] ?? topKey;
}

/* Animation countUp légère ────────────────────────────────── */
function animerKPIVal(id, fin, decimales, suffixe, dureeMs) {
  const el = document.getElementById(id);
  if (!el) return;
  const debut = performance.now();
  function etape(t) {
    const pct  = Math.min((t - debut) / dureeMs, 1);
    const ease = 1 - Math.pow(1 - pct, 3);
    el.textContent = (ease * fin).toFixed(decimales) + suffixe;
    if (pct < 1) requestAnimationFrame(etape);
  }
  requestAnimationFrame(etape);
}

/* ── Feature importance — barres horizontales ─────────────── */
function rendreImportance(features) {
  const ctx = document.getElementById('chart-importance');
  if (!ctx) return;
  if (instanceImportance) { instanceImportance.destroy(); instanceImportance = null; }

  /* Top 12 seulement pour la lisibilité */
  const top = features.slice(0, 12);
  const labels  = top.map(f => LABELS_FEAT[f.feature] ?? f.feature);
  const valeurs = top.map(f => +f.importance.toFixed(2));

  /* Gradient cyan → violet selon le rang */
  const couleurs = top.map((_, i) => {
    const t = i / Math.max(top.length - 1, 1);
    return i === 0 ? '#00d4ff' : i < 3 ? '#00b8dd' : `rgba(0,212,255,${0.7 - t * 0.45})`;
  });

  const nbFeat = document.getElementById('feat-count');
  if (nbFeat) nbFeat.textContent = `${features.length} features`;

  instanceImportance = new Chart(ctx.getContext('2d'), {
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
      animation           : { duration: 900, easing: 'easeOutQuart' },
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
          callbacks       : { label: c => ` Importance : ${c.raw} %` },
        },
      },
      scales: {
        x: {
          min  : 0,
          grid : { color: 'rgba(31,41,55,.8)', drawBorder: false },
          ticks: { color: '#4b5563', font: { size: 9 }, callback: v => v + '%' },
        },
        y: {
          grid : { display: false },
          ticks: { color: '#94a3b8', font: { size: 10 } },
        },
      },
    },
  });
}

/* ── Tableau rapport de classification ────────────────────── */
function rendreRapport(rapport) {
  const tbody = document.getElementById('rapport-tbody');
  if (!tbody) return;

  /* Exclut les lignes de moyennes */
  const classes = Object.keys(rapport).filter(k => !k.includes('avg'));

  if (!classes.length) {
    tbody.innerHTML = '<tr><td colspan="5" class="cell-empty">Aucune donnée</td></tr>';
    return;
  }

  tbody.innerHTML = classes.map((nom, i) => {
    const r    = rapport[nom];
    const f1   = r.f1_score ?? 0;
    const prec = r.precision ?? 0;
    const rec  = r.recall ?? 0;
    const sup  = r.support ?? 0;

    const coulF1 = f1 >= 99 ? 'var(--accent-succes)' : f1 >= 95 ? 'var(--accent-jaune)' : 'var(--accent-danger)';
    const dot    = COULEURS_CLASSES[i] ?? '#94a3b8';

    return `
      <tr style="animation-delay:${i * 0.04}s">
        <td>
          <div style="display:flex;align-items:center;gap:8px">
            <span style="width:8px;height:8px;border-radius:50%;background:${dot};flex-shrink:0"></span>
            ${nom}
          </div>
        </td>
        <td class="cell-metric">${prec.toFixed(1)} %</td>
        <td class="cell-metric">${rec.toFixed(1)} %</td>
        <td class="cell-metric" style="color:${coulF1};font-weight:700">${f1.toFixed(2)} %</td>
        <td class="cell-support">${sup}</td>
      </tr>`;
  }).join('');
}

/* ── Barres F1 par classe ─────────────────────────────────── */
function rendreF1Classes(rapport) {
  const ctx = document.getElementById('chart-f1classes');
  if (!ctx) return;
  if (instanceF1Classes) { instanceF1Classes.destroy(); instanceF1Classes = null; }

  const classes = Object.keys(rapport).filter(k => !k.includes('avg'));
  const labels  = classes.map(c => LABELS_COURTS[c] ?? c);
  const valeurs = classes.map(c => +(rapport[c].f1_score ?? 0).toFixed(2));
  const couleurs = classes.map((_, i) => COULEURS_CLASSES[i] ?? '#94a3b8');

  instanceF1Classes = new Chart(ctx.getContext('2d'), {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label           : 'F1-Score (%)',
        data            : valeurs,
        backgroundColor : couleurs,
        borderRadius    : 5,
        borderSkipped   : false,
      }],
    },
    options: {
      responsive          : true,
      maintainAspectRatio : false,
      animation           : { duration: 800, easing: 'easeOutQuart' },
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
          callbacks       : { label: c => ` F1 : ${c.raw} %` },
        },
      },
      scales: {
        x: {
          grid : { display: false },
          ticks: { color: '#94a3b8', font: { size: 10 } },
        },
        y: {
          min  : 90, max: 101,
          grid : { color: 'rgba(31,41,55,.8)', drawBorder: false },
          ticks: { color: '#4b5563', font: { size: 9 }, callback: v => v + '%' },
        },
      },
    },
  });
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
  } catch {
    if (dot)   dot.className    = 'status-dot offline';
    if (label) label.textContent = 'API hors ligne';
  }
}

/* ── Initialisation ───────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  updateClock();
  setInterval(updateClock, 1000);

  checkAPIStatus();
  setInterval(checkAPIStatus, 30_000);

  chargerMetriques();
});
