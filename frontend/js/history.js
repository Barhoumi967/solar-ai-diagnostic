/* ============================================================
   Solar AI Diagnostic — Logique page Historique
   Chargement, filtres, pagination, graphiques, modal détails,
   export CSV, effacement
   ============================================================ */

/* ── Constantes ───────────────────────────────────────────── */
const LIGNES_PAR_PAGE = 10;

/* ── État de la page ──────────────────────────────────────── */
let tousLesDiags   = [];   /* Données brutes depuis l'API             */
let diagsFiltres   = [];   /* Données après filtres appliqués         */
let pageCourante   = 1;    /* Page actuellement affichée              */
let instanceDonut  = null; /* Instance Chart.js camembert             */
let instanceTimeline = null; /* Instance Chart.js courbe temporelle   */

/* ── Labels des 8 classes de pannes ──────────────────────── */
const LABELS_CL = [
  'Normal', 'Ombrage partiel', 'Court-circuit', 'Circuit ouvert',
  'Dégradation PID', 'Encrassement', 'Défaut connexion', 'Vieillissement',
];

/* ── Couleurs des 8 classes pour le camembert ────────────── */
const COULEURS_CL = [
  '#10b981','#f59e0b','#ef4444','#dc2626',
  '#8b5cf6','#00d4ff','#ff6b35','#94a3b8',
];

/* ── Couleur du point de classe dans le tableau ───────────── */
function couleurClasse(cl) {
  const map = { 0:'var(--accent-succes)', 2:'var(--accent-danger)',
                3:'var(--accent-danger)' };
  return map[cl] ?? 'var(--accent-orange)';
}

/* ── Classe CSS de ligne selon le niveau d'alerte ─────────── */
function rowClass(niveau) {
  const map = { Normal:'row-normal', Attention:'row-attention',
                Alerte:'row-alerte', Critique:'row-critique' };
  return map[niveau] ?? '';
}

/* ── Badge selon le niveau ────────────────────────────────── */
function badgeNiveau(niveau) {
  const cls = { Normal:'badge-succes', Attention:'badge-jaune',
                Alerte:'badge-orange', Critique:'badge-danger' }[niveau] ?? 'badge-muted';
  return `<span class="badge ${cls}">${niveau}</span>`;
}

/* ══════════════════════════════════════════════════════════
   chargerHistorique — Appelle GET /api/history?source=db
   et initialise tout (filtres, graphiques, tableau)
   ══════════════════════════════════════════════════════════ */
async function chargerHistorique() {
  afficherChargementTableau();
  try {
    /* Charge tous les diagnostics d'un coup (pagination côté client) */
    const data = await SolarAPI.loadHistory(200);
    tousLesDiags = data.diagnostics ?? [];
    diagsFiltres = [...tousLesDiags];

    majBadgeTotal(tousLesDiags.length);
    rendreCamembert(tousLesDiags);
    rendreTimeline(tousLesDiags);
    pageCourante = 1;
    rendreTableau();
  } catch (err) {
    document.getElementById('hist-tbody').innerHTML = `
      <tr><td colspan="7" class="cell-loading" style="color:var(--accent-danger)">
        Impossible de charger l'historique — ${err.message}
      </td></tr>`;
  }
}

/* ── Affiche un spinner pendant le chargement ─────────────── */
function afficherChargementTableau() {
  document.getElementById('hist-tbody').innerHTML = `
    <tr><td colspan="7">
      <div class="cell-loading">
        <div class="spinner"></div>
        Chargement depuis la base de données…
      </div>
    </td></tr>`;
}

/* ── Met à jour le badge "N entrées" ─────────────────────── */
function majBadgeTotal(n) {
  const el = document.getElementById('badge-total');
  if (el) el.textContent = `${n} entrée${n !== 1 ? 's' : ''}`;
}

/* ══════════════════════════════════════════════════════════
   rendreCamembert — Doughnut Chart.js : répartition des pannes
   ══════════════════════════════════════════════════════════ */
function rendreCamembert(diagnostics) {
  const ctx = document.getElementById('chart-doughnut');
  if (!ctx) return;

  if (instanceDonut) { instanceDonut.destroy(); instanceDonut = null; }

  /* Compte les occurrences par classe */
  const comptage = Array(8).fill(0);
  diagnostics.forEach(d => { if (d.classe >= 0 && d.classe < 8) comptage[d.classe]++; });

  /* Ne garde que les classes avec au moins 1 occurrence */
  const labels  = LABELS_CL.filter((_, i) => comptage[i] > 0);
  const valeurs = comptage.filter(v => v > 0);
  const couleurs = COULEURS_CL.filter((_, i) => comptage[i] > 0);

  /* Badge de total dans l'en-tête du chart */
  const totalEl = document.getElementById('doughnut-total');
  if (totalEl) totalEl.textContent = `${diagnostics.length} diag.`;

  if (!valeurs.length) {
    ctx.closest('.chart-container').innerHTML =
      '<p style="text-align:center;color:var(--texte-muted);padding:60px 0;font-size:.85rem">Aucune donnée</p>';
    return;
  }

  instanceDonut = new Chart(ctx.getContext('2d'), {
    type: 'doughnut',
    data: {
      labels,
      datasets: [{
        data            : valeurs,
        backgroundColor : couleurs,
        borderColor     : '#111827',
        borderWidth     : 2,
        hoverOffset     : 6,
      }],
    },
    options: {
      responsive          : true,
      maintainAspectRatio : false,
      cutout              : '65%',
      animation           : { duration: 800, easing: 'easeOutQuart' },
      plugins: {
        legend: {
          position  : 'right',
          labels    : {
            color     : '#94a3b8',
            font      : { size: 10, family: "'Inter', sans-serif" },
            boxWidth  : 10,
            padding   : 8,
            generateLabels(chart) {
              const ds = chart.data.datasets[0];
              return chart.data.labels.map((label, i) => ({
                text       : `${label} (${ds.data[i]})`,
                fillStyle  : ds.backgroundColor[i],
                strokeStyle: ds.borderColor,
                lineWidth  : ds.borderWidth,
                hidden     : false,
                index      : i,
              }));
            },
          },
        },
        tooltip: {
          backgroundColor : '#111827',
          borderColor     : '#1f2937',
          borderWidth     : 1,
          titleColor      : '#e2e8f0',
          bodyColor       : '#94a3b8',
          padding         : 8,
          cornerRadius    : 6,
          callbacks       : { label: c => ` ${c.label} : ${c.raw} (${((c.raw/diagnostics.length)*100).toFixed(0)} %)` },
        },
      },
    },
  });
}

/* ══════════════════════════════════════════════════════════
   rendreTimeline — Courbe d'évolution du score de santé
   (100 - score_anomalie) sur les 30 derniers diagnostics
   ══════════════════════════════════════════════════════════ */
function rendreTimeline(diagnostics) {
  const ctx = document.getElementById('chart-timeline');
  if (!ctx) return;

  if (instanceTimeline) { instanceTimeline.destroy(); instanceTimeline = null; }

  /* Prend les 30 plus récents (premier = plus récent en mémoire) */
  const tranche = diagnostics.slice(0, 30).reverse();

  if (!tranche.length) {
    ctx.closest('.chart-container').innerHTML =
      '<p style="text-align:center;color:var(--texte-muted);padding:60px 0;font-size:.85rem">Aucune donnée</p>';
    return;
  }

  /* Score de santé = 100 - score_anomalie */
  const labels  = tranche.map((d, i) => `#${d.id ?? i + 1}`);
  const scores  = tranche.map(d => Math.max(0, 100 - (d.anomalie_score ?? 0)));

  /* Gradient sous la courbe */
  const gradient = ctx.getContext('2d').createLinearGradient(0, 0, 0, 200);
  gradient.addColorStop(0,   'rgba(0,212,255,.28)');
  gradient.addColorStop(0.7, 'rgba(0,212,255,.05)');
  gradient.addColorStop(1,   'rgba(0,212,255,0)');

  instanceTimeline = new Chart(ctx.getContext('2d'), {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label              : 'Score de santé',
        data               : scores,
        borderColor        : '#00d4ff',
        borderWidth        : 2,
        pointRadius        : 3,
        pointHoverRadius   : 6,
        pointBackgroundColor: '#00d4ff',
        fill               : true,
        backgroundColor    : gradient,
        tension            : 0.4,
      }],
    },
    options: {
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
          callbacks       : { label: c => ` Score santé : ${c.raw} / 100` },
        },
      },
      scales: {
        x: {
          grid : { color:'rgba(31,41,55,.8)', drawBorder:false },
          ticks: { color:'#4b5563', font:{size:9}, maxTicksLimit:10 },
        },
        y: {
          min  : 0, max: 100,
          grid : { color:'rgba(31,41,55,.8)', drawBorder:false },
          ticks: { color:'#4b5563', font:{size:9}, callback:v => v + '%' },
        },
      },
    },
  });
}

/* ══════════════════════════════════════════════════════════
   rendreTableau — Affiche la page courante du tableau
   ══════════════════════════════════════════════════════════ */
function rendreTableau() {
  const tbody = document.getElementById('hist-tbody');
  if (!tbody) return;

  /* Calcul des indices de page */
  const total     = diagsFiltres.length;
  const nbPages   = Math.max(1, Math.ceil(total / LIGNES_PAR_PAGE));
  pageCourante    = Math.min(pageCourante, nbPages);
  const debut     = (pageCourante - 1) * LIGNES_PAR_PAGE;
  const page      = diagsFiltres.slice(debut, debut + LIGNES_PAR_PAGE);

  /* Cas vide */
  if (!total) {
    tbody.innerHTML = `
      <tr><td colspan="7" class="cell-empty" style="padding:56px 16px">
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor"
             stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round"
             style="color:var(--texte-muted);opacity:.35;margin:0 auto 14px;display:block">
          <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/>
          <path d="M3 3v5h5"/><path d="M12 7v5l4 2"/>
        </svg>
        Aucun diagnostic enregistré —
        <a href="diagnostic.html" style="color:var(--accent-cyan);text-decoration:underline">
          Lancez une analyse
        </a>
      </td></tr>`;
    majPagination(0, 1);
    return;
  }

  /* Rendu des lignes */
  tbody.innerHTML = page.map((d, idx) => {
    const coulDot = couleurClasse(d.classe);
    const rClass  = rowClass(d.anomalie_niveau ?? d.niveau_alerte);
    const confCoul = (d.confiance ?? 0) >= 90
      ? 'var(--accent-succes)'
      : (d.confiance ?? 0) >= 70
        ? 'var(--texte-principal)'
        : 'var(--accent-orange)';

    return `
      <tr class="${rClass}" style="animation-delay:${idx * 0.03}s">
        <td class="cell-id">#${d.id ?? '—'}</td>
        <td class="cell-ts">${d.timestamp ?? '—'}</td>
        <td>
          <div class="cell-defaut">
            <span class="defaut-dot" style="background:${coulDot}"></span>
            ${d.libelle ?? LABELS_CL[d.classe] ?? '—'}
          </div>
        </td>
        <td class="cell-conf" style="color:${confCoul}">${d.confiance ?? '—'} %</td>
        <td>
          <span class="text-mono" style="font-size:.8rem">${d.anomalie_score ?? '—'}/100</span>
        </td>
        <td>${badgeNiveau(d.anomalie_niveau ?? d.niveau_alerte ?? 'Normal')}</td>
        <td>
          <button class="btn-details" onclick="ouvrirDetails(${debut + idx})">
            👁 Détails
          </button>
        </td>
      </tr>`;
  }).join('');

  majPagination(total, nbPages);
}

/* ── Mise à jour des contrôles de pagination ─────────────── */
function majPagination(total, nbPages) {
  const info = document.getElementById('page-info');
  const prev = document.getElementById('btn-prev');
  const next = document.getElementById('btn-next');

  if (info) info.textContent = `Page ${pageCourante} / ${nbPages}`;
  if (prev) prev.disabled = pageCourante <= 1;
  if (next) next.disabled = pageCourante >= nbPages;
}

/* ── Change la page et redessine le tableau ───────────────── */
function changerPage(delta) {
  pageCourante += delta;
  rendreTableau();
}

/* ══════════════════════════════════════════════════════════
   appliquerFiltres — Filtre côté client selon classe,
   niveau d'alerte et date
   ══════════════════════════════════════════════════════════ */
function appliquerFiltres() {
  const cl    = document.getElementById('f-classe')?.value  ?? '';
  const al    = document.getElementById('f-alerte')?.value  ?? '';
  const date  = document.getElementById('f-date')?.value    ?? '';

  diagsFiltres = tousLesDiags.filter(d => {
    const passeCl   = cl   === '' || String(d.classe) === cl;
    const passeAl   = al   === '' || (d.anomalie_niveau ?? d.niveau_alerte) === al;
    const passeDate = date === '' || (d.timestamp ?? '').startsWith(date);
    return passeCl && passeAl && passeDate;
  });

  pageCourante = 1;
  rendreTableau();
  majBadgeTotal(diagsFiltres.length);
}

/* ── Réinitialise les 3 filtres ───────────────────────────── */
function reinitFiltres() {
  ['f-classe','f-alerte','f-date'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.value = '';
  });
  diagsFiltres = [...tousLesDiags];
  pageCourante = 1;
  rendreTableau();
  majBadgeTotal(tousLesDiags.length);
}

/* ══════════════════════════════════════════════════════════
   ouvrirDetails — Modal avec tous les paramètres mesurés
   ══════════════════════════════════════════════════════════ */
function ouvrirDetails(indexFiltre) {
  const d = diagsFiltres[indexFiltre];
  if (!d) return;

  const modal   = document.getElementById('modal-details');
  const content = document.getElementById('modal-content');
  if (!modal || !content) return;

  /* Labels lisibles des features */
  const LABELS_FEAT = {
    irradiance:'Irradiance', temperature_panneau:'Temp. panneau',
    temperature_ambiante:'Temp. ambiante', tension_voc:'Tension VOC',
    courant_isc:'Courant ISC', puissance_mpp:'Puissance MPP',
    tension_mpp:'Tension MPP', courant_mpp:'Courant MPP',
    resistance_serie:'Résist. série', resistance_shunt:'Résist. shunt',
    fill_factor:'Fill Factor', efficacite:'Efficacité',
    temps_fonctionnement:'Temps fonct.', humidite:'Humidité',
    vitesse_vent:'Vitesse vent', facteur_idealite:'Fact. idéalité',
  };
  const UNITES_FEAT = {
    irradiance:'W/m²', temperature_panneau:'°C', temperature_ambiante:'°C',
    tension_voc:'V', courant_isc:'A', puissance_mpp:'W', tension_mpp:'V',
    courant_mpp:'A', resistance_serie:'Ω', resistance_shunt:'Ω',
    fill_factor:'%', efficacite:'%', temps_fonctionnement:'h',
    humidite:'%', vitesse_vent:'m/s', facteur_idealite:'',
  };

  /* Paramètres mesurés (si disponibles dans l'objet historique) */
  const mesures = d.mesures ?? {};
  const mesuresCles = Object.keys(LABELS_FEAT);
  const hasMesures  = mesuresCles.some(k => mesures[k] !== undefined);

  const mesuresHtml = hasMesures
    ? `<div class="modal-section">Paramètres mesurés</div>
       <div class="modal-params-grid">
         ${mesuresCles.map(k => mesures[k] !== undefined ? `
           <div class="modal-param">
             <div class="mp-label">${LABELS_FEAT[k]} ${UNITES_FEAT[k] ? '('+UNITES_FEAT[k]+')' : ''}</div>
             <div class="mp-val">${Number(mesures[k]).toFixed(2)}</div>
           </div>` : '').join('')}
       </div>`
    : '';

  /* Badge de sévérité */
  const sev = d.classe === 0 ? 'normal' : (d.classe === 2 || d.classe === 3) ? 'danger' : 'warning';
  const sevCoul = { normal:'var(--accent-succes)', warning:'var(--accent-orange)', danger:'var(--accent-danger)' }[sev];

  content.innerHTML = `
    <button class="modal-close" onclick="fermerModal()">✕</button>

    <div class="modal-title">
      Diagnostic #${d.id ?? '—'} —
      <span style="color:${sevCoul}">${d.libelle ?? LABELS_CL[d.classe]}</span>
    </div>

    <!-- Résumé résultat -->
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:4px">
      <div class="modal-param">
        <div class="mp-label">Confiance</div>
        <div class="mp-val" style="color:var(--accent-cyan)">${d.confiance ?? '—'} %</div>
      </div>
      <div class="modal-param">
        <div class="mp-label">Score anomalie</div>
        <div class="mp-val">${d.anomalie_score ?? '—'} / 100</div>
      </div>
      <div class="modal-param">
        <div class="mp-label">Niveau alerte</div>
        <div class="mp-val">${badgeNiveau(d.anomalie_niveau ?? d.niveau_alerte ?? 'Normal')}</div>
      </div>
      <div class="modal-param">
        <div class="mp-label">RUL</div>
        <div class="mp-val">${d.rul_annees != null ? d.rul_annees + ' ans' : '—'}</div>
      </div>
      <div class="modal-param">
        <div class="mp-label">Statut RUL</div>
        <div class="mp-val">${d.rul_statut ?? '—'}</div>
      </div>
      <div class="modal-param">
        <div class="mp-label">Timestamp</div>
        <div class="mp-val" style="font-size:.72rem;color:var(--texte-secondaire)">${d.timestamp ?? '—'}</div>
      </div>
    </div>

    ${mesuresHtml}`;

  modal.classList.remove('hidden');
}

/* ── Ferme le modal détails ───────────────────────────────── */
function fermerModal(evt) {
  /* Ferme si clic sur le fond (pas sur le contenu) */
  if (evt && evt.target.id !== 'modal-details') return;
  document.getElementById('modal-details')?.classList.add('hidden');
}

/* ── Ferme le modal de confirmation ──────────────────────── */
function fermerModalConfirm(evt) {
  if (evt && evt.target.id !== 'modal-confirm') return;
  document.getElementById('modal-confirm')?.classList.add('hidden');
}

/* ── Ouvre le modal de confirmation d'effacement ─────────── */
function confirmerEffacement() {
  document.getElementById('modal-confirm')?.classList.remove('hidden');
}

/* ── Efface tout l'historique ────────────────────────────── */
async function effacerHistorique() {
  fermerModalConfirm();
  try {
    await SolarAPI.clearHistory();
    tousLesDiags = [];
    diagsFiltres = [];
    pageCourante = 1;
    majBadgeTotal(0);
    rendreCamembert([]);
    rendreTimeline([]);
    rendreTableau();
  } catch (err) {
    alert('Impossible de supprimer : ' + err.message);
  }
}

/* ══════════════════════════════════════════════════════════
   exporterCSV — Génère et télécharge le CSV côté client
   ══════════════════════════════════════════════════════════ */
function exporterCSV() {
  if (!diagsFiltres.length) return;

  /* En-tête CSV */
  const colonnes = ['id','timestamp','libelle','classe','confiance',
                    'anomalie_score','anomalie_niveau','rul_annees','rul_statut'];
  const entete   = colonnes.join(';');

  /* Lignes */
  const lignes = diagsFiltres.map(d =>
    colonnes.map(c => {
      const v = d[c] ?? '';
      /* Échappe les virgules et guillemets pour CSV */
      return String(v).includes(';') ? `"${v}"` : v;
    }).join(';')
  );

  const csv     = [entete, ...lignes].join('\n');
  const blob    = new Blob(['﻿' + csv], { type: 'text/csv;charset=utf-8;' });
  const url     = URL.createObjectURL(blob);
  const lien    = document.createElement('a');
  const ts      = new Date().toISOString().slice(0,10);

  lien.href     = url;
  lien.download = `historique_solarai_${ts}.csv`;
  lien.click();
  URL.revokeObjectURL(url);
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

/* ══════════════════════════════════════════════════════════
   Initialisation
   ══════════════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
  updateClock();
  setInterval(updateClock, 1000);

  checkAPIStatus();
  setInterval(checkAPIStatus, 30_000);

  chargerHistorique();
});
