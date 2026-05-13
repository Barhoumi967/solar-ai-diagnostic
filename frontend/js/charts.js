/* ============================================================
   Solar AI Diagnostic — Graphiques Chart.js
   Production 24h (courbe + gradient) + Radar santé système
   ============================================================ */

/* ── Configuration globale Chart.js ──────────────────────── */
Chart.defaults.color       = '#94a3b8';
Chart.defaults.borderColor = '#1f2937';
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.font.size   = 11;

/* Références aux instances pour mise à jour sans recréation */
let instanceProduction = null;
let instanceRadar      = null;

/* ── Style tooltip commun (fond sombre) ───────────────────── */
const tooltipSombre = {
  backgroundColor : '#111827',
  borderColor     : '#1f2937',
  borderWidth     : 1,
  titleColor      : '#e2e8f0',
  bodyColor       : '#94a3b8',
  padding         : 10,
  cornerRadius    : 8,
  titleFont       : { family: "'Rajdhani', sans-serif", weight: '700', size: 13 },
  bodyFont        : { family: "'Inter', sans-serif", size: 11 },
};

/* ── Grille discrète commune ──────────────────────────────── */
const grilleDiscrete = {
  color    : 'rgba(31, 41, 55, 0.9)',
  drawBorder: false,
  lineWidth : 1,
};

/* ══════════════════════════════════════════════════════════
   initProductionChart
   Courbe de puissance (kW) avec gradient cyan + courbe
   irradiance (W/m²) en pointillés orange sur axe secondaire.
   ══════════════════════════════════════════════════════════ */
/**
 * @param {Array} pointsData  — tableau de points /api/demo-data
 *                              { heure, puissance, irradiance }
 */
function initProductionChart(pointsData) {
  const canvas = document.getElementById('chart-production');
  if (!canvas) return;

  /* Destruction de l'instance précédente si elle existe */
  if (instanceProduction) {
    instanceProduction.destroy();
    instanceProduction = null;
  }

  const ctx    = canvas.getContext('2d');
  const labels = pointsData.map(p => `${String(p.heure).padStart(2, '0')}h`);

  /* Puissance convertie en kW */
  const puissances  = pointsData.map(p => +(p.puissance / 1000).toFixed(3));
  const irradiances = pointsData.map(p => p.irradiance);

  /* Gradient vertical cyan sous la courbe puissance */
  const hauteur      = canvas.offsetHeight || 220;
  const gradientCyan = ctx.createLinearGradient(0, 0, 0, hauteur);
  gradientCyan.addColorStop(0,   'rgba(0, 212, 255, 0.35)');
  gradientCyan.addColorStop(0.6, 'rgba(0, 212, 255, 0.08)');
  gradientCyan.addColorStop(1,   'rgba(0, 212, 255, 0.00)');

  instanceProduction = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          /* Courbe puissance (kW) — plein + gradient */
          label              : 'Puissance (kW)',
          data               : puissances,
          borderColor        : '#00d4ff',
          borderWidth        : 2,
          pointRadius        : 2,
          pointHoverRadius   : 5,
          pointBackgroundColor: '#00d4ff',
          fill               : true,
          backgroundColor    : gradientCyan,
          tension            : 0.45,
          yAxisID            : 'y',
        },
        {
          /* Courbe irradiance (W/m²) — pointillés orange */
          label           : 'Irradiance (W/m²)',
          data            : irradiances,
          borderColor     : '#ff6b35',
          borderWidth     : 1.5,
          borderDash      : [5, 4],
          pointRadius     : 0,
          pointHoverRadius: 4,
          fill            : false,
          tension         : 0.4,
          yAxisID         : 'y2',
        },
      ],
    },
    options: {
      responsive          : true,
      maintainAspectRatio : false,
      interaction         : { mode: 'index', intersect: false },
      animation           : { duration: 800, easing: 'easeOutQuart' },
      plugins: {
        legend : { display: false },
        tooltip: {
          ...tooltipSombre,
          callbacks: {
            /* Formate chaque ligne de l'infobulle */
            label(ctx) {
              if (ctx.datasetIndex === 0)
                return ` Puissance : ${Number(ctx.raw).toFixed(3)} kW`;
              return ` Irradiance : ${Number(ctx.raw).toFixed(0)} W/m²`;
            },
          },
        },
      },
      scales: {
        x: {
          grid : grilleDiscrete,
          ticks: {
            maxTicksLimit: 8,
            color        : '#4b5563',
            font         : { size: 10 },
          },
        },
        /* Axe gauche — puissance kW */
        y: {
          position: 'left',
          grid    : grilleDiscrete,
          ticks   : {
            color   : '#00d4ff',
            font    : { size: 10 },
            callback: v => v.toFixed(1) + ' kW',
          },
        },
        /* Axe droit — irradiance W/m² */
        y2: {
          position: 'right',
          grid    : { display: false },
          ticks   : {
            color   : '#ff6b35',
            font    : { size: 10 },
            callback: v => v.toFixed(0),
          },
        },
      },
    },
  });
}

/* ══════════════════════════════════════════════════════════
   initRadarChart
   Radar santé système à 6 axes (scores 0–100).
   Remplissage rgba(0,212,255,0.2) + contour cyan.
   ══════════════════════════════════════════════════════════ */
/**
 * @param {Object|null} scores  — { puissance, efficacite, tension,
 *                                  courant, temperature, connexions }
 *                                Chaque valeur entre 0 et 100.
 *                                Passe null pour utiliser les valeurs par défaut.
 */
function initRadarChart(scores) {
  const canvas = document.getElementById('chart-radar');
  if (!canvas) return;

  if (instanceRadar) {
    instanceRadar.destroy();
    instanceRadar = null;
  }

  /* Valeurs par défaut si aucune donnée réelle disponible */
  const v = scores || {
    puissance: 87, efficacite: 84, tension: 91,
    courant  : 78, temperature: 72, connexions: 95,
  };

  const ctx = canvas.getContext('2d');

  instanceRadar = new Chart(ctx, {
    type: 'radar',
    data: {
      labels  : ['Puissance', 'Efficacité', 'Tension', 'Courant', 'Température', 'Connexions'],
      datasets: [
        {
          label                : 'Santé système',
          data                 : [v.puissance, v.efficacite, v.tension,
                                   v.courant,  v.temperature, v.connexions],
          borderColor          : '#00d4ff',
          borderWidth          : 2,
          backgroundColor      : 'rgba(0, 212, 255, 0.15)',
          pointBackgroundColor : '#00d4ff',
          pointBorderColor     : '#0a0e1a',
          pointBorderWidth     : 2,
          pointRadius          : 4,
          pointHoverRadius     : 6,
        },
      ],
    },
    options: {
      responsive          : true,
      maintainAspectRatio : false,
      animation           : { duration: 900, easing: 'easeOutQuart' },
      plugins: {
        legend : { display: false },
        tooltip: {
          ...tooltipSombre,
          callbacks: {
            label: ctx => ` ${ctx.label} : ${ctx.raw} / 100`,
          },
        },
      },
      scales: {
        r: {
          min        : 0,
          max        : 100,
          beginAtZero: true,
          grid       : { color: 'rgba(31, 41, 55, 0.9)' },
          angleLines : { color: 'rgba(31, 41, 55, 0.9)' },
          pointLabels: {
            color: '#94a3b8',
            font : { family: "'Rajdhani', sans-serif", size: 12, weight: '600' },
          },
          ticks: {
            stepSize      : 25,
            color         : '#4b5563',
            font          : { size: 9 },
            backdropColor : 'transparent',
          },
        },
      },
    },
  });
}

/* ── Mise à jour production sans recréer l'instance ──────── */
/**
 * Rafraîchit les séries du graphique production lors d'un polling.
 * @param {Array} pointsData — mêmes points que initProductionChart
 */
function mettreAJourProduction(pointsData) {
  if (!instanceProduction) { initProductionChart(pointsData); return; }

  instanceProduction.data.labels =
    pointsData.map(p => `${String(p.heure).padStart(2, '0')}h`);
  instanceProduction.data.datasets[0].data =
    pointsData.map(p => +(p.puissance / 1000).toFixed(3));
  instanceProduction.data.datasets[1].data =
    pointsData.map(p => p.irradiance);

  instanceProduction.update('active');
}

/* ── Mise à jour radar sans recréer l'instance ───────────── */
/**
 * Rafraîchit les scores du radar lors d'un polling.
 * @param {Object} scores — mêmes clés que initRadarChart
 */
function mettreAJourRadar(scores) {
  if (!instanceRadar) { initRadarChart(scores); return; }

  const v = scores || {};
  instanceRadar.data.datasets[0].data = [
    v.puissance   ?? 87,
    v.efficacite  ?? 84,
    v.tension     ?? 91,
    v.courant     ?? 78,
    v.temperature ?? 72,
    v.connexions  ?? 95,
  ];
  instanceRadar.update('active');
}
