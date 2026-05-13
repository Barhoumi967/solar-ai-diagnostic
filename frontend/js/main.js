/* ============================================================
   Solar AI Diagnostic — Logique principale du Dashboard
   Horloge temps réel, statut API, chargement données, countUp
   ============================================================ */

/* ── Constantes ───────────────────────────────────────────── */

/* Valeurs cibles des KPI (remplacées par les données API si disponibles) */
const KPI_DEFAUT = {
  panneaux  : 12,
  pannes    : 3,
  efficacite: 87.3,
  alertes   : 2,
};

/* Délai entre deux vérifications du statut API (ms) */
const INTERVALLE_API_MS   = 30_000;

/* ── État interne ─────────────────────────────────────────── */
let donneesDemo   = null;   /* Dernier payload /api/demo-data */
let apiConnectee  = false;  /* Etat courant de la connexion   */

/* ══════════════════════════════════════════════════════════
   updateClock — Horloge temps réel
   Appelée toutes les secondes, met à jour #horloge
   ══════════════════════════════════════════════════════════ */
function updateClock() {
  const el = document.getElementById('horloge');
  if (!el) return;

  const now = new Date();

  /* Formatage HH:MM:SS | JJ/MM/AAAA */
  const heure  = now.toLocaleTimeString('fr-FR');
  const date   = now.toLocaleDateString('fr-FR', {
    day  : '2-digit',
    month: '2-digit',
    year : 'numeric',
  });

  el.textContent = `${heure} | ${date}`;
}

/* ══════════════════════════════════════════════════════════
   checkAPIStatus — Ping /api/health toutes les 30 s
   Met à jour le point de statut dans la sidebar
   ══════════════════════════════════════════════════════════ */
async function checkAPIStatus() {
  const dot   = document.getElementById('sidebar-dot');
  const label = document.getElementById('sidebar-label');

  try {
    const data = await SolarAPI.health();

    apiConnectee = data.models_loaded === true;

    if (dot)   dot.className   = `status-dot ${apiConnectee ? 'online' : 'offline'}`;
    if (label) label.textContent = apiConnectee ? 'API Connectée' : 'Modèles absents';

    /* Met à jour le badge statut global dans le header */
    majBadgeStatut(apiConnectee);

  } catch {
    /* Serveur injoignable */
    apiConnectee = false;
    if (dot)   dot.className    = 'status-dot offline';
    if (label) label.textContent = 'API hors ligne';
    majBadgeStatut(false);
  }
}

/* Met à jour le badge OPÉRATIONNEL / ALERTE dans le header */
function majBadgeStatut(connecte) {
  const badge = document.getElementById('badge-statut');
  if (!badge) return;

  if (connecte) {
    badge.className   = 'badge badge-succes';
    badge.innerHTML   =
      `<svg width="8" height="8" viewBox="0 0 8 8">
         <circle cx="4" cy="4" r="4" fill="currentColor"/>
       </svg> OPÉRATIONNEL`;
  } else {
    badge.className   = 'badge badge-orange';
    badge.innerHTML   =
      `<svg width="8" height="8" viewBox="0 0 8 8">
         <circle cx="4" cy="4" r="4" fill="currentColor"/>
       </svg> ALERTE`;
  }
}

/* ══════════════════════════════════════════════════════════
   countUp — Animation numérique de 0 vers la valeur cible
   ══════════════════════════════════════════════════════════ */
/**
 * @param {string} idElement  — id du DOM à animer
 * @param {number} valeurFin  — valeur finale à afficher
 * @param {number} dureeMs    — durée totale de l'animation en ms
 * @param {number} decimales  — nombre de décimales (0 par défaut)
 * @param {string} suffixe    — suffixe affiché après la valeur (ex: "%")
 */
function countUp(idElement, valeurFin, dureeMs = 1000, decimales = 0, suffixe = '') {
  const el = document.getElementById(idElement);
  if (!el) return;

  const debut     = performance.now();
  const valeurDep = 0;

  function etape(maintenant) {
    const progres  = Math.min((maintenant - debut) / dureeMs, 1);
    /* Easing ease-out cubique */
    const ease     = 1 - Math.pow(1 - progres, 3);
    const valActu  = valeurDep + (valeurFin - valeurDep) * ease;

    el.textContent = valActu.toFixed(decimales) + suffixe;

    if (progres < 1) requestAnimationFrame(etape);
  }

  requestAnimationFrame(etape);
}

/* ══════════════════════════════════════════════════════════
   calculerScoresRadar
   Calcule les 6 scores santé (0–100) depuis les points 24h.
   Utilise les données de la plage 10h–15h (pic solaire).
   ══════════════════════════════════════════════════════════ */
function calculerScoresRadar(points) {
  /* Sélectionne les heures de pic (10h à 15h) */
  const pic = points.filter(p => p.heure >= 10 && p.heure <= 15);
  if (!pic.length) return null;

  /* Efficacité max théorique pour le panneau de test (19.5 %) */
  const EFF_MAX  = 19.5;
  /* Irradiance max de référence (1000 W/m²) */
  const IRR_REF  = 1000;
  /* Puissance de référence en W (panneau 360 W) */
  const PUISS_REF = 360;

  const effMoy  = pic.reduce((s, p) => s + p.efficacite, 0) / pic.length;
  const irrMoy  = pic.reduce((s, p) => s + p.irradiance, 0) / pic.length;
  const puisMoy = pic.reduce((s, p) => s + p.puissance,  0) / pic.length;
  const tempMoy = pic.reduce((s, p) => s + p.temperature_panneau, 0) / pic.length;

  /* Nombre d'alertes sur la journée — dégradation de la note connexions */
  const nbAlertes = points.filter(p => p.alerte).length;

  return {
    /* Puissance normalisée sur la puissance de référence */
    puissance  : Math.min(100, Math.round((puisMoy / PUISS_REF) * 100)),
    /* Efficacité normalisée sur 19.5% */
    efficacite : Math.min(100, Math.round((effMoy  / EFF_MAX)   * 100)),
    /* Irradiance reçue vs référence (proxy "tension") */
    tension    : Math.min(100, Math.round((irrMoy  / IRR_REF)   * 100)),
    /* Score courant inversement proportionnel aux anomalies détectées */
    courant    : Math.max(50, 95 - nbAlertes * 10),
    /* Score température : pénalise si > 65 °C */
    temperature: tempMoy > 0 ? Math.max(40, Math.round(100 - (tempMoy - 40) * 1.2)) : 75,
    /* Connexions parfaites sauf si anomalies */
    connexions : Math.max(60, 100 - nbAlertes * 8),
  };
}

/* ══════════════════════════════════════════════════════════
   loadDashboardData — Charge /api/demo-data et met à jour
   les KPI, les graphiques et le badge santé
   ══════════════════════════════════════════════════════════ */
async function loadDashboardData() {
  try {
    const payload = await SolarAPI.loadDemoData();
    donneesDemo   = payload.points;

    /* ── Calcul des KPI depuis les points ── */
    const nbPanneaux  = 12;
    const alertes     = donneesDemo.filter(p => p.alerte);
    const nbAlertes   = alertes.length;
    const nbPannes    = nbAlertes + 1; /* +1 : défaut #07 simulé */

    /* Efficacité moyenne sur la plage de pic (10h–16h) */
    const picPts  = donneesDemo.filter(p => p.heure >= 10 && p.heure <= 16);
    const effMoy  = picPts.length
      ? picPts.reduce((s, p) => s + p.efficacite, 0) / picPts.length
      : KPI_DEFAUT.efficacite;

    /* ── Animations countUp des 4 KPI ── */
    countUp('kpi-panneaux',   nbPanneaux, 900,  0, '');
    countUp('kpi-pannes',     nbPannes,   900,  0, '');
    countUp('kpi-efficacite', effMoy,     1100, 1, '%');
    countUp('kpi-alertes',    nbAlertes,  900,  0, '');

    /* Animation cloche si alertes > 0 */
    const iconeAlerte = document.getElementById('kpi-alerte-icon');
    if (iconeAlerte) {
      iconeAlerte.classList.toggle('alerte-active', nbAlertes > 0);
    }

    /* ── Graphique production ── */
    initProductionChart(donneesDemo);

    /* ── Radar santé ── */
    const scores = calculerScoresRadar(donneesDemo);
    initRadarChart(scores);

    /* ── Badge santé ── */
    const badgeSante = document.getElementById('badge-sante');
    if (badgeSante && scores) {
      const moyenne = Object.values(scores).reduce((a, b) => a + b, 0)
                      / Object.values(scores).length;
      if (moyenne >= 80) {
        badgeSante.textContent = 'Nominal';
        badgeSante.className   = 'badge badge-succes';
      } else if (moyenne >= 60) {
        badgeSante.textContent = 'Attention';
        badgeSante.className   = 'badge badge-orange';
      } else {
        badgeSante.textContent = 'Critique';
        badgeSante.className   = 'badge badge-danger';
      }
    }

  } catch (err) {
    /* Echec API — affiche les valeurs KPI statiques par défaut */
    console.warn('[SolarAI] Données demo indisponibles, valeurs par défaut :', err.message);
    countUp('kpi-panneaux',   KPI_DEFAUT.panneaux,   800, 0, '');
    countUp('kpi-pannes',     KPI_DEFAUT.pannes,      800, 0, '');
    countUp('kpi-efficacite', KPI_DEFAUT.efficacite, 1000, 1, '%');
    countUp('kpi-alertes',    KPI_DEFAUT.alertes,     800, 0, '');

    /* Graphiques avec données statiques de démonstration */
    initRadarChart(null);
    initProductionChart(genererPointsStatiques());
  }
}

/* ── genererPointsStatiques ───────────────────────────────── */
/**
 * Génère 24 points simulés côté client pour les graphiques
 * quand l'API est indisponible.
 */
function genererPointsStatiques() {
  return Array.from({ length: 24 }, (_, h) => {
    /* Courbe gaussienne centrée à 13h */
    const sigma = 3.5;
    const irr   = Math.max(0, 1050 * Math.exp(-0.5 * Math.pow((h - 13) / sigma, 2)));
    const puiss = irr > 50 ? irr * 0.33 : 0;
    return {
      heure       : h,
      irradiance  : +irr.toFixed(1),
      puissance   : +puiss.toFixed(1),
      efficacite  : 18.2 + Math.random() * 1.5,
      temperature_panneau: 25 + (irr / 1050) * 32,
      alerte      : null,
    };
  });
}

/* ══════════════════════════════════════════════════════════
   Initialisation au chargement du DOM
   ══════════════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {

  /* Horloge — une mise à jour immédiate puis toutes les secondes */
  updateClock();
  setInterval(updateClock, 1000);

  /* Statut API — vérification immédiate puis toutes les 30 s */
  checkAPIStatus();
  setInterval(checkAPIStatus, INTERVALLE_API_MS);

  /* Données dashboard — chargement initial */
  loadDashboardData();
});
