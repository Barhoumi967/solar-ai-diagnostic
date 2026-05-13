"""
Générateur de rapports PDF professionnels — Solar AI Diagnostic.

Produit un document 7 pages avec en-tête/pied de page sur chaque page :
  Page 1 — Page de garde institutionnelle
  Page 2 — Résumé exécutif & score de santé global
  Page 3 — Méthodes IA : principes, formules, hyperparamètres
  Page 4 — Résultats IA détaillés + analyse causale
  Page 5 — Analyse visuelle (probabilités + paramètres normalisés)
  Page 6 — Recommandations techniques & plan d'action
  Page 7 — Architecture système, métriques & référentiel de classes
"""

import io
from datetime import datetime
from pathlib  import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from reportlab.lib            import colors
from reportlab.lib.pagesizes  import A4
from reportlab.lib.styles     import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units      import cm, mm
from reportlab.lib.enums      import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus       import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, Image, KeepTogether, ListFlowable, ListItem,
)
from reportlab.pdfgen         import canvas as pdf_canvas

# ── Répertoire de sortie ─────────────────────────────────────────────
REPORTS_DIR = Path(__file__).resolve().parent.parent / "data" / "reports"

# ── Classes de pannes ────────────────────────────────────────────────
FAULT_LABELS = {
    0: "Normal",
    1: "Ombrage partiel",
    2: "Court-circuit",
    3: "Circuit ouvert",
    4: "Dégradation PID",
    5: "Encrassement",
    6: "Défaut connexion",
    7: "Vieillissement accéléré",
}

# ── Couleurs accent par classe ───────────────────────────────────────
COULEURS_CLASSE = {
    0: "#10b981", 1: "#f59e0b", 2: "#ef4444", 3: "#dc2626",
    4: "#8b5cf6", 5: "#00d4ff", 6: "#ff6b35", 7: "#94a3b8",
}

COULEURS_NIVEAU = {
    "Normal": "#10b981", "Attention": "#f59e0b",
    "Alerte": "#ff6b35",  "Critique":  "#ef4444",
}

# ── Labels mesures ───────────────────────────────────────────────────
LABELS_MESURES = {
    "irradiance":           ("Irradiance solaire",        "W/m²",  0,    1200),
    "temperature_panneau":  ("Température cellule",        "°C",   -10,   100),
    "temperature_ambiante": ("Température ambiante",       "°C",   -20,    60),
    "tension_voc":          ("Tension VOC",                "V",      0,    60),
    "courant_isc":          ("Courant ISC",                "A",      0,    15),
    "puissance_mpp":        ("Puissance MPP",              "W",      0,   500),
    "tension_mpp":          ("Tension MPP",                "V",      0,    55),
    "courant_mpp":          ("Courant MPP",                "A",      0,    15),
    "resistance_serie":     ("Résistance série",           "Ω",      0,    10),
    "resistance_shunt":     ("Résistance shunt",           "Ω",      1, 50000),
    "fill_factor":          ("Fill Factor",                "%",      0,   100),
    "efficacite":           ("Efficacité conversion",      "%",      0,    30),
    "temps_fonctionnement": ("Temps fonctionnement",       "h",      0, 100000),
    "humidite":             ("Humidité relative",          "%",      0,   100),
    "vitesse_vent":         ("Vitesse du vent",            "m/s",    0,    50),
    "facteur_idealite":     ("Facteur d'idéalité diode",  "",      1.0,   2.5),
}

# Seuils nominaux de référence panneau 360 W STC
NOMINAUX = {
    "irradiance": 1000, "temperature_panneau": 25, "temperature_ambiante": 20,
    "tension_voc": 41.5, "courant_isc": 10.2, "puissance_mpp": 360,
    "tension_mpp": 35.8, "courant_mpp": 10.1, "resistance_serie": 0.3,
    "resistance_shunt": 7500, "fill_factor": 76, "efficacite": 19.5,
    "temps_fonctionnement": 8000, "humidite": 35, "vitesse_vent": 5,
    "facteur_idealite": 1.2,
}

# Importance XGBoost (% — issue de l'entraînement sur le dataset de 6 000 lignes)
FEATURE_IMPORTANCE = {
    "ratio_vmpp_voc":       33.6,
    "fill_factor":          14.2,
    "resistance_serie":     10.8,
    "ratio_impp_isc":        9.1,
    "efficacite":            7.5,
    "puissance_mpp":         5.9,
    "resistance_shunt":      4.7,
    "score_vieillissement":  4.1,
    "performance_ratio":     3.6,
    "tension_voc":           2.8,
    "courant_isc":           1.9,
    "delta_temperature":     0.8,
    "autres":                1.0,
}

# Descriptions courtes et longues des 8 classes
DESCRIPTIONS_CLASSES = {
    "Normal":
        "Tous les paramètres sont dans les plages nominales. Le panneau fonctionne correctement.",
    "Ombrage partiel":
        "Une partie de la surface est obstruée (poussière, feuille, ombre portée). "
        "Le courant ISC chute de façon sélective sur les cellules masquées.",
    "Court-circuit":
        "Court-circuit interne — la tension VOC s'effondre (<15 V). "
        "Risque d'endommagement permanent par points chauds.",
    "Circuit ouvert":
        "Rupture de la chaîne électrique — le courant est quasi nul. "
        "Connexion ou cellule défaillante dans le circuit série.",
    "Dégradation PID":
        "Potential-Induced Degradation — fuite de courant vers le châssis sous haute tension. "
        "Efficacité < 60 % du nominal. Causée par les ions sodium migrant dans l'encapsulant.",
    "Encrassement":
        "Dépôt de poussière, pollen ou saleté sur la surface vitrée. "
        "Réduit l'irradiance effective reçue par les cellules sans altérer l'électronique.",
    "Défaut connexion":
        "Résistance de contact élevée (Rs > 1 Ω) due à l'oxydation ou mauvais sertissage. "
        "Augmente les pertes Joule et réduit la puissance de sortie.",
    "Vieillissement accéléré":
        "Dégradation progressive irréversible des cellules (EVA, contacts). "
        "Efficacité et Fill Factor durablement réduits. Mécanismes : délaminage, corrosion.",
}

DESCRIPTIONS_PHYSIQUES = {
    "Normal":
        "État de référence STC (Standard Test Conditions) : irradiance 1 000 W/m², "
        "température cellule 25 °C, masse d'air AM 1.5. Courbe IV optimale.",
    "Ombrage partiel":
        "Les cellules ombragées consomment de la puissance au lieu d'en produire (diodes bypass). "
        "L'ISC chute en proportion de la surface masquée. Peut causer des hot-spots.",
    "Court-circuit":
        "Shunt résistif interne entre p-n junction. VOC ≈ 0 car les porteurs se recombinent "
        "avant d'atteindre les électrodes. Fill Factor effondré (< 30 %).",
    "Circuit ouvert":
        "Rupture dans le chemin série (soudure, connecteur cassé, cellule fracturée). "
        "ISC ≈ 0 A mais VOC peut rester nominal car la jonction est intacte.",
    "Dégradation PID":
        "Migration des ions Na+ sous champ électrique vers les contacts avant. "
        "Augmente la recombinaison en surface. Réversible partiellement par traitement thermique.",
    "Encrassement":
        "Atténuation uniforme du flux solaire. η inchangée mais puissance réduite "
        "proportionnellement. Réversible par nettoyage — opération la moins coûteuse.",
    "Défaut connexion":
        "Rs élevée ≈ augmentation pertes I²R. Point de fonctionnement MPP décalé vers "
        "les basses tensions. Peut causer des arcs électriques dans le câblage.",
    "Vieillissement accéléré":
        "Dégradation annuelle typique : 0.5–1 % / an. Facteurs : UV, thermique, humidité. "
        "Efficacité < 80 % du nominal = fin de vie conventionnelle (garantie 25 ans).",
}

# ── Palette ──────────────────────────────────────────────────────────
C_BG      = colors.HexColor("#0a0e1a")
C_SURFACE = colors.HexColor("#111827")
C_SURF2   = colors.HexColor("#1c2537")
C_BORDER  = colors.HexColor("#1f2937")
C_CYAN    = colors.HexColor("#00d4ff")
C_CYAN_D  = colors.HexColor("#0a2030")
C_SUCCES  = colors.HexColor("#10b981")
C_ORANGE  = colors.HexColor("#f59e0b")
C_DANGER  = colors.HexColor("#ef4444")
C_VIOLET  = colors.HexColor("#8b5cf6")
C_TEXTE   = colors.HexColor("#e2e8f0")
C_MUTED   = colors.HexColor("#64748b")
C_WHITE   = colors.white
C_BLACK   = colors.black
W_PAGE    = A4[0] - 4*cm   # largeur utile


# ════════════════════════════════════════════════════════════════════
# En-tête / pied de page sur chaque page (via canvas callback)
# ════════════════════════════════════════════════════════════════════
class _NumeroteurPages:
    """Dessine un bandeau haut et un pied de page sur chaque feuille."""

    def __init__(self, titre_rapport: str, timestamp: str, total: int = 7):
        self.titre  = titre_rapport
        self.ts     = timestamp
        self.total  = total

    def __call__(self, canvas_obj, doc):
        canvas_obj.saveState()
        w, h = A4

        # ── Bandeau supérieur ─────────────────────────────────────
        canvas_obj.setFillColor(C_BG)
        canvas_obj.rect(0, h - 28, w, 28, fill=1, stroke=0)
        canvas_obj.setFillColor(C_CYAN)
        canvas_obj.rect(0, h - 29, w, 1.5, fill=1, stroke=0)
        canvas_obj.setFillColor(C_CYAN)
        canvas_obj.setFont("Helvetica-Bold", 8)
        canvas_obj.drawString(2*cm, h - 19, "☀  SOLAR AI DIAGNOSTIC")
        canvas_obj.setFillColor(colors.HexColor("#64748b"))
        canvas_obj.setFont("Helvetica", 7.5)
        canvas_obj.drawRightString(w - 2*cm, h - 19, self.titre)

        # ── Pied de page ──────────────────────────────────────────
        canvas_obj.setFillColor(colors.HexColor("#1f2937"))
        canvas_obj.rect(0, 0, w, 22, fill=1, stroke=0)
        canvas_obj.setFillColor(C_CYAN)
        canvas_obj.rect(0, 22, w, 1, fill=1, stroke=0)
        canvas_obj.setFillColor(colors.HexColor("#64748b"))
        canvas_obj.setFont("Helvetica", 7)
        canvas_obj.drawString(2*cm, 8,
            f"INSAT Tunis — Génie Maintenance & Instrumentation — {self.ts}")
        canvas_obj.drawRightString(w - 2*cm, 8,
            f"Page {doc.page} / {self.total}")

        canvas_obj.restoreState()


# ════════════════════════════════════════════════════════════════════
# Fabrique de styles
# ════════════════════════════════════════════════════════════════════
def _mk_styles() -> dict:
    base = getSampleStyleSheet()

    def ps(name, **kw) -> ParagraphStyle:
        parent = kw.pop("parent", base["Normal"])
        return ParagraphStyle(name, parent=parent, **kw)

    return {
        "h1": ps("h1", fontSize=24, textColor=C_CYAN, fontName="Helvetica-Bold",
                  alignment=TA_CENTER, spaceAfter=4, leading=28),
        "h2": ps("h2", fontSize=13, textColor=C_CYAN, fontName="Helvetica-Bold",
                  spaceBefore=14, spaceAfter=6),
        "h3": ps("h3", fontSize=10.5, textColor=C_CYAN, fontName="Helvetica-Bold",
                  spaceBefore=10, spaceAfter=4),
        "h4": ps("h4", fontSize=9.5, textColor=colors.HexColor("#94a3b8"),
                  fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=3),
        "body": ps("body", fontSize=9.5, textColor=C_TEXTE, leading=15,
                   spaceAfter=5, alignment=TA_JUSTIFY),
        "body_muted": ps("body_muted", fontSize=9, textColor=C_MUTED,
                         leading=14, spaceAfter=4),
        "formule": ps("formule", fontSize=9.5, textColor=C_CYAN, fontName="Courier",
                      leading=15, spaceAfter=4, leftIndent=12),
        "mono": ps("mono", fontSize=9, textColor=C_CYAN, fontName="Courier",
                   leading=13),
        "garde_titre": ps("garde_titre", fontSize=26, textColor=C_CYAN,
                          fontName="Helvetica-Bold", alignment=TA_CENTER,
                          leading=32, spaceAfter=6),
        "garde_sous":  ps("garde_sous",  fontSize=12, textColor=C_WHITE,
                          alignment=TA_CENTER, leading=18, spaceAfter=4),
        "garde_inst":  ps("garde_inst",  fontSize=9,  textColor=C_MUTED,
                          alignment=TA_CENTER, leading=14, spaceAfter=3),
        "li": ps("li", fontSize=9.5, textColor=C_TEXTE, leading=15,
                 spaceAfter=4, leftIndent=8),
        "caption": ps("caption", fontSize=7.5, textColor=C_MUTED,
                      alignment=TA_CENTER, spaceAfter=4),
        "kpi_val": ps("kpi_val", fontSize=18, textColor=C_CYAN,
                      fontName="Helvetica-Bold", alignment=TA_CENTER),
        "kpi_lbl": ps("kpi_lbl", fontSize=7.5, textColor=C_MUTED,
                      alignment=TA_CENTER),
        "kpi_sub": ps("kpi_sub", fontSize=8, textColor=C_TEXTE,
                      alignment=TA_CENTER),
        "alerte_titre": ps("alerte_titre", fontSize=10, textColor=C_DANGER,
                           fontName="Helvetica-Bold", alignment=TA_CENTER),
    }


# ════════════════════════════════════════════════════════════════════
# Utilitaires de mise en page
# ════════════════════════════════════════════════════════════════════

def _hr(color=C_BORDER, thick=0.5, space_before=6, space_after=8):
    return HRFlowable(width="100%", thickness=thick, color=color,
                      spaceBefore=space_before, spaceAfter=space_after)


def _table_base(data, col_widths, style_extra=None):
    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    base = [
        ("BACKGROUND",    (0, 0), (-1, 0),  C_BG),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  C_CYAN),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0),  8.5),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_SURFACE, C_SURF2]),
        ("TEXTCOLOR",     (0, 1), (-1, -1), C_TEXTE),
        ("FONTSIZE",      (0, 1), (-1, -1), 9),
        ("GRID",          (0, 0), (-1, -1), 0.3, C_BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]
    if style_extra:
        base += style_extra
    tbl.setStyle(TableStyle(base))
    return tbl


def _encadre(contenu_list, bg=C_SURFACE, border=C_CYAN, padding=10):
    inner = Table([[el] for el in contenu_list],
                  colWidths=[W_PAGE - 2.2*cm])
    inner.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), bg),
        ("GRID",          (0, 0), (-1, -1), 0, colors.transparent),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
    ]))
    outer = Table([[inner]], colWidths=[W_PAGE])
    outer.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), bg),
        ("BOX",           (0, 0), (-1, -1), 1.5, border),
        ("TOPPADDING",    (0, 0), (-1, -1), padding),
        ("BOTTOMPADDING", (0, 0), (-1, -1), padding),
        ("LEFTPADDING",   (0, 0), (-1, -1), padding),
        ("RIGHTPADDING",  (0, 0), (-1, -1), padding),
    ]))
    return outer


# ════════════════════════════════════════════════════════════════════
# Graphiques Matplotlib
# ════════════════════════════════════════════════════════════════════

def _graphique_probas(probabilites: dict, panne_predite: str,
                      width_cm=15.5, height_cm=7.5) -> io.BytesIO:
    entrees  = sorted(probabilites.items(), key=lambda x: x[1], reverse=True)
    labels   = [l for l, _ in entrees]
    valeurs  = [v for _, v in entrees]

    couleurs_bar = []
    for l, v in entrees:
        if l == panne_predite:
            couleurs_bar.append("#00d4ff")
        elif v >= 10:
            couleurs_bar.append("#f59e0b")
        else:
            couleurs_bar.append("#2d3748")

    fig, ax = plt.subplots(figsize=(width_cm / 2.54, height_cm / 2.54))
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")

    bars = ax.barh(labels, valeurs, color=couleurs_bar,
                   height=0.52, edgecolor="none")

    for bar, val in zip(bars, valeurs):
        if val > 0.5:
            ax.text(min(val + 1.5, 105), bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", ha="left",
                    color="#e2e8f0", fontsize=8, fontweight="bold")

    ax.set_xlim(0, 110)
    ax.set_xlabel("Probabilité assignée par le modèle XGBoost (%)",
                  color="#64748b", fontsize=8)
    ax.tick_params(colors="#94a3b8", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#2d3748")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    patch_pred  = mpatches.Patch(color="#00d4ff", label="Classe prédite")
    patch_2nd   = mpatches.Patch(color="#f59e0b", label="Prob. ≥ 10 %")
    patch_other = mpatches.Patch(color="#2d3748", label="Prob. négligeable")
    ax.legend(handles=[patch_pred, patch_2nd, patch_other],
              loc="lower right", framealpha=0.15,
              facecolor="#111827", edgecolor="#2d3748",
              labelcolor="#94a3b8", fontsize=7.5)

    plt.tight_layout(pad=0.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


def _graphique_sante(score: int) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(2.8, 2.8),
                           subplot_kw=dict(aspect="equal"))
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")

    coul = "#10b981" if score >= 70 else "#f59e0b" if score >= 40 else "#ef4444"

    theta = np.linspace(np.pi, 0, 200)
    ax.plot(np.cos(theta), np.sin(theta), lw=12, color="#1f2937",
            solid_capstyle="round")
    theta_s = np.linspace(np.pi, np.pi - (score / 100) * np.pi, 200)
    ax.plot(np.cos(theta_s), np.sin(theta_s), lw=12, color=coul,
            solid_capstyle="round")

    ax.text(0, 0.08, str(score), ha="center", va="center",
            fontsize=28, fontweight="bold", color=coul)
    ax.text(0, -0.35, "/ 100", ha="center", va="center",
            fontsize=10, color="#64748b")
    ax.text(0, -0.65, "Score de santé", ha="center", va="center",
            fontsize=7.5, color="#94a3b8")

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.9, 1.2)
    ax.axis("off")

    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="#111827")
    plt.close(fig)
    buf.seek(0)
    return buf


def _graphique_parametres(mesures: dict) -> io.BytesIO:
    cles_importantes = [
        "tension_voc", "courant_isc", "puissance_mpp", "fill_factor",
        "efficacite", "resistance_serie", "resistance_shunt", "irradiance",
    ]
    labels_courts = ["VOC", "ISC", "P_MPP", "FF", "η (%)", "Rs", "Rsh", "Irr."]

    scores = []
    for cle in cles_importantes:
        val = float(mesures.get(cle, 0))
        nom = float(NOMINAUX.get(cle, 1))
        scores.append(min(1.0, max(0.0, val / nom)) * 100 if nom != 0 else 50)

    couleurs_bar = [
        "#10b981" if s >= 80 else "#f59e0b" if s >= 50 else "#ef4444"
        for s in scores
    ]

    fig, ax = plt.subplots(figsize=(8, 2.6))
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")

    x = range(len(scores))
    ax.bar(x, scores, color=couleurs_bar, width=0.55, edgecolor="none")
    ax.axhline(y=80, color="#10b981", linestyle="--", linewidth=0.8, alpha=0.6,
               label="Seuil normal (80 %)")
    ax.axhline(y=50, color="#f59e0b", linestyle="--", linewidth=0.8, alpha=0.6,
               label="Seuil alerte (50 %)")

    for i, (xi, s) in enumerate(zip(x, scores)):
        ax.text(xi, s + 2.5, f"{s:.0f}%", ha="center", va="bottom",
                fontsize=7.5, color="#e2e8f0", fontweight="bold")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels_courts, fontsize=8.5, color="#94a3b8")
    ax.set_ylim(0, 125)
    ax.set_ylabel("% du nominal STC", fontsize=7.5, color="#64748b")
    ax.tick_params(axis="y", colors="#64748b", labelsize=7.5)
    ax.legend(loc="upper right", framealpha=0.15, facecolor="#111827",
              edgecolor="#2d3748", labelcolor="#94a3b8", fontsize=7)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(axis="x", length=0)

    plt.tight_layout(pad=0.4)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="#111827")
    plt.close(fig)
    buf.seek(0)
    return buf


def _graphique_feature_importance() -> io.BytesIO:
    """Barres horizontales des 12 features les plus importantes."""
    items = sorted(FEATURE_IMPORTANCE.items(), key=lambda x: x[1], reverse=True)
    labels  = [k for k, _ in items]
    valeurs = [v for _, v in items]

    # Couleur selon le rang
    couleurs = []
    for i, v in enumerate(valeurs):
        if i == 0:
            couleurs.append("#00d4ff")
        elif v >= 5:
            couleurs.append("#8b5cf6")
        else:
            couleurs.append("#2d4a6b")

    fig, ax = plt.subplots(figsize=(7, 3.8))
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")

    bars = ax.barh(labels, valeurs, color=couleurs, height=0.55, edgecolor="none")
    for bar, val in zip(bars, valeurs):
        ax.text(val + 0.4, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", ha="left",
                color="#e2e8f0", fontsize=7.5, fontweight="bold")

    ax.set_xlim(0, 42)
    ax.set_xlabel("Importance relative (%)", color="#64748b", fontsize=8)
    ax.tick_params(colors="#94a3b8", labelsize=7.5)
    for sp in ax.spines.values():
        sp.set_color("#2d3748")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Importance des features — XGBoost (feature_importances_)",
                 color="#64748b", fontsize=7.5, pad=6)

    plt.tight_layout(pad=0.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="#111827")
    plt.close(fig)
    buf.seek(0)
    return buf


def _graphique_rul_courbe(rul_annees: float) -> io.BytesIO:
    """Courbe de dégradation affichant la position RUL sur 25 ans."""
    t = np.linspace(0, 25, 300)
    # Modèle de dégradation polynomial simplifié : η(t) = 1 - 0.005*t - 0.0002*t²
    degradation = 1 - 0.005 * t - 0.0002 * t ** 2
    degradation = np.clip(degradation, 0, 1)

    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")

    ax.fill_between(t, degradation, alpha=0.12, color="#00d4ff")
    ax.plot(t, degradation, color="#00d4ff", linewidth=1.8, label="Courbe de dégradation")

    # Ligne de fin de vie (80 % du nominal)
    ax.axhline(y=0.80, color="#f59e0b", linestyle="--", linewidth=1,
               alpha=0.8, label="Seuil fin de vie (80 %)")
    ax.axhline(y=0.60, color="#ef4444", linestyle="--", linewidth=1,
               alpha=0.8, label="Seuil critique (60 %)")

    # Position actuelle du panneau
    t_consumed = max(0, 25 - rul_annees)
    idx = int(t_consumed / 25 * 299)
    eta_actuelle = float(degradation[min(idx, 299)])
    coul_pt = "#10b981" if rul_annees > 15 else "#f59e0b" if rul_annees > 5 else "#ef4444"
    ax.axvline(x=t_consumed, color=coul_pt, linestyle=":", linewidth=1.5, alpha=0.9)
    ax.scatter([t_consumed], [eta_actuelle], color=coul_pt, s=60, zorder=5)
    ax.text(t_consumed + 0.4, eta_actuelle + 0.02,
            f"Position actuelle\n({t_consumed:.1f} ans)",
            color=coul_pt, fontsize=7, va="bottom")

    # Zone RUL restante
    ax.axvspan(t_consumed, 25, alpha=0.05, color="#10b981", label=f"RUL = {rul_annees:.1f} ans")

    ax.set_xlim(0, 25)
    ax.set_ylim(0.45, 1.08)
    ax.set_xlabel("Années de fonctionnement", color="#64748b", fontsize=8)
    ax.set_ylabel("Efficacité relative", color="#64748b", fontsize=8)
    ax.tick_params(colors="#64748b", labelsize=7.5)
    ax.legend(loc="lower left", framealpha=0.15, facecolor="#111827",
              edgecolor="#2d3748", labelcolor="#94a3b8", fontsize=7)
    for sp in ax.spines.values():
        sp.set_color("#2d3748")

    plt.tight_layout(pad=0.4)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="#111827")
    plt.close(fig)
    buf.seek(0)
    return buf


# ════════════════════════════════════════════════════════════════════
# PAGE 1 — Page de garde
# ════════════════════════════════════════════════════════════════════
def _p1_garde(el: list, diag: dict, s: dict) -> None:
    now    = datetime.now()
    classe = diag.get("classe", 0)
    niveau = diag.get("niveau_alerte", "Normal")
    conf   = diag.get("confiance", 0)
    panne  = diag.get("panne_detectee", "—")

    bandeau = Table([[
        Paragraph("☀  RAPPORT DE DIAGNOSTIC<br/>PANNEAU PHOTOVOLTAÏQUE", s["garde_titre"]),
    ]], colWidths=[W_PAGE])
    bandeau.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, -1), C_BG),
        ("TOPPADDING",     (0, 0), (-1, -1), 24),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 18),
        ("LEFTPADDING",    (0, 0), (-1, -1), 12),
        ("RIGHTPADDING",   (0, 0), (-1, -1), 12),
        ("BOX",            (0, 0), (-1, -1), 2, C_CYAN),
    ]))
    el.append(bandeau)
    el.append(Spacer(1, 8))

    el.append(Paragraph(
        "Système de Diagnostic Intelligent basé sur l'Intelligence Artificielle",
        s["garde_sous"]))
    el.append(Paragraph(
        "XGBoost · Isolation Forest · Régression Polynomiale · ReportLab",
        ParagraphStyle("tech_sub", parent=s["garde_inst"],
                       textColor=C_CYAN, fontSize=8)))
    el.append(Spacer(1, 16))

    # ── KPI principaux ─────────────────────────────────────────────
    coul_cl  = colors.HexColor(COULEURS_CLASSE.get(classe, "#e2e8f0"))
    coul_niv = colors.HexColor(COULEURS_NIVEAU.get(niveau, "#e2e8f0"))

    kpi_data = [
        [Paragraph("DÉFAUT IDENTIFIÉ",   s["kpi_lbl"]),
         Paragraph("CONFIANCE DU MODÈLE", s["kpi_lbl"]),
         Paragraph("NIVEAU D'ALERTE",     s["kpi_lbl"])],
        [Paragraph(panne,
                   ParagraphStyle("pv", parent=s["kpi_val"],
                                  textColor=coul_cl, fontSize=14)),
         Paragraph(f"{conf:.2f} %", s["kpi_val"]),
         Paragraph(niveau,
                   ParagraphStyle("nv", parent=s["kpi_val"],
                                  textColor=coul_niv, fontSize=14))],
        [Paragraph("Classe XGBoost prédite",     s["kpi_sub"]),
         Paragraph("Probabilité classe gagnante", s["kpi_sub"]),
         Paragraph("Seuil d'intervention",        s["kpi_sub"])],
    ]
    kpi_tbl = Table(kpi_data, colWidths=[W_PAGE / 3] * 3)
    kpi_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), C_SURFACE),
        ("BOX",           (0, 0), (-1, -1), 1, C_BORDER),
        ("INNERGRID",     (0, 0), (-1, -1), 0.4, C_BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    el.append(kpi_tbl)
    el.append(Spacer(1, 18))

    # ── Informations institutionnelles ────────────────────────────
    el.append(_hr(C_BORDER, space_before=0, space_after=10))

    info = [
        ["Institution",  "Institut National des Sciences Appliquées et de Technologie"],
        ["",             "INSAT Tunis — Université de Carthage"],
        ["Filière",      "Génie Maintenance & Instrumentation — 3ème année"],
        ["Projet",       "Projet de Fin d'Année — Diagnostic Intelligent Panneaux Solaires"],
        ["Auteur",       "Barhoumi Montassar"],
        ["Année",        "2025 – 2026"],
        ["Généré le",    now.strftime("%d/%m/%Y à %H:%M:%S")],
        ["Référence",    f"SOLARAI-DIAG-{now.strftime('%Y%m%d-%H%M%S')}"],
    ]
    tbl_info = Table(info, colWidths=[3.8*cm, W_PAGE - 3.8*cm])
    tbl_info.setStyle(TableStyle([
        ("TEXTCOLOR",    (0, 0), (0, -1), C_CYAN),
        ("TEXTCOLOR",    (1, 0), (1, -1), C_TEXTE),
        ("FONTNAME",     (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 9),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("LINEBELOW",    (0, 0), (-1, -2), 0.2, colors.HexColor("#1e293b")),
        ("BACKGROUND",   (0, 0), (-1, -1), colors.transparent),
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
    ]))
    el.append(tbl_info)
    el.append(Spacer(1, 14))
    el.append(_hr(C_MUTED))

    # ── Résumé de la structure du rapport ─────────────────────────
    el.append(Paragraph("Structure de ce rapport (7 pages)", s["h4"]))
    pages_info = [
        ["Page 1", "Page de garde — Identification du diagnostic et informations institutionnelles"],
        ["Page 2", "Résumé exécutif — Score de santé, KPI, tableau des 16 paramètres mesurés"],
        ["Page 3", "Méthodes IA — Principes, formules mathématiques, hyperparamètres des 3 algorithmes"],
        ["Page 4", "Résultats IA détaillés — Analyse causale, features dérivées, interprétation"],
        ["Page 5", "Analyse visuelle — Probabilités par classe, paramètres vs nominal, courbe RUL"],
        ["Page 6", "Recommandations — Plan d'action structuré, procédures de maintenance"],
        ["Page 7", "Architecture & métriques — Pipeline complet, performance, référentiel 8 classes"],
    ]
    tbl_pages = Table(pages_info, colWidths=[2.0*cm, W_PAGE - 2.0*cm])
    tbl_pages.setStyle(TableStyle([
        ("TEXTCOLOR",    (0, 0), (0, -1), C_CYAN),
        ("FONTNAME",     (0, 0), (0, -1), "Helvetica-Bold"),
        ("TEXTCOLOR",    (1, 0), (1, -1), C_MUTED),
        ("FONTSIZE",     (0, 0), (-1, -1), 8.5),
        ("TOPPADDING",   (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
        ("LINEBELOW",    (0, 0), (-1, -2), 0.2, colors.HexColor("#1e293b")),
        ("BACKGROUND",   (0, 0), (-1, -1), colors.transparent),
    ]))
    el.append(tbl_pages)
    el.append(Spacer(1, 10))
    el.append(Paragraph(
        "Ce rapport a été généré automatiquement par le pipeline Solar AI Diagnostic. "
        "Il ne remplace pas l'expertise d'un technicien de maintenance qualifié.",
        s["body_muted"]))
    el.append(PageBreak())


# ════════════════════════════════════════════════════════════════════
# PAGE 2 — Résumé exécutif & score de santé
# ════════════════════════════════════════════════════════════════════
def _p2_resume(el: list, diag: dict, mesures: dict, s: dict) -> None:
    el.append(Paragraph("Résumé Exécutif", s["h2"]))
    el.append(Paragraph(
        "Synthèse complète du diagnostic IA effectué sur le panneau photovoltaïque. "
        "Le score de santé global (0–100) est calculé comme le complément du score d'anomalie. "
        "Chaque indicateur est accompagné de son interprétation pour faciliter la décision.",
        s["body"]))
    el.append(Spacer(1, 8))

    score_sante = max(0, min(100, round(100 - diag.get("score_anomalie", 0))))
    classe      = diag.get("classe", 0)
    conf        = diag.get("confiance", 0)
    score_anom  = diag.get("score_anomalie", 0)
    niveau      = diag.get("niveau_alerte", "Normal")
    rul_a       = diag.get("rul_annees",  0)
    rul_h       = diag.get("rul_heures",  0)

    buf_jauge = _graphique_sante(score_sante)
    img_jauge = Image(buf_jauge, width=5.2*cm, height=5.2*cm)

    coul_anom = (C_SUCCES if score_anom < 30
                 else C_ORANGE if score_anom < 60 else C_DANGER)
    coul_rul  = (C_SUCCES if rul_a > 15 else C_ORANGE if rul_a > 5 else C_DANGER)
    coul_cl   = colors.HexColor(COULEURS_CLASSE.get(classe, "#e2e8f0"))
    coul_niv  = colors.HexColor(COULEURS_NIVEAU.get(niveau, "#e2e8f0"))
    coul_conf = C_SUCCES if conf >= 95 else (C_ORANGE if conf >= 70 else C_DANGER)

    kpi2_data = [
        ["Indicateur",         "Valeur",                  "Interprétation"],
        ["Défaut identifié",
         diag.get("panne_detectee","—"),
         f"Classe n°{classe} sur 8 — {DESCRIPTIONS_CLASSES.get(FAULT_LABELS.get(classe,''), '')[:45]}…"],
        ["Confiance XGBoost",
         f"{conf:.2f} %",
         "≥ 95 % = résultat très fiable · 70–95 % = fiable · < 70 % = vérifier"],
        ["Score d'anomalie (IF)",
         f"{score_anom:.1f} / 100",
         "< 30 = normal · 30–59 = attention · 60–79 = alerte · ≥ 80 = critique"],
        ["Niveau d'alerte",
         niveau,
         "Seuil d'intervention basé sur le score d'anomalie"],
        ["RUL estimée",
         f"{rul_a:.2f} ans",
         f"≈ {int(rul_h):,} heures restantes (base nominale 25 ans = 219 000 h)"],
        ["Statut RUL",
         diag.get("rul_statut","—"),
         "Bon > 15 ans · Surveiller 5–15 ans · Critique < 5 ans"],
    ]
    tbl_kpi2 = Table(kpi2_data, colWidths=[3.8*cm, 3.0*cm, 8.4*cm])
    tbl_kpi2.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  C_BG),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  C_CYAN),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0),  8.5),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_SURFACE, C_SURF2]),
        ("TEXTCOLOR",     (0, 1), (0, -1),  C_MUTED),
        ("FONTNAME",      (0, 1), (0, -1),  "Helvetica-Bold"),
        ("TEXTCOLOR",     (1, 1), (1, -1),  C_TEXTE),
        ("FONTNAME",      (1, 1), (1, -1),  "Helvetica-Bold"),
        ("TEXTCOLOR",     (2, 1), (2, -1),  C_MUTED),
        ("TEXTCOLOR",     (1, 1), (1, 1),   coul_cl),
        ("TEXTCOLOR",     (1, 2), (1, 2),   coul_conf),
        ("TEXTCOLOR",     (1, 3), (1, 3),   coul_anom),
        ("TEXTCOLOR",     (1, 4), (1, 4),   coul_niv),
        ("TEXTCOLOR",     (1, 5), (1, 5),   coul_rul),
        ("TEXTCOLOR",     (1, 6), (1, 6),   coul_rul),
        ("FONTSIZE",      (0, 1), (-1, -1), 8.5),
        ("GRID",          (0, 0), (-1, -1), 0.3, C_BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))

    layout = Table([[img_jauge, tbl_kpi2]],
                   colWidths=[5.8*cm, W_PAGE - 5.8*cm])
    layout.setStyle(TableStyle([
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (0, -1), 0),
        ("RIGHTPADDING",(0, 0), (0, -1), 8),
    ]))
    el.append(layout)
    el.append(Spacer(1, 14))

    # ── Tableau des 16 mesures ──────────────────────────────────────
    el.append(Paragraph("Paramètres Mesurés — 16 Features Brutes", s["h3"]))
    el.append(Paragraph(
        "Les 16 paramètres ci-dessous constituent le vecteur d'entrée brut du système IA. "
        "La colonne <b>Écart (%)</b> indique la déviation par rapport aux valeurs nominales "
        "d'un panneau 360 W en conditions STC (Standard Test Conditions : 1 000 W/m², 25 °C, AM 1.5). "
        "Un écart coloré en <b style='color:#ef4444'>rouge</b> signale un paramètre anormal.",
        s["body"]))
    el.append(Spacer(1, 6))

    mes_data = [["Paramètre", "Unité", "Valeur mesurée", "Nominal STC", "Écart (%)"]]
    cles = list(LABELS_MESURES.keys())
    for cle in cles:
        label, unite, vmin, vmax = LABELS_MESURES[cle]
        val  = float(mesures.get(cle, 0))
        nom  = float(NOMINAUX.get(cle, 1))
        ecart = round((val - nom) / nom * 100, 1) if nom != 0 else 0
        mes_data.append([label, unite, f"{val:.3f}", f"{nom}", f"{ecart:+.1f} %"])

    tbl_mes = Table(mes_data, colWidths=[5.8*cm, 1.5*cm, 2.8*cm, 2.8*cm, 2.3*cm])
    style_mes = []
    for i, cle in enumerate(cles, start=1):
        val = float(mesures.get(cle, 0))
        nom = float(NOMINAUX.get(cle, 1))
        ecart = (val - nom) / nom * 100 if nom != 0 else 0
        c = C_SUCCES if abs(ecart) < 15 else (C_ORANGE if abs(ecart) < 40 else C_DANGER)
        style_mes.append(("TEXTCOLOR", (4, i), (4, i), c))
        style_mes.append(("FONTNAME",  (4, i), (4, i), "Helvetica-Bold"))
    tbl_mes.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  C_BG),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  C_CYAN),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0),  8.5),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_SURFACE, C_SURF2]),
        ("TEXTCOLOR",     (0, 1), (0, -1),  C_TEXTE),
        ("TEXTCOLOR",     (1, 1), (1, -1),  C_MUTED),
        ("TEXTCOLOR",     (2, 1), (2, -1),  C_CYAN),
        ("FONTNAME",      (2, 1), (2, -1),  "Helvetica-Bold"),
        ("TEXTCOLOR",     (3, 1), (3, -1),  C_MUTED),
        ("FONTSIZE",      (0, 1), (-1, -1), 8.5),
        ("GRID",          (0, 0), (-1, -1), 0.3, C_BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("ALIGN",         (2, 0), (-1, -1), "RIGHT"),
    ] + style_mes))
    el.append(tbl_mes)
    el.append(PageBreak())


# ════════════════════════════════════════════════════════════════════
# PAGE 3 — Méthodes IA : principes, formules, hyperparamètres
# ════════════════════════════════════════════════════════════════════
def _p3_methodes_ia(el: list, diag: dict, s: dict) -> None:
    el.append(Paragraph("Méthodes d'Intelligence Artificielle — Principes & Formules", s["h2"]))
    el.append(Paragraph(
        "Le pipeline Solar AI Diagnostic associe trois algorithmes complémentaires. "
        "Ils opèrent en séquence sur le même vecteur de features et produisent des résultats "
        "indépendants fusionnés dans le diagnostic global. Chaque algorithme est expliqué "
        "ci-dessous avec son principe mathématique, ses hyperparamètres et son rôle.",
        s["body"]))
    el.append(Spacer(1, 6))

    # ════ ALGORITHME 1 — XGBoost ════════════════════════════════════
    el.append(Paragraph("① XGBoost — Classifieur de Pannes (8 classes)", s["h3"]))

    xgb_contenu = [
        Paragraph(
            "<b>Principe — Gradient Boosting :</b> XGBoost construit un ensemble de K arbres "
            "de décision en additivité : ŷ = Σ f_k(x), où chaque f_k corrige les résidus de "
            "l'itération précédente. L'objectif minimisé est :",
            s["body"]),
        Paragraph(
            "L(Θ) = Σᵢ l(yᵢ, ŷᵢ) + Σ_k Ω(f_k)",
            s["formule"]),
        Paragraph(
            "avec l = entropie croisée multiclasse et Ω(f) = γT + ½λ‖w‖² (régularisation). "
            "T = nombre de feuilles, w = poids des feuilles, γ et λ = paramètres de pénalité.",
            s["body"]),
        Paragraph(
            "<b>Pourquoi XGBoost pour ce problème ?</b> Les 8 classes de pannes "
            "photovoltaïques correspondent à des profils électriques distincts et bien "
            "séparables dans l'espace des features. XGBoost excelle sur des données tabulaires "
            "avec des frontières de décision non linéaires et gère nativement les features "
            "redondantes via la régularisation L1/L2.",
            s["body"]),
        Paragraph(
            "<b>Hyperparamètres utilisés :</b>",
            s["body"]),
    ]
    el.append(_encadre(xgb_contenu, bg=colors.HexColor("#0d1f2d"), border=C_CYAN))
    el.append(Spacer(1, 4))

    hyp_xgb = [
        ["Hyperparamètre",      "Valeur",     "Rôle"],
        ["n_estimators",        "300",        "Nombre d'arbres dans l'ensemble (profondeur du boosting)"],
        ["max_depth",           "6",          "Profondeur maximale — évite l'overfitting sur features bruit"],
        ["learning_rate (η)",   "0.1",        "Taux d'apprentissage — contrôle la contribution de chaque arbre"],
        ["subsample",           "1.0",        "Fraction d'échantillons par arbre (valeur par défaut XGBoost)"],
        ["colsample_bytree",    "1.0",        "Fraction de features par arbre (valeur par défaut XGBoost)"],
        ["reg_lambda (L2)",     "1.0",        "Régularisation ridge sur les poids des feuilles (défaut)"],
        ["reg_alpha (L1)",      "0.0",        "Régularisation lasso (non activée — défaut)"],
        ["objective",           "multi:softprob", "Sortie : probabilités pour chacune des 8 classes"],
        ["eval_metric",         "mlogloss",   "Perte logarithmique multiclasse sur le jeu de validation"],
    ]
    tbl_hyp = Table(hyp_xgb, colWidths=[3.5*cm, 2.5*cm, W_PAGE - 6*cm])
    tbl_hyp.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  C_BG),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  C_CYAN),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0),  8),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_SURFACE, C_SURF2]),
        ("TEXTCOLOR",     (0, 1), (0, -1),  C_CYAN),
        ("FONTNAME",      (0, 1), (0, -1),  "Courier"),
        ("TEXTCOLOR",     (1, 1), (1, -1),  C_ORANGE),
        ("FONTNAME",      (1, 1), (1, -1),  "Helvetica-Bold"),
        ("TEXTCOLOR",     (2, 1), (2, -1),  C_MUTED),
        ("FONTSIZE",      (0, 1), (-1, -1), 8.5),
        ("GRID",          (0, 0), (-1, -1), 0.3, C_BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]))
    el.append(tbl_hyp)
    el.append(Spacer(1, 10))

    # ════ ALGORITHME 2 — Isolation Forest ═══════════════════════════
    el.append(Paragraph("② Isolation Forest — Détecteur d'Anomalies (non supervisé)", s["h3"]))

    ifor_contenu = [
        Paragraph(
            "<b>Principe — Isolation par partitions aléatoires :</b> L'algorithme construit "
            "n_estimators arbres en sélectionnant aléatoirement une feature puis un seuil "
            "de coupure dans son intervalle de valeurs. La profondeur d'isolement h(x) d'un "
            "point x est le nombre de coupures nécessaires pour l'isoler dans un seul nœud.",
            s["body"]),
        Paragraph(
            "score(x, n) = 2^[ −E(h(x)) / c(n) ]   avec   c(n) = 2H(n−1) − 2(n−1)/n",
            s["formule"]),
        Paragraph(
            "H(·) = nombre harmonique, n = taille du sous-échantillon. "
            "score ≈ 1 → anomalie forte · score ≈ 0.5 → ambigu · score < 0.5 → normal. "
            "Le score brut est converti en score lisible 0–100 par transformation linéaire.",
            s["body"]),
        Paragraph(
            "<b>Avantage clé pour le PV :</b> Aucune donnée labellisée de panne n'est nécessaire. "
            "L'algorithme apprend uniquement la distribution normale et détecte toute "
            "déviation, y compris des défauts inconnus non couverts par les 8 classes.",
            s["body"]),
        Paragraph(
            "<b>Seuils d'alerte :</b> Score 0–29 = Normal · 30–59 = Attention · "
            "60–79 = Alerte · 80–100 = Critique.",
            s["body"]),
    ]
    el.append(_encadre(ifor_contenu, bg=colors.HexColor("#1a1020"), border=C_VIOLET))
    el.append(Spacer(1, 4))

    hyp_if = [
        ["Hyperparamètre",   "Valeur",  "Rôle"],
        ["n_estimators",     "100",     "Nombre d'arbres d'isolation — trade-off vitesse/précision"],
        ["max_samples",      "\"auto\"","Sous-échantillon de 256 points par arbre (défaut sklearn)"],
        ["contamination",    "0.1",     "Fraction attendue d'anomalies dans les données d'entraînement (10 %)"],
        ["max_features",     "1.0",     "Fraction de features utilisées pour la sélection de coupures"],
        ["random_state",     "42",      "Seed pour la reproductibilité des expériences"],
    ]
    tbl_if = Table(hyp_if, colWidths=[3.5*cm, 2.5*cm, W_PAGE - 6*cm])
    tbl_if.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  C_BG),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  C_CYAN),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0),  8),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_SURFACE, C_SURF2]),
        ("TEXTCOLOR",     (0, 1), (0, -1),  C_VIOLET),
        ("FONTNAME",      (0, 1), (0, -1),  "Courier"),
        ("TEXTCOLOR",     (1, 1), (1, -1),  C_ORANGE),
        ("FONTNAME",      (1, 1), (1, -1),  "Helvetica-Bold"),
        ("TEXTCOLOR",     (2, 1), (2, -1),  C_MUTED),
        ("FONTSIZE",      (0, 1), (-1, -1), 8.5),
        ("GRID",          (0, 0), (-1, -1), 0.3, C_BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]))
    el.append(tbl_if)
    el.append(Spacer(1, 10))

    # ════ ALGORITHME 3 — Régression Polynomiale ═════════════════════
    el.append(Paragraph("③ Régression Polynomiale — Prédiction RUL (Durée de Vie Restante)", s["h3"]))

    rul_contenu = [
        Paragraph(
            "<b>Principe — Pipeline Scikit-learn :</b> Le prédicteur RUL enchaîne deux étapes : "
            "(1) PolynomialFeatures(degree=3) qui étend le vecteur d'entrée x de dimension d "
            "en tous les monômes jusqu'au degré 3 — dimension résultante C(d+3,3) ; "
            "(2) Ridge regression qui ajuste les coefficients β sur les données de dégradation :",
            s["body"]),
        Paragraph(
            "RUL(x) = β₀ + β₁x₁ + β₂x₂ + … + βₖxₖᵖᵒˡʸ     minimisant  ‖Xβ − y‖² + α‖β‖²",
            s["formule"]),
        Paragraph(
            "<b>Features d'entrée (4 variables physiques) :</b> efficacité de conversion (%), "
            "résistance série Rs (Ω), Fill Factor (%), temps de fonctionnement accumulé (h). "
            "Ce sous-ensemble capture l'essentiel de la courbe de dégradation électrique.",
            s["body"]),
        Paragraph(
            "<b>Entraînement :</b> Le modèle est entraîné uniquement sur les classes "
            "<b>Normal</b> et <b>Vieillissement accéléré</b> (les seules avec une dynamique "
            "temporelle continue). Les étiquettes RUL sont dérivées du temps de fonctionnement "
            "et de l'écart d'efficacité par rapport au nominal (η_nominal = 19.5 %).",
            s["body"]),
        Paragraph(
            "<b>Base de calcul :</b> Durée nominale = 25 ans = 219 000 h. "
            "Seuil fin de vie conventionnel = 80 % de l'efficacité nominale.",
            s["body"]),
    ]
    el.append(_encadre(rul_contenu, bg=colors.HexColor("#0d2010"), border=C_SUCCES))
    el.append(PageBreak())


# ════════════════════════════════════════════════════════════════════
# PAGE 4 — Résultats IA détaillés + features dérivées + analyse causale
# ════════════════════════════════════════════════════════════════════
def _p4_resultats_detailles(el: list, diag: dict, mesures: dict, s: dict) -> None:
    el.append(Paragraph("Résultats IA Détaillés & Analyse Causale", s["h2"]))
    el.append(Paragraph(
        "Cette page présente les sorties brutes de chaque algorithme, les 6 features "
        "physiques dérivées calculées à partir des mesures brutes, et une analyse causale "
        "identifiant les paramètres responsables du diagnostic.",
        s["body"]))
    el.append(Spacer(1, 6))

    classe = diag.get("classe", 0)
    conf   = diag.get("confiance", 0)
    panne  = diag.get("panne_detectee", "—")
    score_anom = diag.get("score_anomalie", 0)
    rul_a  = diag.get("rul_annees", 0)
    rul_h  = diag.get("rul_heures", 0)
    niveau = diag.get("niveau_alerte", "Normal")

    # ── Features dérivées ──────────────────────────────────────────
    el.append(Paragraph("Features Physiques Dérivées (6 variables calculées)", s["h3"]))
    el.append(Paragraph(
        "En complément des 16 mesures brutes, 6 features sont calculées par ingénierie "
        "des caractéristiques (feature engineering). Ces variables synthétiques encodent "
        "des grandeurs physiques de haut niveau non directement mesurables et constituent "
        "les descripteurs les plus discriminants pour la classification.",
        s["body"]))
    el.append(Spacer(1, 5))

    irr   = float(mesures.get("irradiance", 0))
    puiss = float(mesures.get("puissance_mpp", 0))
    t_pan = float(mesures.get("temperature_panneau", 0))
    t_amb = float(mesures.get("temperature_ambiante", 0))
    i_mpp = float(mesures.get("courant_mpp", 0))
    i_sc  = float(mesures.get("courant_isc", 1e-9))
    v_mpp = float(mesures.get("tension_mpp", 0))
    v_oc  = float(mesures.get("tension_voc", 1e-9))
    rs    = float(mesures.get("resistance_serie", 0))
    duree = float(mesures.get("temps_fonctionnement", 0))
    eff   = float(mesures.get("efficacite", 0))
    ETA_NOM = 19.5
    SURF = 1.87

    pr   = min(1.0, puiss / (irr * SURF)) if irr > 0 else 0.0
    dt   = t_pan - t_amb
    ri   = min(1.0, i_mpp / i_sc) if i_sc > 0 else 0.0
    rv   = min(1.0, v_mpp / v_oc) if v_oc > 0 else 0.0
    rsn  = rs / 5.0
    sv   = (duree / 50_000) * (1 - eff / ETA_NOM)

    feat_data = [
        ["Feature dérivée",     "Formule",                          "Valeur calculée", "Importance", "Interprétation"],
        ["ratio_vmpp_voc",      "Vmpp / Voc",                       f"{rv:.4f}",        "33.6 %",
         "Qualité IV côté tension. Normal ≈ 0.86. ↓ = court-circuit ou PID"],
        ["fill_factor",         "(Vmpp × Impp) / (Voc × Isc)",      f"{float(mesures.get('fill_factor',0)):.2f} %",
         "14.2 %",
         "Rectangularité courbe IV. Normal > 70 %. ↓ = perte série/shunt"],
        ["ratio_impp_isc",      "Impp / Isc",                       f"{ri:.4f}",        "9.1 %",
         "Qualité IV côté courant. Normal ≈ 0.93. ↓ = ombrage ou PID"],
        ["performance_ratio",   "Pmpp / (Irr × Surface)",           f"{pr:.4f}",        "3.6 %",
         "Rendement réel vs théorique. Normal ≈ 0.75–0.85"],
        ["delta_temperature",   "T_cellule − T_ambiante",           f"{dt:.1f} °C",     "0.8 %",
         "Stress thermique. Normal 25–35 °C. ↑↑ = point chaud possible"],
        ["rs_normalise",        "Rs / 5.0",                         f"{rsn:.4f}",       "—",
         "Rs adimensionnelle. > 0.2 = défaut connexion suspect"],
        ["score_vieillissement","(t / 50 000) × (1 − η / η₀)",     f"{sv:.4f}",        "4.1 %",
         "Indice composite âge + dégradation. ↑ = vieillissement avancé"],
    ]

    tbl_feat = Table(feat_data, colWidths=[3.2*cm, 4.0*cm, 2.4*cm, 1.8*cm, W_PAGE - 11.4*cm])
    tbl_feat.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  C_BG),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  C_CYAN),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0),  8),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_SURFACE, C_SURF2]),
        ("TEXTCOLOR",     (0, 1), (0, -1),  C_CYAN),
        ("FONTNAME",      (0, 1), (0, -1),  "Courier"),
        ("TEXTCOLOR",     (1, 1), (1, -1),  C_MUTED),
        ("FONTNAME",      (1, 1), (1, -1),  "Courier"),
        ("TEXTCOLOR",     (2, 1), (2, -1),  C_ORANGE),
        ("FONTNAME",      (2, 1), (2, -1),  "Helvetica-Bold"),
        ("TEXTCOLOR",     (3, 1), (3, -1),  C_VIOLET),
        ("FONTNAME",      (3, 1), (3, -1),  "Helvetica-Bold"),
        ("TEXTCOLOR",     (4, 1), (4, -1),  C_MUTED),
        ("FONTSIZE",      (0, 1), (-1, -1), 8),
        ("GRID",          (0, 0), (-1, -1), 0.3, C_BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]))
    el.append(tbl_feat)
    el.append(Spacer(1, 10))

    # ── Analyse causale : paramètres les plus déviants ──────────────
    el.append(Paragraph("Analyse Causale — Paramètres Responsables du Diagnostic", s["h3"]))
    el.append(Paragraph(
        "Le tableau ci-dessous identifie les 8 paramètres qui s'écartent le plus des valeurs "
        "nominales STC. Ces écarts sont les principaux signaux utilisés par le modèle XGBoost "
        "pour attribuer la classe <b>" + panne + "</b>. "
        "Un écart fortement négatif sur VOC par exemple oriente vers un court-circuit ; "
        "une chute du Fill Factor pointe vers un vieillissement ou un défaut connexion.",
        s["body"]))
    el.append(Spacer(1, 5))

    # Calcul des écarts et tri par valeur absolue
    ecarts = []
    for cle, (label, unite, _, _) in LABELS_MESURES.items():
        val = float(mesures.get(cle, 0))
        nom = float(NOMINAUX.get(cle, 1))
        if nom != 0:
            pct = (val - nom) / nom * 100
            ecarts.append((abs(pct), pct, label, unite, val, nom, cle))
    ecarts.sort(reverse=True)

    causal_data = [["Paramètre",  "Valeur", "Nominal",  "Écart (%)", "Signification diagnostique"]]
    INTERPRETATIONS = {
        "irradiance":           ("Irradiance réduite", "Encrassement ou ombrage probables"),
        "tension_voc":          ("VOC dévié", "Court-circuit interne si VOC << nominal"),
        "courant_isc":          ("ISC dévié", "Ombrage partiel si ISC << nominal"),
        "puissance_mpp":        ("Puissance réduite", "Perte globale de rendement"),
        "tension_mpp":          ("Vmpp dévié", "Décalage du point de puissance max"),
        "courant_mpp":          ("Impp dévié", "Décalage du point de puissance max"),
        "resistance_serie":     ("Rs élevée", "Défaut connexion ou oxydation"),
        "resistance_shunt":     ("Rsh faible", "Court-circuit partiel ou PID"),
        "fill_factor":          ("FF réduit", "Dégradation courbe IV — multiples causes"),
        "efficacite":           ("η réduite", "Perte globale — vieillissement ou défaut"),
        "facteur_idealite":     ("n dévié", "Modifications jonction p-n — PID ou vieillissement"),
        "temps_fonctionnement": ("Durée élevée", "Vieillissement naturel attendu"),
        "humidite":             ("Humidité élevée", "Risque PID, corrosion, encapsulant"),
        "vitesse_vent":         ("Vent fort", "Stress mécanique, refroidissement accru"),
        "temperature_panneau":  ("T élevée", "Perte de rendement +0.4%/°C, hot-spot possible"),
        "temperature_ambiante": ("T ambiante élevée", "Facteur environnemental — surveillance"),
    }
    for _, pct, label, unite, val, nom, cle in ecarts[:8]:
        c_ecart = C_SUCCES if abs(pct) < 15 else (C_ORANGE if abs(pct) < 40 else C_DANGER)
        titre_interp, detail_interp = INTERPRETATIONS.get(cle, ("—", "—"))
        causal_data.append([
            label, f"{val:.3f} {unite}", f"{nom} {unite}",
            f"{pct:+.1f} %", detail_interp
        ])

    tbl_causal = Table(causal_data, colWidths=[3.5*cm, 2.3*cm, 2.3*cm, 1.9*cm, W_PAGE - 10*cm])
    style_causal = []
    for i, (_, pct, *_) in enumerate(ecarts[:8], start=1):
        c = C_SUCCES if abs(pct) < 15 else (C_ORANGE if abs(pct) < 40 else C_DANGER)
        style_causal += [
            ("TEXTCOLOR",  (3, i), (3, i), c),
            ("FONTNAME",   (3, i), (3, i), "Helvetica-Bold"),
        ]
    tbl_causal.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  C_BG),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  C_CYAN),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0),  8),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_SURFACE, C_SURF2]),
        ("TEXTCOLOR",     (0, 1), (0, -1),  C_TEXTE),
        ("TEXTCOLOR",     (1, 1), (1, -1),  C_CYAN),
        ("FONTNAME",      (1, 1), (1, -1),  "Helvetica-Bold"),
        ("TEXTCOLOR",     (2, 1), (2, -1),  C_MUTED),
        ("TEXTCOLOR",     (4, 1), (4, -1),  C_MUTED),
        ("FONTSIZE",      (0, 1), (-1, -1), 8.5),
        ("GRID",          (0, 0), (-1, -1), 0.3, C_BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ] + style_causal))
    el.append(tbl_causal)
    el.append(PageBreak())


# ════════════════════════════════════════════════════════════════════
# PAGE 5 — Analyse visuelle : graphiques
# ════════════════════════════════════════════════════════════════════
def _p5_analyse_visuelle(el: list, diag: dict, mesures: dict, s: dict) -> None:
    el.append(Paragraph("Analyse Visuelle des Résultats", s["h2"]))

    # ── Graphique probabilités ─────────────────────────────────────
    el.append(Paragraph("① Distribution des Probabilités XGBoost par Classe de Panne", s["h3"]))
    el.append(Paragraph(
        "Le modèle XGBoost attribue une probabilité à chacune des 8 classes via la fonction "
        "softmax (normalisation exponentielle). La somme des probabilités = 100 %. "
        "La barre <b style='color:#00d4ff'>cyan</b> = classe prédite · "
        "<b style='color:#f59e0b'>orange</b> = classe candidate secondaire (prob. ≥ 10 %).",
        s["body"]))
    el.append(Spacer(1, 4))

    probas = diag.get("probabilites", {})
    panne  = diag.get("panne_detectee", "")
    if probas:
        buf_proba = _graphique_probas(probas, panne)
        el.append(Image(buf_proba, width=W_PAGE, height=7.5*cm))
        el.append(Paragraph(
            "Figure 1 — Distribution des probabilités XGBoost. Classe prédite : " + panne +
            f" ({diag.get('confiance',0):.2f} %).",
            s["caption"]))
    el.append(Spacer(1, 8))

    # ── Graphique paramètres normalisés ────────────────────────────
    el.append(Paragraph("② Paramètres Mesurés vs Valeurs Nominales STC", s["h3"]))
    el.append(Paragraph(
        "Chaque barre = valeur mesurée / valeur nominale × 100. "
        "<b style='color:#10b981'>Vert (≥ 80 %)</b> = dans la norme · "
        "<b style='color:#f59e0b'>Orange (50–80 %)</b> = déviation modérée · "
        "<b style='color:#ef4444'>Rouge (< 50 %)</b> = anomalie significative.",
        s["body"]))
    el.append(Spacer(1, 4))

    buf_params = _graphique_parametres(mesures)
    el.append(Image(buf_params, width=W_PAGE, height=4.2*cm))
    el.append(Paragraph(
        "Figure 2 — Paramètres normalisés (% du nominal STC). Panneau référence : "
        "360 W, η = 19.5 %, FF = 76 %, Voc = 41.5 V, Isc = 10.2 A.",
        s["caption"]))
    el.append(Spacer(1, 8))

    # ── Courbe de dégradation RUL ──────────────────────────────────
    el.append(Paragraph("③ Courbe de Dégradation & Position RUL sur 25 ans", s["h3"]))
    el.append(Paragraph(
        "La courbe modélise l'évolution de l'efficacité relative sur la durée de vie nominale "
        "(25 ans). Le point coloré indique la position actuelle du panneau. "
        "La zone verte représente la durée de vie restante estimée (RUL).",
        s["body"]))
    el.append(Spacer(1, 4))

    rul_a = diag.get("rul_annees", 0)
    buf_rul = _graphique_rul_courbe(rul_a)
    el.append(Image(buf_rul, width=W_PAGE * 0.8, height=4.5*cm))
    el.append(Paragraph(
        f"Figure 3 — Courbe de dégradation estimée. RUL = {rul_a:.2f} ans "
        f"({int(diag.get('rul_heures',0)):,} h restantes).",
        s["caption"]))
    el.append(Spacer(1, 8))

    # ── Tableau probabilités détaillé ──────────────────────────────
    el.append(Paragraph("④ Tableau Complet des Probabilités par Classe", s["h3"]))
    el.append(Spacer(1, 4))

    proba_data = [["Classe", "Proba. (%)", "Statut", "Description physique"]]
    for nom, val in sorted(probas.items(), key=lambda x: x[1], reverse=True):
        est_predite = nom == panne
        statut_txt  = "▶ PRÉDITE" if est_predite else ("Secondaire" if val >= 10 else "—")
        desc_courte = DESCRIPTIONS_CLASSES.get(nom, "")[:55] + "…"
        proba_data.append([nom, f"{val:.2f} %", statut_txt, desc_courte])

    tbl_p = Table(proba_data, colWidths=[4*cm, 2.2*cm, 2.8*cm, W_PAGE - 9*cm])
    style_p = []
    for i, (nom, val) in enumerate(
            sorted(probas.items(), key=lambda x: x[1], reverse=True), start=1):
        if nom == panne:
            style_p += [
                ("BACKGROUND", (0, i), (-1, i), colors.HexColor("#0a2030")),
                ("TEXTCOLOR",  (0, i), (-1, i), C_CYAN),
                ("FONTNAME",   (0, i), (-1, i), "Helvetica-Bold"),
            ]
        elif val >= 10:
            style_p.append(("TEXTCOLOR", (1, i), (1, i), C_ORANGE))

    tbl_p.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  C_BG),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  C_CYAN),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0),  8.5),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_SURFACE, C_SURF2]),
        ("TEXTCOLOR",     (0, 1), (-1, -1), C_TEXTE),
        ("TEXTCOLOR",     (3, 1), (3, -1),  C_MUTED),
        ("FONTSIZE",      (0, 1), (-1, -1), 8.5),
        ("GRID",          (0, 0), (-1, -1), 0.3, C_BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 7),
        ("ALIGN",         (1, 0), (2, -1),  "CENTER"),
    ] + style_p))
    el.append(tbl_p)
    el.append(PageBreak())


# ════════════════════════════════════════════════════════════════════
# PAGE 6 — Recommandations & plan d'action
# ════════════════════════════════════════════════════════════════════
def _p6_recommandations(el: list, diag: dict, s: dict) -> None:
    el.append(Paragraph("Recommandations Techniques & Plan d'Action", s["h2"]))

    classe = diag.get("classe", 0)
    niveau = diag.get("niveau_alerte", "Normal")
    panne  = diag.get("panne_detectee", "—")
    recs   = diag.get("recommendations", [])

    # ── Description physique complète de la panne ──────────────────
    el.append(Paragraph(f"Analyse de la panne : {panne}", s["h3"]))

    desc_data = [
        [Paragraph("<b>Description courte</b>", s["body_muted"]),
         Paragraph(DESCRIPTIONS_CLASSES.get(panne, "—"), s["body"])],
        [Paragraph("<b>Mécanisme physique</b>", s["body_muted"]),
         Paragraph(DESCRIPTIONS_PHYSIQUES.get(panne, "—"), s["body"])],
    ]
    tbl_desc = Table(desc_data, colWidths=[3.5*cm, W_PAGE - 3.5*cm])
    tbl_desc.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), C_SURFACE),
        ("BOX",           (0, 0), (-1, -1), 1, C_BORDER),
        ("INNERGRID",     (0, 0), (-1, -1), 0.3, C_BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]))
    el.append(tbl_desc)
    el.append(Spacer(1, 10))

    # ── Recommandations ────────────────────────────────────────────
    el.append(Paragraph("Recommandations Générées par le Système IA", s["h3"]))
    el.append(Paragraph(
        "Les recommandations ci-dessous sont générées en fonction de la classe de panne "
        "identifiée (<b>" + panne + "</b>) et du niveau d'alerte (<b>" + niveau + "</b>). "
        "Les actions de priorité <b style='color:#ef4444'>haute ⚠</b> doivent être traitées "
        "en premier pour prévenir une dégradation irréversible.",
        s["body"]))
    el.append(Spacer(1, 6))

    if recs:
        rec_data = [["Priorité", "Recommandation technique"]]
        for i, rec in enumerate(recs, 1):
            prio_label = "⚠ HAUTE"   if i <= 2 else ("→ MOYENNE" if i <= 4 else "○ FAIBLE")
            rec_data.append([prio_label, rec])

        tbl_rec = Table(rec_data, colWidths=[2.6*cm, W_PAGE - 2.6*cm])
        style_rec = []
        for i in range(1, len(recs) + 1):
            c = C_DANGER if i <= 2 else (C_ORANGE if i <= 4 else C_SUCCES)
            style_rec += [
                ("TEXTCOLOR", (0, i), (0, i), c),
                ("FONTNAME",  (0, i), (0, i), "Helvetica-Bold"),
            ]
        tbl_rec.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0),  C_BG),
            ("TEXTCOLOR",     (0, 0), (-1, 0),  C_CYAN),
            ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, 0),  8.5),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_SURFACE, C_SURF2]),
            ("TEXTCOLOR",     (1, 1), (1, -1),  C_TEXTE),
            ("FONTSIZE",      (0, 1), (-1, -1), 9.5),
            ("GRID",          (0, 0), (-1, -1), 0.3, C_BORDER),
            ("TOPPADDING",    (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ] + style_rec))
        el.append(tbl_rec)
    else:
        el.append(Paragraph("Aucune recommandation spécifique générée.", s["body_muted"]))

    el.append(Spacer(1, 12))

    # ── Plan d'action par horizon temporel ────────────────────────
    el.append(Paragraph("Plan d'Action Structuré par Horizon Temporel", s["h3"]))
    el.append(Paragraph(
        "Adapté au niveau d'alerte <b>" + niveau + "</b> et à la classe <b>" + panne + "</b>.",
        s["body"]))
    el.append(Spacer(1, 6))

    if classe == 0:
        actions = [
            ("Immédiat",    "Normal",    "Aucune intervention corrective requise. Continuer la surveillance standard."),
            ("Court terme", "Préventif", "Nettoyage préventif de la surface vitrée (trimestriel recommandé). Inspection visuelle."),
            ("Moyen terme", "Préventif", "Contrôle des connexions, mesure Riso (résistance d'isolement), test boîte de jonction."),
            ("Long terme",  "Planifié",  "Suivi de la dégradation naturelle. Prévoir le remplacement selon l'estimation RUL."),
        ]
    elif niveau == "Critique":
        actions = [
            ("Immédiat",    "CRITIQUE", f"MISE HORS SERVICE immédiate — défaut {panne} critique détecté par le système IA."),
            ("< 24 h",      "Urgent",   "Inspection physique complète sur site par technicien habilité. Mesures électriques (IV curve tracer)."),
            ("< 72 h",      "Urgent",   "Remplacement ou réparation selon diagnostic terrain. Tests de sécurité IEC 62446."),
            ("Long terme",  "Planifié", "Révision du plan de maintenance préventive. Analyse des causes racines. Traçabilité GMAO."),
        ]
    elif niveau == "Alerte":
        actions = [
            ("Immédiat",    "Alerte",    "Surveillance renforcée. Doublement de la fréquence de relevé des métriques (horaire)."),
            ("< 72 h",      "Urgent",    f"Planification d'une intervention corrective pour {panne}. Commander les pièces si besoin."),
            ("Moyen terme", "Préventif", "Vérification complète : câblage, connecteurs MC4, boîte de jonction, onduleur."),
            ("Long terme",  "Planifié",  "Mise à jour du calendrier de maintenance. Suivi évolution du défaut dans le système GMAO."),
        ]
    else:
        actions = [
            ("Immédiat",    "Attention", "Enregistrement de l'alerte. Surveillance quotidienne des métriques sur le tableau de bord."),
            ("Court terme", "Préventif", f"Inspection visuelle et mesures complémentaires — confirmer {panne} sur site."),
            ("Moyen terme", "Préventif", "Maintenance préventive programmée dans le prochain cycle d'entretien périodique."),
            ("Long terme",  "Planifié",  "Traçabilité dans le système de gestion de maintenance (GMAO). Analyse tendance."),
        ]

    coul_statuts = {
        "CRITIQUE": C_DANGER, "Urgent": C_DANGER, "Alerte": C_ORANGE,
        "Attention": C_ORANGE, "Préventif": C_SUCCES, "Normal": C_SUCCES,
        "Planifié": C_MUTED,
    }

    act_data = [["Horizon", "Niveau", "Action recommandée"]]
    for h, niv_act, act in actions:
        act_data.append([h, niv_act, act])

    tbl_act = Table(act_data, colWidths=[2.4*cm, 2.4*cm, W_PAGE - 4.8*cm])
    style_act = []
    for i, (_, niv_act, _) in enumerate(actions, start=1):
        c = coul_statuts.get(niv_act, C_TEXTE)
        style_act += [
            ("TEXTCOLOR", (1, i), (1, i), c),
            ("FONTNAME",  (1, i), (1, i), "Helvetica-Bold"),
        ]
    tbl_act.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  C_BG),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  C_CYAN),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0),  8.5),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_SURFACE, C_SURF2]),
        ("TEXTCOLOR",     (0, 1), (0, -1),  C_CYAN),
        ("FONTNAME",      (0, 1), (0, -1),  "Helvetica-Bold"),
        ("TEXTCOLOR",     (2, 1), (2, -1),  C_TEXTE),
        ("FONTSIZE",      (0, 1), (-1, -1), 9.5),
        ("GRID",          (0, 0), (-1, -1), 0.3, C_BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ] + style_act))
    el.append(tbl_act)
    el.append(PageBreak())


# ════════════════════════════════════════════════════════════════════
# PAGE 7 — Architecture, métriques, référentiel 8 classes
# ════════════════════════════════════════════════════════════════════
def _p7_technique(el: list, diag: dict, s: dict) -> None:
    el.append(Paragraph("Architecture Système, Métriques & Référentiel de Classes", s["h2"]))
    el.append(Paragraph(
        "Cette page documente l'architecture complète du pipeline Solar AI Diagnostic, "
        "les performances mesurées sur le jeu de test, le dataset d'entraînement "
        "et le référentiel complet des 8 classes de pannes photovoltaïques.",
        s["body"]))
    el.append(Spacer(1, 8))

    # ── Architecture pipeline ─────────────────────────────────────
    el.append(Paragraph("Architecture du Pipeline IA (5 étapes)", s["h3"]))
    arch_data = [
        ["Étape",       "Composant",            "Algorithme / Technologie",                     "Sortie produite"],
        ["① Features",  "FeatureEngineering",   "16 brutes → +6 dérivées (formules physiques)", "Vecteur 22 dim."],
        ["② Classif.",  "FaultClassifier",      "XGBoost — n_est=300, depth=6, multi:softprob", "Classe 0–7 + probas"],
        ["③ Anomalie",  "AnomalyDetector",      "Isolation Forest — n_est=200, cont.=0.02",     "Score 0–100"],
        ["④ RUL",       "LifetimePredictor",    "Ridge + PolynomialFeatures(deg=3)",             "RUL (h et années)"],
        ["⑤ Rapport",   "ReportGenerator",      "ReportLab 4.2 + Matplotlib 3.9",               "PDF 7 pages"],
    ]
    tbl_arch = Table(arch_data, colWidths=[2.1*cm, 3.2*cm, 6.5*cm, W_PAGE - 11.8*cm])
    tbl_arch.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  C_BG),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  C_CYAN),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0),  8.5),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_SURFACE, C_SURF2]),
        ("TEXTCOLOR",     (0, 1), (0, -1),  C_CYAN),
        ("FONTNAME",      (0, 1), (0, -1),  "Helvetica-Bold"),
        ("TEXTCOLOR",     (1, 1), (1, -1),  C_TEXTE),
        ("FONTNAME",      (1, 1), (1, -1),  "Helvetica-Bold"),
        ("TEXTCOLOR",     (2, 1), (2, -1),  C_MUTED),
        ("FONTNAME",      (2, 1), (2, -1),  "Courier"),
        ("TEXTCOLOR",     (3, 1), (3, -1),  C_SUCCES),
        ("FONTSIZE",      (0, 1), (-1, -1), 8.5),
        ("GRID",          (0, 0), (-1, -1), 0.3, C_BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]))
    el.append(tbl_arch)
    el.append(Spacer(1, 10))

    # ── Métriques de performance ──────────────────────────────────
    el.append(Paragraph("Métriques de Performance — XGBoost sur Jeu de Test Isolé", s["h3"]))
    el.append(Paragraph(
        "Toutes les métriques sont calculées sur le jeu de <b>test isolé</b> "
        "(1 200 exemples, non vus à l'entraînement). "
        "Dataset total : <b>6 000 échantillons synthétiques</b> "
        "(750 par classe × 8 classes), split 80/20, random_state=42.",
        s["body"]))
    el.append(Spacer(1, 6))

    met_data = [
        ["Métrique",              "Valeur globale", "Signification"],
        ["Accuracy",              "99.92 %",
         "Fraction de diagnostics corrects sur 1 200 exemples de test"],
        ["F1-Score Macro",        "99.92 %",
         "Moyenne non pondérée du F1 sur 8 classes — équitable entre classes"],
        ["F1-Score Pondéré",      "99.92 %",
         "Pondéré par le support de chaque classe (150 exemples chacune)"],
        ["Précision moyenne",     "99.92 %",
         "Vrais positifs / (VP + FP) — mesure la qualité des prédictions positives"],
        ["Rappel moyen",          "99.92 %",
         "Vrais positifs / (VP + FN) — mesure la détection des cas réels"],
        ["Temps de réponse API",  "< 200 ms",
         "Diagnostic complet (feature engineering + 3 modèles) en local"],
        ["Jeu d'entraînement",    "4 800",
         "80 % du dataset · 600 exemples × 8 classes · random_state=42"],
        ["Jeu de test",           "1 200",
         "20 % du dataset · 150 exemples × 8 classes · jamais vus"],
    ]
    tbl_met = Table(met_data, colWidths=[4.0*cm, 2.8*cm, W_PAGE - 6.8*cm])
    tbl_met.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  C_BG),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  C_CYAN),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0),  8.5),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_SURFACE, C_SURF2]),
        ("TEXTCOLOR",     (0, 1), (0, -1),  C_MUTED),
        ("FONTNAME",      (0, 1), (0, -1),  "Helvetica-Bold"),
        ("TEXTCOLOR",     (1, 1), (1, -1),  C_SUCCES),
        ("FONTNAME",      (1, 1), (1, -1),  "Helvetica-Bold"),
        ("FONTSIZE",      (1, 1), (1, -1),  11),
        ("TEXTCOLOR",     (2, 1), (2, -1),  C_TEXTE),
        ("FONTSIZE",      (0, 1), (-1, -1), 8.5),
        ("GRID",          (0, 0), (-1, -1), 0.3, C_BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
    ]))
    el.append(tbl_met)
    el.append(Spacer(1, 10))

    # ── Référentiel des 8 classes ─────────────────────────────────
    el.append(Paragraph("Référentiel Complet des 8 Classes de Pannes", s["h3"]))
    el.append(Paragraph(
        "Chaque classe correspond à un mécanisme physique de dégradation distinct. "
        "Le classifieur est entraîné sur des données synthétiques générées selon des "
        "modèles physiques validés par la littérature photovoltaïque (IEC 61215, IEC 61730).",
        s["body"]))
    el.append(Spacer(1, 6))

    indicateurs = {
        "Normal":                "η nominal · FF > 70 % · Rs < 0.5 Ω · Voc nominal",
        "Ombrage partiel":       "ISC ↓ (−20 à −80 %) · Puissance ↓ · Voc stable",
        "Court-circuit":         "VOC ↓↓ (< 15 V) · FF ↓↓ (< 30 %) · ISC stable",
        "Circuit ouvert":        "ISC ≈ 0 A · Puissance ≈ 0 W · VOC peut rester nominal",
        "Dégradation PID":       "η < 60 % nominal · Rsh ↓↓ · facteur idéalité ↑",
        "Encrassement":          "Irradiance ↓ (−10 à −40 %) · η stable · FF stable",
        "Défaut connexion":      "Rs > 1 Ω · Puissance ↓ · ratio_vmpp_voc ↓",
        "Vieillissement accéléré":"η ↓ progressif · FF ↓ · Rs ↑ · score_vieillissement ↑",
    }

    cls_data = [["Id", "Classe de panne", "Description physique (mécanisme)", "Indicateurs électriques"]]
    for cl_id, cl_nom in FAULT_LABELS.items():
        cls_data.append([
            str(cl_id),
            cl_nom,
            DESCRIPTIONS_PHYSIQUES.get(cl_nom, "")[:55],
            indicateurs.get(cl_nom, "—"),
        ])

    tbl_cls = Table(cls_data, colWidths=[0.8*cm, 3.5*cm, 6.8*cm, W_PAGE - 11.1*cm])
    style_cls = []
    for i, cl_id in enumerate(FAULT_LABELS.keys(), start=1):
        c = colors.HexColor(COULEURS_CLASSE.get(cl_id, "#e2e8f0"))
        style_cls += [
            ("TEXTCOLOR", (0, i), (0, i), c),
            ("TEXTCOLOR", (1, i), (1, i), c),
            ("FONTNAME",  (1, i), (1, i), "Helvetica-Bold"),
        ]
    tbl_cls.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  C_BG),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  C_CYAN),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0),  8.5),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_SURFACE, C_SURF2]),
        ("TEXTCOLOR",     (2, 1), (2, -1),  C_TEXTE),
        ("TEXTCOLOR",     (3, 1), (3, -1),  C_MUTED),
        ("FONTNAME",      (0, 1), (0, -1),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 1), (-1, -1), 8.5),
        ("GRID",          (0, 0), (-1, -1), 0.3, C_BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ] + style_cls))
    el.append(tbl_cls)
    el.append(Spacer(1, 10))
    el.append(_hr(C_MUTED))
    el.append(Paragraph(
        "Dataset synthétique généré à des fins pédagogiques — INSAT Tunis 2025–2026. "
        "Projet de Fin d'Année · Barhoumi Montassar · Génie Maintenance & Instrumentation.",
        s["body_muted"]))


# ════════════════════════════════════════════════════════════════════
# Fonction publique
# ════════════════════════════════════════════════════════════════════
def generate_report(diagnostic: dict, mesures: dict,
                    output_dir: str | None = None) -> str:
    """
    Génère un rapport PDF professionnel 7 pages pour un diagnostic solaire.

    Paramètres :
        diagnostic  : dict retourné par /api/diagnose
        mesures     : dict des 16 features brutes saisies
        output_dir  : répertoire de sortie (défaut : backend/data/reports/)

    Retourne :
        Chemin absolu du fichier PDF généré.
    """
    out_dir = Path(output_dir) if output_dir else REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    ts       = datetime.now()
    ts_str   = ts.strftime("%Y%m%d_%H%M%S")
    nom_file = out_dir / f"rapport_diagnostic_{ts_str}.pdf"

    s = _mk_styles()

    panne  = diagnostic.get("panne_detectee", "Diagnostic")
    ts_hum = ts.strftime("%d/%m/%Y %H:%M")
    cb     = _NumeroteurPages(f"Diagnostic : {panne}", ts_hum, total=7)

    doc = SimpleDocTemplate(
        str(nom_file),
        pagesize=A4,
        leftMargin=2*cm,   rightMargin=2*cm,
        topMargin=2.2*cm,  bottomMargin=1.6*cm,
        title=f"Rapport Diagnostic Solaire — {panne}",
        author="Solar AI Diagnostic — INSAT Tunis",
        subject="Diagnostic Photovoltaïque par Intelligence Artificielle",
    )

    elements: list = []
    _p1_garde(elements,               diagnostic, s)
    _p2_resume(elements,              diagnostic, mesures, s)
    _p3_methodes_ia(elements,         diagnostic, s)
    _p4_resultats_detailles(elements, diagnostic, mesures, s)
    _p5_analyse_visuelle(elements,    diagnostic, mesures, s)
    _p6_recommandations(elements,     diagnostic, s)
    _p7_technique(elements,           diagnostic, s)

    doc.build(elements, onFirstPage=cb, onLaterPages=cb)
    return str(nom_file)
