"""
Génération du dataset synthétique de défauts photovoltaïques.

Chaque classe suit des règles physiques réalistes dérivées du modèle
de diode unique (single-diode model) utilisé en électronique PV.

Classes :
    0 - Normal
    1 - Ombrage partiel
    2 - Court-circuit
    3 - Circuit ouvert
    4 - Dégradation PID
    5 - Encrassement
    6 - Défaut connexion
    7 - Vieillissement accéléré
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Reproductibilité
RNG = np.random.default_rng(seed=42)

# Nombre de lignes par classe (6000 total / 8 classes = 750 par classe)
N_PAR_CLASSE = 750
N_CLASSES = 8
OUTPUT_CSV = Path(__file__).parent / "solar_faults.csv"


# ---------------------------------------------------------------------------
# Fonctions utilitaires de bruit
# ---------------------------------------------------------------------------

def bruit(taille: int, sigma: float) -> np.ndarray:
    """Bruit gaussien centré sur zéro."""
    return RNG.normal(0, sigma, taille)


def uniforme(bas: float, haut: float, taille: int) -> np.ndarray:
    """Tirage uniforme dans [bas, haut]."""
    return RNG.uniform(bas, haut, taille)


def clip(arr: np.ndarray, bas: float, haut: float) -> np.ndarray:
    """Borne un array dans [bas, haut]."""
    return np.clip(arr, bas, haut)


# ---------------------------------------------------------------------------
# Générateurs par classe
# ---------------------------------------------------------------------------

def classe_normal(n: int) -> pd.DataFrame:
    """
    Classe 0 — Fonctionnement nominal.
    Toutes les mesures dans leurs plages opérationnelles standards.
    Le fill_factor nominal est typiquement entre 70 et 82 % pour du silicium monocristallin.
    """
    irr  = uniforme(600, 1100, n) + bruit(n, 30)
    t_pa = uniforme(25, 65, n)    + bruit(n, 2)
    t_am = uniforme(10, 35, n)    + bruit(n, 1.5)
    voc  = uniforme(36, 46, n)    + bruit(n, 0.8)
    isc  = uniforme(8, 11.5, n)   + bruit(n, 0.2)

    # Puissance = η × aire × irradiance (nominal : 300–420 W pour un module 400 Wc)
    pmp  = uniforme(280, 420, n)  + bruit(n, 10)
    vmp  = uniforme(30, 40, n)    + bruit(n, 0.5)
    imp  = pmp / vmp
    rs   = uniforme(0.1, 0.8, n)  + bruit(n, 0.05)
    rsh  = uniforme(3000, 9000, n) + bruit(n, 200)
    ff   = (pmp / (voc * isc)) * 100
    eff  = uniforme(16, 21, n)    + bruit(n, 0.5)
    tfonc = uniforme(0, 20000, n)
    hum  = uniforme(20, 70, n)    + bruit(n, 3)
    vent = uniforme(0, 12, n)     + bruit(n, 1)
    nid  = uniforme(1.1, 1.4, n)  + bruit(n, 0.05)

    return _assemble(0, n, irr, t_pa, t_am, voc, isc, pmp, vmp, imp, rs, rsh, ff, eff, tfonc, hum, vent, nid)


def classe_ombrage(n: int) -> pd.DataFrame:
    """
    Classe 1 — Ombrage partiel.
    L'ombrage crée des cellules froides qui agissent comme charges résistives.
    - Puissance réduite de 30 à 60 %
    - Courant ISC réduit (cellules en série affectées)
    - Fill factor dégradé (courbe IV déformée avec plusieurs maxima locaux)
    - Resistance shunt peut baisser (bypass diodes activées)
    """
    irr  = uniforme(400, 1000, n) + bruit(n, 40)
    t_pa = uniforme(20, 60, n)    + bruit(n, 3)
    t_am = uniforme(10, 35, n)    + bruit(n, 2)
    voc  = uniforme(30, 44, n)    + bruit(n, 1)     # légèrement réduit
    isc  = uniforme(4, 8, n)      + bruit(n, 0.3)   # réduit (cellules ombrées)

    # Puissance réduite 30–60 %
    facteur_reduction = uniforme(0.40, 0.70, n)
    pmp  = uniforme(280, 420, n) * facteur_reduction + bruit(n, 15)
    vmp  = uniforme(25, 38, n)   + bruit(n, 1)
    imp  = pmp / np.where(vmp > 0, vmp, 1)
    rs   = uniforme(0.2, 1.5, n) + bruit(n, 0.1)
    rsh  = uniforme(300, 1500, n) + bruit(n, 50)    # réduit (bypass actifs)
    ff   = clip((pmp / (voc * isc + 1e-9)) * 100 + bruit(n, 2), 35, 54)  # < 55 %
    eff  = uniforme(8, 15, n)    + bruit(n, 0.8)
    tfonc = uniforme(0, 40000, n)
    hum  = uniforme(20, 80, n)   + bruit(n, 4)
    vent = uniforme(0, 15, n)    + bruit(n, 1)
    nid  = uniforme(1.3, 1.8, n) + bruit(n, 0.05)

    return _assemble(1, n, irr, t_pa, t_am, voc, isc, pmp, vmp, imp, rs, rsh, ff, eff, tfonc, hum, vent, nid)


def classe_court_circuit(n: int) -> pd.DataFrame:
    """
    Classe 2 — Court-circuit (shunt interne).
    La résistance shunt très faible dérive le courant de la charge :
    - Rsh << 150 Ω (chemin de fuite direct entre p et n)
    - Courant ISC très faible (courant dévié par le shunt)
    - Tension VOC s'effondre
    - Fill factor très bas
    """
    irr  = uniforme(200, 1100, n) + bruit(n, 50)
    t_pa = uniforme(20, 75, n)    + bruit(n, 3)
    t_am = uniforme(5, 40, n)     + bruit(n, 2)
    voc  = uniforme(5, 22, n)     + bruit(n, 1.5)   # chute sévère de VOC
    isc  = uniforme(0.1, 2.5, n)  + bruit(n, 0.2)   # très bas

    pmp  = uniforme(2, 40, n)     + bruit(n, 3)      # quasi nulle
    vmp  = uniforme(2, 15, n)     + bruit(n, 0.5)
    imp  = pmp / np.where(vmp > 0, vmp, 1)
    rs   = uniforme(0.1, 1.0, n)  + bruit(n, 0.05)
    rsh  = uniforme(5, 149, n)    + bruit(n, 5)      # < 150 Ω (règle physique)
    ff   = clip((pmp / (voc * isc + 1e-9)) * 100 + bruit(n, 3), 20, 50)
    eff  = uniforme(0.5, 5, n)    + bruit(n, 0.3)
    tfonc = uniforme(500, 30000, n)
    hum  = uniforme(30, 95, n)    + bruit(n, 5)
    vent = uniforme(0, 18, n)     + bruit(n, 2)
    nid  = uniforme(1.5, 2.0, n)  + bruit(n, 0.05)

    return _assemble(2, n, irr, t_pa, t_am, voc, isc, pmp, vmp, imp, rs, rsh, ff, eff, tfonc, hum, vent, nid)


def classe_circuit_ouvert(n: int) -> pd.DataFrame:
    """
    Classe 3 — Circuit ouvert.
    Rupture totale du circuit : courant = 0 A, puissance = 0 W.
    La résistance série tend vers l'infini (modélisée ici > 4 Ω).
    VOC peut rester mesurable (tension en circuit ouvert).
    """
    irr  = uniforme(100, 1100, n) + bruit(n, 40)
    t_pa = uniforme(15, 70, n)    + bruit(n, 3)
    t_am = uniforme(-5, 40, n)    + bruit(n, 2)
    voc  = uniforme(28, 46, n)    + bruit(n, 1)      # VOC mesurable (pas de courant)
    isc  = clip(bruit(n, 0.05), -0.1, 0.15)          # ≈ 0 A

    pmp  = clip(bruit(n, 0.5), 0, 2)                 # ≈ 0 W
    vmp  = clip(bruit(n, 0.5), 0, 2)
    imp  = clip(bruit(n, 0.05), 0, 0.1)
    rs   = uniforme(4.1, 5.0, n)  + bruit(n, 0.1)    # > 4 Ω (règle physique)
    rsh  = uniforme(1000, 9000, n) + bruit(n, 200)    # élevé (pas de shunt)
    ff   = clip(bruit(n, 2) + 45, 35, 55)
    eff  = clip(bruit(n, 0.3), 0, 1.5)               # quasi nulle
    tfonc = uniforme(1000, 45000, n)
    hum  = uniforme(15, 90, n)    + bruit(n, 4)
    vent = uniforme(0, 20, n)     + bruit(n, 1.5)
    nid  = uniforme(1.0, 1.5, n)  + bruit(n, 0.05)

    return _assemble(3, n, irr, t_pa, t_am, voc, isc, pmp, vmp, imp, rs, rsh, ff, eff, tfonc, hum, vent, nid)


def classe_pid(n: int) -> pd.DataFrame:
    """
    Classe 4 — Dégradation PID (Potential Induced Degradation).
    La haute tension entre cadre et cellules crée un courant de fuite à travers
    l'encapsulant, réduisant drastiquement Rsh et l'efficacité.
    - Efficacité < 12 %
    - Rsh basse (fuites ioniques)
    - VOC réduit (dérive des porteurs)
    """
    irr  = uniforme(400, 1100, n) + bruit(n, 35)
    t_pa = uniforme(30, 80, n)    + bruit(n, 3)      # modules chauds favorisent PID
    t_am = uniforme(15, 45, n)    + bruit(n, 2)
    voc  = uniforme(22, 34, n)    + bruit(n, 1.2)    # réduit (dérive de tension)
    isc  = uniforme(5, 9, n)      + bruit(n, 0.3)

    pmp  = uniforme(80, 200, n)   + bruit(n, 10)     # fortement réduit
    vmp  = uniforme(18, 30, n)    + bruit(n, 0.8)
    imp  = pmp / np.where(vmp > 0, vmp, 1)
    rs   = uniforme(0.3, 2.0, n)  + bruit(n, 0.1)
    rsh  = uniforme(80, 600, n)   + bruit(n, 30)     # basse (fuites ioniques)
    ff   = clip((pmp / (voc * isc + 1e-9)) * 100 + bruit(n, 2), 40, 62)
    eff  = clip(uniforme(5, 11.9, n) + bruit(n, 0.5), 4, 11.9)  # < 12 %
    tfonc = uniforme(5000, 40000, n)
    hum  = uniforme(40, 95, n)    + bruit(n, 4)      # humidité élevée aggrave PID
    vent = uniforme(0, 15, n)     + bruit(n, 1)
    nid  = uniforme(1.5, 2.0, n)  + bruit(n, 0.05)

    return _assemble(4, n, irr, t_pa, t_am, voc, isc, pmp, vmp, imp, rs, rsh, ff, eff, tfonc, hum, vent, nid)


def classe_encrassement(n: int) -> pd.DataFrame:
    """
    Classe 5 — Encrassement (soiling).
    Couche de poussière/pollen qui filtre le rayonnement incident.
    L'irradiance effective est réduite, mais la physique de la cellule reste saine.
    - Efficacité modérément basse (module intact mais moins de lumière)
    - ISC proportionnel à l'irradiance effective réduite
    - Rsh et Rs dans les plages normales
    """
    # Irradiance effective = irradiance brute × (1 - taux d'encrassement)
    irr_brute = uniforme(500, 1100, n)
    taux_encr = uniforme(0.15, 0.55, n)              # 15 à 55 % d'occultation
    irr  = irr_brute * (1 - taux_encr) + bruit(n, 20)

    t_pa = uniforme(20, 60, n)    + bruit(n, 2)
    t_am = uniforme(5, 38, n)     + bruit(n, 2)
    voc  = uniforme(33, 44, n)    + bruit(n, 0.8)    # peu affecté (tension log de l'irr)
    isc  = (irr / 1000) * uniforme(7, 10, n) + bruit(n, 0.2)  # ∝ irradiance

    pmp  = isc * uniforme(28, 36, n) * uniforme(0.70, 0.80, n) + bruit(n, 8)
    vmp  = uniforme(28, 38, n)    + bruit(n, 0.5)
    imp  = pmp / np.where(vmp > 0, vmp, 1)
    rs   = uniforme(0.2, 1.0, n)  + bruit(n, 0.05)
    rsh  = uniforme(2000, 8000, n) + bruit(n, 150)   # Rsh normale
    ff   = clip((pmp / (voc * isc + 1e-9)) * 100 + bruit(n, 2), 58, 78)
    eff  = clip(uniforme(10, 15.5, n) + bruit(n, 0.6), 8, 16)  # modérément basse
    tfonc = uniforme(0, 35000, n)
    hum  = uniforme(10, 85, n)    + bruit(n, 4)
    vent = uniforme(0, 20, n)     + bruit(n, 1.5)
    nid  = uniforme(1.1, 1.5, n)  + bruit(n, 0.05)

    return _assemble(5, n, irr, t_pa, t_am, voc, isc, pmp, vmp, imp, rs, rsh, ff, eff, tfonc, hum, vent, nid)


def classe_defaut_connexion(n: int) -> pd.DataFrame:
    """
    Classe 6 — Défaut de connexion (mauvais contact, corrosion, soudure froide).
    La résistance série augmente fortement à cause de la résistance de contact.
    - Rs > 3.5 Ω (résistance de contact élevée)
    - Puissance instable (fluctuations importantes)
    - Fill factor réduit (pertes ohmiques)
    """
    irr  = uniforme(300, 1100, n) + bruit(n, 40)
    t_pa = uniforme(20, 75, n)    + bruit(n, 3)
    t_am = uniforme(5, 42, n)     + bruit(n, 2)
    voc  = uniforme(32, 44, n)    + bruit(n, 1)
    isc  = uniforme(6, 10, n)     + bruit(n, 0.4)

    # Puissance instable : bruit standard × 3 pour simuler les fluctuations
    pmp_base = uniforme(150, 350, n)
    pmp  = pmp_base + bruit(n, 35)                   # forte variance (instabilité)
    vmp  = uniforme(26, 38, n)    + bruit(n, 1.2)
    imp  = pmp / np.where(vmp > 0, vmp, 1)
    rs   = uniforme(3.6, 4.9, n)  + bruit(n, 0.15)   # > 3.5 Ω (règle physique)
    rsh  = uniforme(800, 5000, n) + bruit(n, 100)
    ff   = clip((pmp / (voc * isc + 1e-9)) * 100 + bruit(n, 3), 42, 65)
    eff  = uniforme(10, 17, n)    + bruit(n, 0.8)
    tfonc = uniforme(2000, 45000, n)
    hum  = uniforme(20, 90, n)    + bruit(n, 5)       # humidité favorise la corrosion
    vent = uniforme(0, 18, n)     + bruit(n, 1.5)
    nid  = uniforme(1.2, 1.8, n)  + bruit(n, 0.05)

    return _assemble(6, n, irr, t_pa, t_am, voc, isc, pmp, vmp, imp, rs, rsh, ff, eff, tfonc, hum, vent, nid)


def classe_vieillissement(n: int) -> pd.DataFrame:
    """
    Classe 7 — Vieillissement accéléré.
    Dégradation lente mais cumulative : jaunissement EVA, micro-cracks,
    délaminiation. Typique après > 30 000 h de fonctionnement.
    - Efficacité dégradée (taux de dégradation ~0.5 %/an nominal)
    - Fill factor en baisse progressive
    - Temps de fonctionnement > 30 000 h (règle physique)
    """
    irr  = uniforme(400, 1100, n) + bruit(n, 35)
    t_pa = uniforme(25, 75, n)    + bruit(n, 3)
    t_am = uniforme(5, 42, n)     + bruit(n, 2)
    voc  = uniforme(30, 42, n)    + bruit(n, 1)      # légèrement réduit
    isc  = uniforme(6, 10, n)     + bruit(n, 0.3)

    # Dégradation proportionnelle à la durée de fonctionnement
    tfonc = uniforme(30001, 50000, n)                 # > 30 000 h (règle physique)
    facteur_vieillissement = 1 - (tfonc - 30000) / 200000  # max ~10 % de dégradation supplémentaire

    pmp  = uniforme(180, 320, n) * facteur_vieillissement + bruit(n, 12)
    vmp  = uniforme(26, 37, n)   + bruit(n, 0.7)
    imp  = pmp / np.where(vmp > 0, vmp, 1)
    rs   = uniforme(0.8, 2.5, n) + bruit(n, 0.1)    # augmente avec le vieillissement
    rsh  = uniforme(500, 3000, n) + bruit(n, 100)    # se dégrade
    ff   = clip((pmp / (voc * isc + 1e-9)) * 100 + bruit(n, 2), 45, 68)
    eff  = clip(uniforme(10, 15, n) * facteur_vieillissement + bruit(n, 0.6), 7, 15)
    hum  = uniforme(20, 80, n)   + bruit(n, 4)
    vent = uniforme(0, 18, n)    + bruit(n, 1.5)
    nid  = uniforme(1.3, 1.9, n) + bruit(n, 0.05)   # facteur d'idéalité augmente avec âge

    return _assemble(7, n, irr, t_pa, t_am, voc, isc, pmp, vmp, imp, rs, rsh, ff, eff, tfonc, hum, vent, nid)


# ---------------------------------------------------------------------------
# Assemblage d'un DataFrame par classe
# ---------------------------------------------------------------------------

def _assemble(classe: int, n: int,
              irr, t_pa, t_am, voc, isc, pmp, vmp, imp,
              rs, rsh, ff, eff, tfonc, hum, vent, nid) -> pd.DataFrame:
    """Assemble et borne les valeurs dans leurs plages physiques admissibles."""
    return pd.DataFrame({
        "irradiance":            clip(irr,   0,     1200),
        "temperature_panneau":   clip(t_pa,  15,    85),
        "temperature_ambiante":  clip(t_am,  -5,    45),
        "tension_voc":           clip(voc,   0,     50),
        "courant_isc":           clip(isc,   0,     12),
        "puissance_mpp":         clip(pmp,   0,     550),
        "tension_mpp":           clip(vmp,   0,     42),
        "courant_mpp":           clip(imp,   0,     13),
        "resistance_serie":      clip(rs,    0.1,   5),
        "resistance_shunt":      clip(rsh,   100,   10000),
        "fill_factor":           clip(ff,    40,    85),
        "efficacite":            clip(eff,   0,     22),
        "temps_fonctionnement":  clip(tfonc, 0,     50000),
        "humidite":              clip(hum,   10,    95),
        "vitesse_vent":          clip(vent,  0,     20),
        "facteur_idealite":      clip(nid,   1,     2),
        "classe":                np.full(n, classe, dtype=int),
    })


# ---------------------------------------------------------------------------
# Générateur principal
# ---------------------------------------------------------------------------

def generate_solar_dataset(n_par_classe: int = N_PAR_CLASSE,
                           output_path: Path = OUTPUT_CSV) -> pd.DataFrame:
    """
    Génère le dataset complet en combinant toutes les classes.
    Mélange aléatoire pour éviter tout ordre systématique.
    """
    generateurs = [
        classe_normal,
        classe_ombrage,
        classe_court_circuit,
        classe_circuit_ouvert,
        classe_pid,
        classe_encrassement,
        classe_defaut_connexion,
        classe_vieillissement,
    ]

    blocs = [gen(n_par_classe) for gen in generateurs]
    df = pd.concat(blocs, ignore_index=True)

    # Mélange reproductible
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Arrondi à 4 décimales pour lisibilité
    colonnes_num = [c for c in df.columns if c != "classe"]
    df[colonnes_num] = df[colonnes_num].round(4)

    # Sauvegarde
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset sauvegardé : {output_path}  ({len(df)} lignes)")

    return df


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Génération du dataset solaire photovoltaïque")
    print("=" * 60)

    df = generate_solar_dataset()

    # Noms des classes pour l'affichage
    NOMS_CLASSES = {
        0: "Normal",
        1: "Ombrage partiel",
        2: "Court-circuit",
        3: "Circuit ouvert",
        4: "Dégradation PID",
        5: "Encrassement",
        6: "Défaut connexion",
        7: "Vieillissement accéléré",
    }

    print("\n--- 5 premières lignes ---")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.3f}".format)
    print(df.head())

    print("\n--- Distribution des classes ---")
    distrib = df["classe"].value_counts().sort_index()
    total = len(df)
    for cls_id, count in distrib.items():
        pct = count / total * 100
        barre = "█" * int(pct / 2)
        print(f"  Classe {cls_id} — {NOMS_CLASSES[cls_id]:<25} : {count:>5} ({pct:5.1f}%)  {barre}")

    print(f"\nTotal : {total} lignes | {df['classe'].nunique()} classes | "
          f"{df.shape[1] - 1} features")
    print("=" * 60)
