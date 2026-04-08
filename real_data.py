"""
real_data.py — Chargement et prétraitement des données expérimentales réelles.

Description des datasets
─────────────────────────
+------------------+-------+--------+--------+------------------------------------------------+
| Dataset          |  n    |  K/min |  τ(min)|  Contexte biologique                           |
+------------------+-------+--------+--------+------------------------------------------------+
| Eric1002         |  1202 | 0.0055 |  ~126  | E. coli, croissance lente (glycérol ?),        |
|                  |       |        |        | tailles en pixels (1 frame ≈ 5 min)            |
| Eric1009         |  1679 | 0.0055 |  ~127  | idem, replicate indépendant                    |
| glycerol         | 11272 | 0.0134 |   ~52  | E. coli en milieu glycérol (croissance lente)  |
|                  |       |        |        | tailles en µm, âges en min                     |
| synthetic_rich   | 10915 | 0.0304 |   ~23  | Données synthétiques riches (LB-like),         |
|                  |       |        |        | tailles en µm, âges en min                     |
| Lydia3101_new    |   682 | 0.0321 |   ~22  | E. coli pôle neuf (cellules filles récentes)   |
| Lydia3101_old    |   682 | 0.0321 |   ~22  | E. coli pôle vieux (cellules filles "âgées")   |
| TailleToutes     | 65100 | —      |   —    | Toutes les tailles à la naissance (Lydia)      |
+------------------+-------+--------+--------+------------------------------------------------+

Format des colonnes par fichier :
  Eric*          : sb, sd, ad         (taille naissance, taille division, âge, en pixels/frames)
  glycerol       : ad, sb, sd, id     (âge min, tailles µm, incrément µm)
  synthetic_rich : ad, sb, sd, id     (même format que glycerol)
  Lydia3101_*    : ad, sb, sd         (âge en frames, tailles en pixels ~0.065 µm/px)
  TailleToutes   : sb uniquement      (RTF, tailles en pixels)

Remarques :
  - Pour Eric et Lydia : 1 frame ≈ 5 min (fréquence d'acquisition microscopie)
  - Relation vérifiée : id = sd - sb (à précision numérique)
  - K estimé par régression : K = mean(log(sd/sb) / ad) sur chaque dataset
"""

from __future__ import annotations
import re
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ── Constante de conversion temporelle ─────────────────────────────────────
FRAME_TO_MIN = 5.0    # 1 frame microscope ≈ 5 min


@dataclass
class CellDataset:
    """
    Représente un dataset expérimental ou synthétique.

    Attributes
    ----------
    name        : identifiant court ('glycerol', 'Lydia_new', …)
    label       : nom complet pour les figures
    ad          : âge à la division [min]
    sb          : taille à la naissance [µm ou pixels — cohérent en interne]
    sd          : taille à la division [même unité que sb]
    increment   : Z = sd - sb
    K           : taux de croissance exponentielle estimé [1/min]
    tau         : temps de doublement [min] = ln(2)/K
    n           : nombre de cellules
    unit_size   : unité des tailles ('µm' ou 'px')
    condition   : condition biologique ('glycerol', 'rich', 'old_pole', …)
    notes       : informations supplémentaires
    """
    name      : str
    label     : str
    ad        : np.ndarray
    sb        : np.ndarray
    sd        : np.ndarray
    increment : np.ndarray
    K         : float
    tau       : float
    n         : int
    unit_size : str = 'µm'
    condition : str = ''
    notes     : str = ''

    def summary(self) -> str:
        return (
            f"{self.name:20s}  n={self.n:6d}  "
            f"K={self.K:.5f}/min  τ={self.tau:.1f} min  "
            f"<sb>={self.sb.mean():.2f}  <sd>={self.sd.mean():.2f}  "
            f"<ad>={self.ad.mean():.1f} min"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Fonctions de chargement
# ═══════════════════════════════════════════════════════════════════════════

def _estimate_K(sb: np.ndarray, sd: np.ndarray, ad: np.ndarray) -> float:
    """
    Estime K = mean[log(sd/sb) / ad] sur les cellules valides.
    Filtre les ratios incohérents (sd < sb ou ad ≤ 0).
    """
    mask = (sd > sb) & (sb > 0) & (ad > 0)
    return float(np.mean(np.log(sd[mask] / sb[mask]) / ad[mask]))


def _filter_outliers(arr: np.ndarray, q_low: float = 0.01,
                     q_high: float = 0.99) -> np.ndarray:
    """Masque booléen éliminant les valeurs hors du quantile [q_low, q_high]."""
    lo = np.quantile(arr, q_low)
    hi = np.quantile(arr, q_high)
    return (arr >= lo) & (arr <= hi)


def load_glycerol(path: str, q: float = 0.005) -> CellDataset:
    """
    Format : ad, sb, sd, id  (âge min, tailles µm, incrément µm)
    Milieu glycérol → croissance lente (τ ≈ 52 min).
    """
    d = np.loadtxt(path, delimiter=',')
    ad, sb, sd, inc = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
    # Vérification incrément
    assert np.max(np.abs(inc - (sd - sb))) < 1e-6, "Incrément incohérent"
    # Filtre outliers multivariés
    mask = (_filter_outliers(ad, q, 1-q) &
            _filter_outliers(sb, q, 1-q) &
            _filter_outliers(sd, q, 1-q) &
            (sd > sb) & (sb > 0) & (ad > 0))
    ad, sb, sd = ad[mask], sb[mask], sd[mask]
    K = _estimate_K(sb, sd, ad)
    return CellDataset(
        name='glycerol', label='E. coli – Glycérol (croissance lente)',
        ad=ad, sb=sb, sd=sd, increment=sd - sb,
        K=K, tau=np.log(2) / K, n=len(ad),
        unit_size='µm', condition='glycerol',
        notes='Milieu glycérol, τ≈52 min, tailles en µm, âges en min',
    )


def load_synthetic_rich(path: str, q: float = 0.005) -> CellDataset:
    """
    Format : ad, sb, sd, id  (même format que glycerol)
    Données synthétiques simulant un milieu riche (τ ≈ 23 min).
    """
    d = np.loadtxt(path, delimiter=',')
    ad, sb, sd, inc = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
    mask = (_filter_outliers(ad, q, 1-q) &
            _filter_outliers(sb, q, 1-q) &
            _filter_outliers(sd, q, 1-q) &
            (sd > sb) & (sb > 0) & (ad > 0))
    ad, sb, sd = ad[mask], sb[mask], sd[mask]
    K = _estimate_K(sb, sd, ad)
    return CellDataset(
        name='synthetic_rich', label='Synthétique – Milieu riche',
        ad=ad, sb=sb, sd=sd, increment=sd - sb,
        K=K, tau=np.log(2) / K, n=len(ad),
        unit_size='µm', condition='rich',
        notes='Données synthétiques, τ≈23 min',
    )


def load_lydia(path_new: str, path_old: str,
               frame_to_min: float = FRAME_TO_MIN) -> tuple:
    """
    Format : ad, sb, sd  (frames, pixels, pixels)
    Paire new/old pole de la même expérience (E. coli Lydia3101).
    Conversion : âge en minutes, tailles gardées en pixels (cohérence interne).
    τ ≈ 22 min (milieu riche).
    """
    datasets = []
    for path, pole in [(path_new, 'new'), (path_old, 'old')]:
        d    = np.loadtxt(path, delimiter=',')
        ad_f = d[:, 0]   # frames
        sb   = d[:, 1]   # pixels
        sd   = d[:, 2]   # pixels
        ad   = ad_f * frame_to_min   # → minutes

        mask = (_filter_outliers(ad_f, 0.01, 0.99) &
                _filter_outliers(sb, 0.01, 0.99) &
                _filter_outliers(sd, 0.01, 0.99) &
                (sd > sb) & (sb > 0) & (ad_f > 0))
        ad, sb, sd = ad[mask], sb[mask], sd[mask]
        K = _estimate_K(sb, sd, ad)
        datasets.append(CellDataset(
            name=f'Lydia_{pole}',
            label=f'E. coli Lydia3101 – Pôle {"neuf" if pole=="new" else "vieux"}',
            ad=ad, sb=sb, sd=sd, increment=sd - sb,
            K=K, tau=np.log(2) / K, n=len(ad),
            unit_size='px', condition=f'{pole}_pole',
            notes=f'Pôle {pole}, tailles en pixels, âge en min (5min/frame)',
        ))
    return datasets[0], datasets[1]


def load_eric(path: str, name: str, frame_to_min: float = FRAME_TO_MIN) -> CellDataset:
    """
    Format : sb, sd, ad  (pixels, pixels, frames)
    Expériences microfluidique Eric (lame+couverture), croissance lente τ≈126 min.
    """
    d    = np.loadtxt(path, delimiter=',')
    sb   = d[:, 0]
    sd   = d[:, 1]
    ad_f = d[:, 2]
    ad   = ad_f * frame_to_min

    mask = (_filter_outliers(ad_f, 0.01, 0.99) &
            _filter_outliers(sb, 0.01, 0.99) &
            _filter_outliers(sd, 0.01, 0.99) &
            (sd > sb) & (sb > 0) & (ad_f > 0))
    ad, sb, sd = ad[mask], sb[mask], sd[mask]
    K = _estimate_K(sb, sd, ad)
    return CellDataset(
        name=name,
        label=f'E. coli {name} – Croissance lente',
        ad=ad, sb=sb, sd=sd, increment=sd - sb,
        K=K, tau=np.log(2) / K, n=len(ad),
        unit_size='px', condition='slow_growth',
        notes='Tailles en pixels, âge en min (5min/frame), τ≈126 min',
    )


# ═══════════════════════════════════════════════════════════════════════════
# Loader principal
# ═══════════════════════════════════════════════════════════════════════════

def load_all_datasets(data_dir: str) -> dict[str, CellDataset]:
    """
    Charge tous les datasets disponibles depuis le répertoire data_dir.

    Returns
    -------
    dict {name: CellDataset}
    """
    p = Path(data_dir)
    datasets = {}

    # Glycérol
    g_path = next(p.glob('*glycerol*'), None)
    if g_path:
        ds = load_glycerol(str(g_path))
        datasets[ds.name] = ds

    # Synthétique riche
    s_path = next(p.glob('*synthetic_rich*'), None)
    if s_path:
        ds = load_synthetic_rich(str(s_path))
        datasets[ds.name] = ds

    # Lydia new/old
    l_new = next(p.glob('*Lydia*new*'), None)
    l_old = next(p.glob('*Lydia*old*'), None)
    if l_new and l_old:
        ds_new, ds_old = load_lydia(str(l_new), str(l_old))
        datasets[ds_new.name] = ds_new
        datasets[ds_old.name] = ds_old

    # Eric
    for eric_path in sorted(p.glob('*Eric*MDJ*')):
        name_raw = eric_path.stem
        # Extraire identifiant court
        m = re.search(r'(Eric\d+)', name_raw, re.IGNORECASE)
        name = m.group(1) if m else name_raw[:10]
        ds = load_eric(str(eric_path), name)
        datasets[ds.name] = ds

    return datasets


# ═══════════════════════════════════════════════════════════════════════════
# Statistiques descriptives
# ═══════════════════════════════════════════════════════════════════════════

def dataset_summary_table(datasets: dict) -> pd.DataFrame:
    """
    Tableau récapitulatif de tous les datasets.
    """
    rows = []
    for name, ds in datasets.items():
        rows.append({
            'Dataset'        : ds.label,
            'n'              : ds.n,
            'K (1/min)'      : round(ds.K, 5),
            'τ (min)'        : round(ds.tau, 1),
            '<sb>'           : round(ds.sb.mean(), 3),
            '<sd>'           : round(ds.sd.mean(), 3),
            '<ad> (min)'     : round(ds.ad.mean(), 1),
            'CV(sb)'         : round(ds.sb.std() / ds.sb.mean(), 3),
            'CV(ad)'         : round(ds.ad.std() / ds.ad.mean(), 3),
            '<sd>/<sb>'      : round(ds.sd.mean() / ds.sb.mean(), 3),
            'Unité taille'   : ds.unit_size,
        })
    return pd.DataFrame(rows).set_index('Dataset')


def compute_correlations(ds: CellDataset) -> dict:
    """
    Calcule les corrélations entre variables clés.
    Utile pour discriminer le modèle (age, size, increment).

    Corrélations biologiquement importantes :
      corr(sb, sd) ≈ 1         → cohérent avec tout modèle
      corr(sb, increment)       → discriminant entre adder (≈0) et sizer (>0)
      corr(sb, ad)              → lien taille naissance / age division
    """
    from scipy.stats import pearsonr, spearmanr
    pairs = {
        'corr(sb, sd)'        : (ds.sb, ds.sd),
        'corr(sb, increment)' : (ds.sb, ds.increment),
        'corr(sb, ad)'        : (ds.sb, ds.ad),
        'corr(ad, increment)' : (ds.ad, ds.increment),
        'corr(sd, increment)' : (ds.sd, ds.increment),
    }
    result = {}
    for label, (x, y) in pairs.items():
        r, p = pearsonr(x, y)
        result[label] = {'pearson_r': round(r, 4), 'p_value': round(p, 6)}
    return result
