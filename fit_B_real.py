
#!/usr/bin/env python3
"""
fit_B_real.py — Fit paramétrique de B̂(z) sur les données réelles Eric.

Pour Eric1002 et Eric1009 (modèle incrément, meilleur selon model_selection.json) :
  1. Estime B̂(z) par Tikhonov p=1
  2. Fitte 4 familles paramétriques par moindres carrés :
       - Gamma généralisé : B(z) = (k/σ)(z/σ)^{k−1}
       - Weibull2 (k=2 fixé) : B(z) = 2z/σ²
       - Constant            : B(z) = λ
       - Puissance           : B(z) = β·z^γ
  3. Compare les fits : R², résidu L², AIC
  4. Bootstrap (n_boot=200 répétitions) pour obtenir des IC 95% sur k et σ² 

Usage
─────
    python fit_B_real.py                       # tout analyser
    python fit_B_real.py --dataset Eric1002    # un seul dataset
    python fit_B_real.py --n-boot 100          # bootstrap plus rapide
    python fit_B_real.py --no-plots            # sans figures

Sorties
───────
    results_real/fit_B_parametric.json  : paramètres fittés + IC bootstrap
    figures_real/fit_B_<dataset>.png    : figure de comparaison des fits
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

FIG_DIR = ROOT / 'figures_real'
RES_DIR = ROOT / 'results_real'
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

from real_data import load_all_datasets
from real_analysis import estimate_B_real
from src import DirectProblemSolver, NelsonAalanEstimator, TikhonovRegularizer, DiscrepancyPrinciple

plt.rcParams.update({
    'axes.spines.top': False, 'axes.spines.right': False,
    'font.size': 10, 'figure.dpi': 110,
})


# ═══════════════════════════════════════════════════════════════════════════
# 1. Familles paramétriques pour B
# ═══════════════════════════════════════════════════════════════════════════

def B_weibull(z: np.ndarray, k: float, sigma: float) -> np.ndarray:
    """
    B(z) = (k/σ)(z/σ)^{k−1}  — Weibull(k, σ).

    C'est le taux de hasard de la loi de Weibull.
    Cas particuliers : k=1 → constant, k=2 → Weibull2.
    NB : c'est un cas particulier du gamma généralisé avec d=p=k.
    """
    z = np.clip(np.asarray(z, float), 1e-12, None)
    return (k / sigma) * (z / sigma) ** (k - 1)


def B_gamma_generalized(z: np.ndarray, a: float, d: float, p: float) -> np.ndarray:
    """
    Taux de hasard de la loi Gamma Généralisée à 3 paramètres (Stacy, 1962).

    La densité est :
        f(z) = (p / a^d / Γ(d/p)) · z^{d−1} · exp(−(z/a)^p)

    Le taux de hasard est :
        B(z) = f(z) / S(z)

    où S(z) = 1 − F(z) est la survie. Comme B(z) n'a pas de forme
    fermée simple en général, on l'approxime ici par :

        B(z) ≈ (p / a^p) · z^{p−1} · [d/p termes de correction]

    En pratique on utilise la forme approximée :
        B(z) = (p · d) / (a^p · Γ(d/p)) · z^{p−1} · exp(−(z/a)^p)
               ────────────────────────────────────────────────────────
               Γ(d/p) · S_GG(z ; a, d, p)

    Puisque S_GG n'est pas analytique, on utilise la forme du taux de hasard
    de la GG telle que définie dans la littérature de survie (Cox, 2008) :

        B(z) = [p · (z/a)^d · exp(−(z/a)^p)] / [a · Γ(d/p) · S_GG(z)]

    Pour l'estimation par curve_fit, on utilise la forme proportionnelle
    (constante de normalisation absorbée dans le fit) :

        B(z) ∝ z^{d−1} · exp(−(z/a)^p)

    avec un préfacteur C ajusté automatiquement — ce qui revient à fitter
    une loi à 3 paramètres (a, d, p) + 1 constante.

    Paramètres
    ----------
    a : paramètre d'échelle (a > 0)
    d : paramètre de forme 1 (d > 0) — contrôle la queue gauche
    p : paramètre de forme 2 (p > 0) — contrôle la queue droite
        • p=1, d=1 → Exponentielle
        • p=1      → Gamma(d)
        • d=p      → Weibull(p, a)
        • d=p=1    → Exponentielle
    """
    from scipy.special import gamma as gamma_fn
    z = np.clip(np.asarray(z, float), 1e-12, None)
    # Taux de hasard exact de la GG (forme de Prentice 1974) :
    # h(z) = (p/a) * (z/a)^{d-1} * exp(-(z/a)^p) / [Γ(d/p) * (1 - I(d/p, (z/a)^p))]
    # où I est la fonction Gamma incomplète régularisée.
    # On utilise scipy pour la fonction de survie exacte.
    from scipy.stats import gengamma
    # gengamma(c, a) dans scipy : c = p/d, a = p  (convention différente)
    # On passe par la formule directe de la densité / survie
    lnf = (np.log(p) - np.log(a)
           + (d - 1) * np.log(z / a)
           - (z / a) ** p
           - np.log(gamma_fn(d / p)))
    f   = np.exp(np.clip(lnf, -500, 500))
    # Survie via intégration numérique sur la grille locale
    # (approximation valable pour le fit)
    S   = np.exp(-(z / a) ** p) * _gamma_survival_correction(z, a, d, p)
    S   = np.clip(S, 1e-15, 1.0)
    return f / S


def _gamma_survival_correction(z, a, d, p):
    """
    Correction de survie pour la GG : ratio Γ_sup(d/p, (z/a)^p) / Γ(d/p).
    Calculé via scipy.special.gammaincc (fonction Gamma incomplète supérieure).
    """
    from scipy.special import gammaincc
    u = (z / a) ** p
    return np.clip(gammaincc(d / p, u), 1e-15, 1.0)


def B_weibull2(z: np.ndarray, sigma: float) -> np.ndarray:
    """B(z) = 2z/σ²  — Weibull k=2 (cas particulier)."""
    return 2 * np.asarray(z, float) / sigma ** 2


def B_constant(z: np.ndarray, lam: float) -> np.ndarray:
    """B(z) = λ  — taux constant (memoryless)."""
    return np.full_like(np.asarray(z, float), lam)


def B_power(z: np.ndarray, beta: float, gamma: float) -> np.ndarray:
    """B(z) = β·z^γ  — puissance."""
    z = np.clip(np.asarray(z, float), 1e-12, None)
    return beta * z ** gamma


# Catalogue des familles : nom → (fonction, p0, bornes)
FAMILIES = {
    'weibull': {
        'func'  : B_weibull,
        'p0'    : [2.0, 1.0],
        'bounds': ([0.1, 1e-3], [10.0, 1e3]),
        'labels': ['k', 'σ'],
        'latex' : r'$B(z) = \frac{k}{\sigma}\left(\frac{z}{\sigma}\right)^{k-1}$',
    },
    'gamma_generalized': {
        'func'  : B_gamma_generalized,
        'p0'    : [30.0, 3.0, 3.0],          # calé sur échelle des données Eric
        'bounds': ([1e-2, 0.5, 0.5], [500.0, 15.0, 15.0]),
        'labels': ['a', 'd', 'p'],
        'latex' : r'$B(z) \propto z^{d-1} e^{-(z/a)^p}$  [GG(a,d,p)]',
    },
    'weibull2': {
        'func'  : B_weibull2,
        'p0'    : [1.0],
        'bounds': ([1e-3], [1e3]),
        'labels': ['σ'],
        'latex' : r'$B(z) = 2z/\sigma^2$',
    },
    'constant': {
        'func'  : B_constant,
        'p0'    : [1.0],
        'bounds': ([1e-4], [1e4]),
        'labels': ['λ'],
        'latex' : r'$B(z) = \lambda$',
    },
    'power': {
        'func'  : B_power,
        'p0'    : [1.0, 1.0],
        'bounds': ([1e-4, 0.01], [1e4, 10.0]),
        'labels': ['β', 'γ'],
        'latex' : r'$B(z) = \beta z^\gamma$',
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# 2. Estimation de B̂ depuis les données brutes
# ═══════════════════════════════════════════════════════════════════════════

def estimate_B_hat(T: np.ndarray,
                   entry_times: Optional[np.ndarray] = None,
                   n_grid: int = 200,
                   quantile: float = 0.97) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Estime B̂(z) par Tikhonov p=1 + GCV.

    Returns
    -------
    grid   : grille de discrétisation
    B_hat  : estimation de B sur la grille
    alpha  : paramètre de régularisation sélectionné
    """
    n    = len(T)
    eps  = 1.0 / np.sqrt(n)
    grid = DirectProblemSolver.grid_from_data(T, n_grid, quantile)
    A    = DirectProblemSolver(grid).integration_matrix

    na    = NelsonAalanEstimator().fit(T, entry_times=entry_times)
    H_eps = na.smooth(grid, sigma_grid=2.0)

    tikh  = TikhonovRegularizer(A, p=1).fit(H_eps)
    alpha = DiscrepancyPrinciple(tau=1.05).select(tikh, eps)
    B_hat = tikh.predict(alpha)

    # Nettoyer les valeurs négatives et NaN
    B_hat = np.where(np.isfinite(B_hat) & (B_hat >= 0), B_hat, 0.0)

    return grid, B_hat, float(alpha)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Fit paramétrique sur B̂
# ═══════════════════════════════════════════════════════════════════════════

def fit_family(grid: np.ndarray, B_hat: np.ndarray,
               family_name: str) -> Dict:
    """
    Fitte une famille paramétrique sur B̂ par moindres carrés.

    Returns
    -------
    dict avec : params, param_labels, r2, l2_resid, aic, B_fit
    """
    fam    = FAMILIES[family_name]
    func   = fam['func']
    p0     = fam['p0']
    bounds = fam['bounds']
    labels = fam['labels']

    # Masque : exclure les extremités bruitées
    mask = (grid > grid[5]) & (grid < grid[-5]) & np.isfinite(B_hat) & (B_hat > 0)
    z_fit = grid[mask]
    B_fit_data = B_hat[mask]

    if np.sum(mask) < 10:
        return {'error': 'Pas assez de points valides'}

    try:
        popt, pcov = curve_fit(func, z_fit, B_fit_data,
                               p0=p0, bounds=bounds,
                               maxfev=10_000)
    except Exception as e:
        return {'error': str(e)}

    B_pred = func(z_fit, *popt)

    # Vérification : si B_pred contient des NaN/Inf ou est aberrant, rejeter
    if not np.all(np.isfinite(B_pred)) or np.max(np.abs(B_pred)) > 1e6:
        return {'error': f'Fit divergent : max(|B_pred|)={np.max(np.abs(B_pred)):.2e}'}

    # R²
    ss_res = np.sum((B_fit_data - B_pred) ** 2)
    ss_tot = np.sum((B_fit_data - np.mean(B_fit_data)) ** 2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-15)

    # Rejeter si R² trop négatif (fit complètement raté)
    if r2 < -10.0:
        return {'error': f'R² = {r2:.2f} : fit non convergé'}

    # Résidu L²
    l2_resid = float(np.sqrt(np.trapezoid((B_fit_data - B_pred) ** 2, z_fit))
                     / np.sqrt(np.trapezoid(B_fit_data ** 2, z_fit)))

    # AIC (log-vraisemblance gaussienne)
    n_pts = np.sum(mask)
    k_par = len(popt)
    sigma2 = ss_res / n_pts
    aic = n_pts * np.log(max(sigma2, 1e-30)) + 2 * k_par

    # B fittée sur toute la grille
    B_fitted = func(grid, *popt)
    if not np.all(np.isfinite(B_fitted)):
        B_fitted = np.where(np.isfinite(B_fitted), B_fitted, 0.0)

    return {
        'params'      : {lab: round(float(v), 6) for lab, v in zip(labels, popt)},
        'param_labels': labels,
        'r2'          : round(float(r2), 5),
        'l2_resid'    : round(float(l2_resid), 5),
        'aic'         : round(float(aic), 3),
        'B_fit'       : B_fitted,
        'mask'        : mask,
    }


def fit_all_families(grid: np.ndarray, B_hat: np.ndarray) -> Dict:
    """Fitte toutes les familles et retourne les résultats triés par AIC."""
    results = {}
    for name in FAMILIES:
        results[name] = fit_family(grid, B_hat, name)

    # Trouver le meilleur modèle (AIC minimal, parmi les fits valides avec R²>0)
    valid = {k: v for k, v in results.items()
             if 'error' not in v and v.get('r2', -999) > 0}
    if valid:
        best = min(valid, key=lambda k: valid[k]['aic'])
        results['_best'] = best

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 4. Bootstrap
# ═══════════════════════════════════════════════════════════════════════════

def bootstrap_fit(T: np.ndarray,
                  family_name: str = 'generalized_gamma',
                  n_boot: int = 200,
                  entry_times: Optional[np.ndarray] = None,
                  seed: int = 42) -> Dict:
    """
    Bootstrap sur les T_i pour quantifier l'incertitude sur les paramètres.

    Pour chaque répétition :
      1. Tire n valeurs avec remise depuis T
      2. Estime B̂ par Tikhonov p=1 + GCV
      3. Fitte la famille paramétrique choisie
      4. Stocke les paramètres

    Returns
    -------
    dict avec : params_boots, ci_95, ci_50, mean, std
    """
    rng    = np.random.default_rng(seed)
    fam    = FAMILIES[family_name]
    labels = fam['labels']
    n      = len(T)

    all_params = {lab: [] for lab in labels}
    n_failed   = 0

    print(f'    Bootstrap {family_name}  (n_boot={n_boot}) ...', end='', flush=True)
    t0 = time.perf_counter()

    for i in range(n_boot):
        # Rééchantillonnage avec remise
        idx  = rng.integers(0, n, n)
        T_b  = T[idx]
        Xub_b = entry_times[idx] if entry_times is not None else None

        try:
            grid_b, B_hat_b, _ = estimate_B_hat(T_b, Xub_b)
            fit_b = fit_family(grid_b, B_hat_b, family_name)
            if 'error' in fit_b:
                n_failed += 1
                continue
            for lab in labels:
                all_params[lab].append(fit_b['params'][lab])
        except Exception:
            n_failed += 1

        if (i + 1) % 50 == 0:
            print(f' {i+1}', end='', flush=True)

    dt = time.perf_counter() - t0
    print(f'  ({dt:.1f}s, {n_failed} échecs)')

    # Intervalles de confiance
    ci = {}
    for lab in labels:
        vals = np.array(all_params[lab])
        if len(vals) < 10:
            ci[lab] = {'mean': float('nan'), 'std': float('nan'),
                       'ci_95': [float('nan'), float('nan')],
                       'ci_50': [float('nan'), float('nan')]}
        else:
            ci[lab] = {
                'mean' : round(float(np.mean(vals)),   6),
                'std'  : round(float(np.std(vals)),    6),
                'ci_95': [round(float(np.percentile(vals,  2.5)), 6),
                          round(float(np.percentile(vals, 97.5)), 6)],
                'ci_50': [round(float(np.percentile(vals, 25.0)), 6),
                          round(float(np.percentile(vals, 75.0)), 6)],
                'all'  : vals.tolist(),
            }

    return {
        'family'  : family_name,
        'n_boot'  : n_boot,
        'n_failed': n_failed,
        'params'  : ci,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5. Figure
# ═══════════════════════════════════════════════════════════════════════════

COLORS_FAM = {
    'weibull'           : '#d62728',   # rouge
    'gamma_generalized' : '#8c564b',   # marron
    'weibull2'          : '#ff7f0e',   # orange
    'constant'          : '#2ca02c',   # vert
    'power'             : '#9467bd',   # violet
}


def plot_fit_results(grid: np.ndarray, B_hat: np.ndarray,
                     fits: Dict, boot: Dict,
                     dataset_name: str) -> plt.Figure:
    """
    Figure 2×2 :
      Haut-gauche  : B̂ + toutes les familles fittées
      Haut-droite  : zoom sur le meilleur fit + IC bootstrap
      Bas-gauche   : barplot R² des familles
      Bas-droite   : distribution bootstrap de k et σ (gamma généralisé)
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f'Fit paramétrique de B̂(z)  —  {dataset_name}  (modèle incrément)',
        fontsize=12
    )

    best_name = fits.get('_best', 'generalized_gamma')

    # ── Haut-gauche : toutes les familles ────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(grid, B_hat, 'k-', lw=2.5, label='B̂ Tikhonov p=1', zorder=5)
    for fname, fres in fits.items():
        if fname.startswith('_') or 'error' in fres:
            continue
        B_f = fres['B_fit']
        r2  = fres['r2']
        lbl = f"{fname}  (R²={r2:.3f})"
        ax.plot(grid, B_f, '--', color=COLORS_FAM.get(fname, 'gray'),
                lw=1.8, label=lbl)
    ax.set_xlabel('Incrément z')
    ax.set_ylabel('B(z)')
    ax.set_title('B̂ et fits paramétriques')
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)

    # ── Haut-droite : meilleur fit + IC bootstrap ─────────────────────────────
    ax = axes[0, 1]
    ax.plot(grid, B_hat, 'k-', lw=2.5, label='B̂ Tikhonov p=1', zorder=5)

    if best_name in fits and 'error' not in fits[best_name]:
        best_fit = fits[best_name]
        ax.plot(grid, best_fit['B_fit'], '--',
                color=COLORS_FAM.get(best_name, 'red'),
                lw=2.5, label=f'Meilleur fit : {best_name}')

        # Bande bootstrap IC 95%
        if boot and 'params' in boot:
            fam_func = FAMILIES[best_name]['func']
            labels   = FAMILIES[best_name]['labels']
            boots_ci = boot['params']

            # Générer des courbes bootstrap
            B_boots = []
            for lab in labels:
                if 'all' not in boots_ci.get(lab, {}):
                    break
            else:
                n_draw = min(500, len(boots_ci[labels[0]]['all']))
                idx_draw = np.random.choice(
                    len(boots_ci[labels[0]]['all']), n_draw, replace=False)
                for i in idx_draw:
                    p = [boots_ci[lab]['all'][i] for lab in labels]
                    try:
                        B_b = fam_func(grid, *p)
                        if np.all(np.isfinite(B_b)) and np.all(B_b >= 0):
                            B_boots.append(B_b)
                    except Exception:
                        pass

                if B_boots:
                    B_arr = np.array(B_boots)
                    lo = np.percentile(B_arr, 2.5, axis=0)
                    hi = np.percentile(B_arr, 97.5, axis=0)
                    ax.fill_between(grid, lo, hi, alpha=0.2,
                                    color=COLORS_FAM.get(best_name, 'red'),
                                    label='IC 95% bootstrap')

    ax.set_xlabel('Incrément z')
    ax.set_ylabel('B(z)')
    ax.set_title(f'Meilleur fit ({best_name}) + IC 95%')
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)

    # ── Bas-gauche : barplot R² ───────────────────────────────────────────────
    ax = axes[1, 0]
    names_valid = [k for k in fits if not k.startswith('_') and 'error' not in fits[k]]
    r2_vals     = [fits[k]['r2']       for k in names_valid]
    aic_vals    = [fits[k]['aic']      for k in names_valid]
    l2_vals     = [fits[k]['l2_resid'] for k in names_valid]
    colors_bar  = [COLORS_FAM.get(k, 'gray') for k in names_valid]

    x = np.arange(len(names_valid))
    ax.bar(x, r2_vals, color=colors_bar, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names_valid, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('R²')
    ax.set_title('Qualité du fit (R²)')
    ax.set_ylim(0, 1.05)
    for i, (r2, l2) in enumerate(zip(r2_vals, l2_vals)):
        ax.text(i, r2 + 0.01, f'{r2:.3f}', ha='center', fontsize=8)

    # ── Bas-droite : distribution bootstrap de k et σ ────────────────────────
    ax = axes[1, 1]
    if boot and 'params' in boot:
        bp = boot['params']
        labels_boot = list(bp.keys())
        n_params    = len(labels_boot)
        colors_hist = ['#d62728', '#1f77b4', '#2ca02c']

        for i, lab in enumerate(labels_boot):
            if 'all' not in bp.get(lab, {}):
                continue
            vals = np.array(bp[lab]['all'])
            ci95 = bp[lab]['ci_95']
            mu   = bp[lab]['mean']
            ax2  = ax if i == 0 else ax.twinx()
            ax2.hist(vals, bins=30, alpha=0.5,
                     color=colors_hist[i % len(colors_hist)],
                     label=f'{lab}  μ={mu:.3f}  IC95=[{ci95[0]:.3f}, {ci95[1]:.3f}]')
            ax2.axvline(mu, color=colors_hist[i % len(colors_hist)],
                        lw=2, ls='--')
            if i == 0:
                ax2.set_xlabel('Valeur du paramètre')
                ax2.set_ylabel(f'Comptage  ({lab})')
                ax2.legend(fontsize=8, loc='upper left')
            else:
                ax2.set_ylabel(f'Comptage  ({lab})', color=colors_hist[i])
                ax2.legend(fontsize=8, loc='upper right')
    ax.set_title(f'Distribution bootstrap  —  {best_name}')

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 6. Pipeline principal
# ═══════════════════════════════════════════════════════════════════════════

def analyze_dataset_fit(ds, n_boot: int = 200,
                        show_plot: bool = True) -> Dict:
    """
    Pipeline complet pour un dataset :
      1. Estime B̂(z) — modèle incrément, Tikhonov p=1
      2. Fitte les 4 familles
      3. Bootstrap sur le gamma généralisé
    """
    print(f'\n{"═"*58}')
    print(f'  Dataset : {ds.name}  (n={ds.n}, τ={ds.tau:.1f} min)')
    print(f'{"─"*58}')

    # ── 1. Estimation de B̂ ──────────────────────────────────────────────────
    print('  [1/3] Estimation B̂(z) par Tikhonov p=1 + GCV ...', end='', flush=True)
    grid, B_hat, alpha = estimate_B_hat(ds.increment)
    print(f'  α={alpha:.3e}')

    # ── 2. Fits paramétriques ────────────────────────────────────────────────
    print('  [2/3] Fit des 4 familles paramétriques ...')
    fits = fit_all_families(grid, B_hat)

    print(f'\n  {"Famille":22s}  {"R²":>8}  {"L² résidu":>10}  {"AIC":>10}  {"Paramètres"}')
    print(f'  {"─"*70}')
    for fname, fres in fits.items():
        if fname.startswith('_'):
            continue
        if 'error' in fres:
            print(f'  {fname:22s}  ERREUR: {fres["error"]}')
            continue
        params_str = '  '.join(f'{k}={v:.4f}' for k, v in fres['params'].items())
        star = ' ★' if fname == fits.get('_best') else ''
        print(f'  {fname:22s}  {fres["r2"]:>8.4f}  {fres["l2_resid"]:>10.5f}'
              f'  {fres["aic"]:>10.2f}  {params_str}{star}')

    # ── 3. Bootstrap ─────────────────────────────────────────────────────────
    print(f'\n  [3/3] Bootstrap (n_boot={n_boot}) ...')
    boot_weibull = bootstrap_fit(ds.increment, family_name='weibull',
                                 n_boot=n_boot, seed=42)
    boot_gg      = bootstrap_fit(ds.increment, family_name='gamma_generalized',
                                 n_boot=n_boot, seed=42)

    # Affichage IC Weibull
    print(f'\n  Bootstrap — Weibull(k, σ) :')
    for lab, ci in boot_weibull['params'].items():
        print(f'    {lab:6s}  μ={ci["mean"]:.4f}  σ={ci["std"]:.4f}'
              f'  IC95=[{ci["ci_95"][0]:.4f}, {ci["ci_95"][1]:.4f}]')

    # Affichage IC Gamma généralisé
    print(f'\n  Bootstrap — Gamma généralisé GG(a, d, p) :')
    for lab, ci in boot_gg['params'].items():
        print(f'    {lab:6s}  μ={ci["mean"]:.4f}  σ={ci["std"]:.4f}'
              f'  IC95=[{ci["ci_95"][0]:.4f}, {ci["ci_95"][1]:.4f}]')

    # ── Figure ───────────────────────────────────────────────────────────────
    if show_plot:
        fig = plot_fit_results(grid, B_hat, fits, boot_weibull, ds.name)
        out = FIG_DIR / f'fit_B_{ds.name}.png'
        fig.savefig(out, dpi=130, bbox_inches='tight')
        plt.close(fig)
        print(f'\n  Figure → {out.name}')

    # Sérialiser les résultats (sans les arrays numpy)
    fits_serial = {}
    for fname, fres in fits.items():
        if fname.startswith('_'):
            fits_serial[fname] = fres
            continue
        if 'error' in fres:
            fits_serial[fname] = fres
            continue
        fits_serial[fname] = {
            k: v for k, v in fres.items()
            if k not in ('B_fit', 'mask')
        }

    def _ser_boot(boot):
        return {
            k: ({lab: {kk: vv for kk, vv in ci.items() if kk != 'all'}
                  for lab, ci in v.items()}
                 if k == 'params' else v)
            for k, v in boot.items()
        }

    return {
        'dataset'           : ds.name,
        'n'                 : ds.n,
        'tau'               : ds.tau,
        'alpha'             : alpha,
        'fits'              : fits_serial,
        'bootstrap_weibull' : _ser_boot(boot_weibull),
        'bootstrap_gg'      : _ser_boot(boot_gg),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 7. Point d'entrée
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Fit paramétrique de B̂(z) sur données réelles Eric'
    )
    p.add_argument('--dataset', nargs='+', default=['Eric1002', 'Eric1009'],
                   help='Datasets à analyser (défaut: Eric1002 Eric1009)')
    p.add_argument('--n-boot', type=int, default=200,
                   help='Nombre de répétitions bootstrap (défaut: 200)')
    p.add_argument('--no-plots', action='store_true')
    p.add_argument('--data-dir', default=None,
                   help='Dossier des données réelles')
    return p.parse_args()


def main():
    args = parse_args()
    t0   = time.perf_counter()

    # Dossier de données
    if args.data_dir:
        data_dir = args.data_dir
    else:
        for candidate in ['real_data_analysis', 'data', '/data']:
            if Path(candidate).exists():
                data_dir = candidate
                break
        else:
            data_dir = 'real_data_analysis'

    print(f'\n{"═"*58}')
    print(f'  Fit paramétrique de B̂(z) — données réelles')
    print(f'  Datasets : {args.dataset}')
    print(f'  n_boot   : {args.n_boot}')
    print(f'  Données  : {data_dir}')
    print(f'{"═"*58}')

    # Chargement des datasets
    all_ds = load_all_datasets(data_dir)
    target = {k: v for k, v in all_ds.items()
              if any(d.lower() in k.lower() for d in args.dataset)}

    if not target:
        print(f'  Aucun dataset trouvé parmi {list(all_ds.keys())}')
        sys.exit(1)

    # Analyse de chaque dataset
    all_results = {}
    for name, ds in target.items():
        res = analyze_dataset_fit(ds, n_boot=args.n_boot,
                                  show_plot=not args.no_plots)
        all_results[name] = res

    # Résumé comparatif si plusieurs datasets
    if len(all_results) > 1:
        print(f'\n{"═"*58}')
        print('  RÉSUMÉ COMPARATIF')
        print(f'{"─"*58}')
        print(f'  {"Dataset":15s}  {"Meilleur":22s}  {"k":>8}  {"σ":>8}  {"R²":>8}')
        print(f'  {"─"*56}')
        for name, res in all_results.items():
            best  = res['fits'].get('_best', '?')
            wb    = res['fits'].get('weibull', {})
            k_val = wb.get('params', {}).get('k', float('nan'))
            s_val = wb.get('params', {}).get('σ', float('nan'))
            if isinstance(s_val, float) and np.isnan(s_val):
                # Essayer avec la clé unicode sigma
                s_val = wb.get('params', {}).get('\u03c3', float('nan'))
            r2    = wb.get('r2', float('nan'))
            print(f'  {name:15s}  {best:22s}  {k_val:>8.4f}  {s_val:>8.4f}  {r2:>8.4f}')
        print(f'{"═"*58}')

    # Sauvegarde JSON
    out_path = RES_DIR / 'fit_B_parametric.json'

    def _serial(x):
        if isinstance(x, (np.floating, np.integer)): return float(x)
        if isinstance(x, np.ndarray): return x.tolist()
        return str(x)

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=_serial)
    print(f'\n  Résultats → {out_path}')
    print(f'  Temps total : {time.perf_counter() - t0:.1f} s\n')


if __name__ == '__main__':
    main()
