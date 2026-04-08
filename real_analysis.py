"""
real_analysis.py — Estimation du taux B sur les données réelles et conclusions.

Pour chaque dataset, on estime B par les 3 modèles (âge, taille, incrément)
avec les méthodes Tikhonov p=0, p=1 et KDE.

Stratégie d'adaptation au données réelles :
─────────────────────────────────────────────
1. K est estimé directement depuis les données : K = mean(log(sd/sb)/ad)
2. Les grilles sont construites sur les données réelles (quantiles)
3. Le niveau de bruit ε = 1/√n est estimé par la taille de l'échantillon
4. On compare les 3 modèles via leur cohérence et l'interprétation biologique
"""

from __future__ import annotations
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import (
    DirectProblemSolver, NelsonAalanEstimator, TikhonovRegularizer,
    TruncatedSVD, DiscrepancyPrinciple, GeneralizedCrossValidation,
    KDEHazardEstimator,
)
from real_data import CellDataset


# ── Style ────────────────────────────────────────────────────────────────────
COLORS = {
    'tikhonov_0': '#4daf4a',
    'tikhonov_1': '#ff7f00',
    'kde'       : '#e41a1c',
}
LABELS = {
    'tikhonov_0': 'Tikhonov p=0',
    'tikhonov_1': 'Tikhonov p=1',
    'kde'       : 'KDE (f/S)',
}
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False,
                     'font.size': 10, 'figure.dpi': 110})


# ═══════════════════════════════════════════════════════════════════════════
# Estimation de B pour un modèle donné
# ═══════════════════════════════════════════════════════════════════════════

def estimate_B_real(observations: np.ndarray,
                    entry_times: Optional[np.ndarray] = None,
                    n_grid: int = 200,
                    quantile: float = 0.97,
                    methods: list = ('tikhonov_0', 'tikhonov_1', 'kde'),
                    ) -> Dict[str, dict]:
    """
    Estime B par plusieurs méthodes sur des observations réelles.

    Parameters
    ----------
    observations : T_i (âges, incréments ou tailles division)
    entry_times  : X_ub (seulement pour modèle taille, troncature gauche)
    n_grid, quantile : paramètres de la grille

    Returns
    -------
    {method: {'grid': array, 'B_hat': array, 'alpha': float,
              'H_na': array, 'H_smooth': array}}
    """
    T    = np.asarray(observations, dtype=float)
    n    = len(T)
    eps  = 1.0 / np.sqrt(n)
    grid = DirectProblemSolver.grid_from_data(T, n_grid, quantile)

    # Nelson-Aalen
    na      = NelsonAalanEstimator().fit(T, entry_times=entry_times)
    H_raw   = na.predict(grid)
    H_eps   = na.smooth(grid, sigma_grid=3.0)

    # Matrice d'intégration
    solver = DirectProblemSolver(grid)
    A      = solver.integration_matrix

    results = {}

    if 'tikhonov_0' in methods or 'tikhonov_1' in methods:
        for p in [0, 1]:
            key = f'tikhonov_{p}'
            if key not in methods:
                continue
            tikh  = TikhonovRegularizer(A, p=p).fit(H_eps)
            alpha = DiscrepancyPrinciple(tau=1.05).select(tikh, eps)
            B_hat = tikh.predict(alpha)
            # Contrainte de positivité par rectification (B ≥ 0)
            B_hat = np.maximum(B_hat, 0.0)
            results[key] = {
                'grid': grid, 'B_hat': B_hat, 'alpha': alpha,
                'H_na': H_raw, 'H_smooth': H_eps,
                'residual': tikh.residual(alpha), 'n': n, 'eps': eps,
            }

    if 'kde' in methods:
        kde_est = KDEHazardEstimator(bandwidth='silverman', tail_clip=0.04)
        kde_est.fit(T, entry_times=entry_times)
        B_hat = kde_est.predict_B(grid)
        results['kde'] = {
            'grid': grid, 'B_hat': B_hat,
            'alpha': kde_est.kde.bandwidth_value,
            'H_na': H_raw, 'H_smooth': H_eps, 'n': n, 'eps': eps,
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Analyse complète d'un dataset
# ═══════════════════════════════════════════════════════════════════════════

def analyze_dataset(ds: CellDataset) -> Dict[str, Dict]:
    """
    Lance les 3 modèles (âge, taille, incrément) sur un CellDataset.

    Returns
    -------
    {'age': {...}, 'size': {...}, 'increment': {...}}
    """
    print(f'\n  {ds.name}  (n={ds.n}, τ={ds.tau:.1f} min)')

    results = {}

    # Modèle âge
    print('    [age]      ', end='', flush=True)
    results['age'] = estimate_B_real(ds.ad)
    print('OK')

    # Modèle taille (avec troncature gauche X_ub)
    print('    [size]     ', end='', flush=True)
    results['size'] = estimate_B_real(ds.sd, entry_times=ds.sb)
    print('OK')

    # Modèle incrément
    print('    [increment]', end='', flush=True)
    results['increment'] = estimate_B_real(ds.increment)
    print('OK')

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════════════

def plot_dataset_overview(ds: CellDataset, save_path: Optional[str] = None):
    """
    Vue d'ensemble exploratoire d'un dataset : distributions empiriques,
    corrélations et estimation de K.
    """
    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.35)
    fig.suptitle(f'{ds.label}  (n={ds.n},  K={ds.K:.5f}/min,  τ={ds.tau:.1f} min)',
                 fontsize=12)

    # Histogramme âges
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(ds.ad, bins=40, color='steelblue', alpha=0.7, density=True)
    ax.axvline(ds.ad.mean(), color='red', ls='--', lw=1.5,
               label=f'moy={ds.ad.mean():.1f} min')
    ax.set_xlabel('Age a division [min]'); ax.set_ylabel('Densite')
    ax.set_title('Distribution des ages'); ax.legend(fontsize=8)

    # Histogramme taille naissance
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(ds.sb, bins=40, color='darkorange', alpha=0.7, density=True)
    ax.axvline(ds.sb.mean(), color='red', ls='--', lw=1.5,
               label=f'moy={ds.sb.mean():.2f} {ds.unit_size}')
    ax.set_xlabel(f'Taille naissance [{ds.unit_size}]')
    ax.set_title('Taille a la naissance'); ax.legend(fontsize=8)

    # Histogramme incrément
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(ds.increment, bins=40, color='mediumpurple', alpha=0.7, density=True)
    ax.axvline(ds.increment.mean(), color='red', ls='--', lw=1.5,
               label=f'moy={ds.increment.mean():.2f}')
    ax.set_xlabel(f'Increment Z=sd-sb [{ds.unit_size}]')
    ax.set_title('Increment de taille'); ax.legend(fontsize=8)

    # Corrélation sb vs incrément — diagnostic adder/sizer
    ax = fig.add_subplot(gs[1, 0])
    from scipy.stats import pearsonr
    r, _ = pearsonr(ds.sb, ds.increment)
    # sous-échantillon pour la visualisation
    idx = np.random.choice(len(ds.sb), min(600, len(ds.sb)), replace=False)
    ax.scatter(ds.sb[idx], ds.increment[idx], s=4, alpha=0.5, color='gray')
    # Droite de régression
    p_coef = np.polyfit(ds.sb, ds.increment, 1)
    x_r = np.linspace(ds.sb.min(), ds.sb.max(), 100)
    ax.plot(x_r, np.polyval(p_coef, x_r), 'r-', lw=2, label=f'r={r:.3f}')
    ax.set_xlabel(f'sb [{ds.unit_size}]'); ax.set_ylabel('increment')
    ax.set_title('Diagnostic Adder/Sizer\ncorr(sb, Z)')
    ax.legend(fontsize=9)
    # Annotation
    model_hint = 'Adder (r≈0)' if abs(r) < 0.1 else ('Sizer (r>0)' if r > 0.15 else 'Mixte')
    ax.text(0.97, 0.05, model_hint, transform=ax.transAxes,
            ha='right', fontsize=9, color='darkblue',
            bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.8))

    # Corrélation sb vs sd — vérification doublement
    ax = fig.add_subplot(gs[1, 1])
    r2, _ = pearsonr(ds.sb, ds.sd)
    ax.scatter(ds.sb[idx], ds.sd[idx], s=4, alpha=0.5, color='gray')
    x_r2 = np.linspace(ds.sb.min(), ds.sb.max(), 100)
    p2    = np.polyfit(ds.sb, ds.sd, 1)
    ax.plot(x_r2, np.polyval(p2, x_r2), 'r-', lw=2, label=f'r={r2:.3f}')
    ax.plot(x_r2, 2 * x_r2, 'b--', lw=1.5, label='sd = 2·sb  (doublement)')
    ax.set_xlabel(f'sb [{ds.unit_size}]'); ax.set_ylabel(f'sd [{ds.unit_size}]')
    ax.set_title('Taille naissance vs division'); ax.legend(fontsize=8)

    # Estimation de K : distribution de log(sd/sb)/ad par cellule
    ax = fig.add_subplot(gs[1, 2])
    K_cell = np.log(ds.sd / ds.sb) / ds.ad
    ax.hist(K_cell, bins=40, color='teal', alpha=0.7, density=True)
    ax.axvline(ds.K, color='red', ls='--', lw=2,
               label=f'K={ds.K:.5f}/min')
    ax.set_xlabel('K individuel [1/min]'); ax.set_ylabel('Densite')
    ax.set_title(f'Taux de croissance K\n(tau = {ds.tau:.1f} min)'); ax.legend(fontsize=8)

    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return fig


def plot_B_three_models(ds: CellDataset, results: Dict,
                         save_path: Optional[str] = None):
    """
    Taux B estimés pour les 3 modèles (âge, taille, incrément),
    chaque méthode en couleur différente.
    """
    model_labels = {
        'age'      : f'Modele AGE  B(a)  [a en min]',
        'size'     : f'Modele TAILLE  B(x)  [x en {ds.unit_size}]',
        'increment': f'Modele INCREMENT  B(z)  [z en {ds.unit_size}]',
    }
    xlabels = {
        'age'      : 'Age a [min]',
        'size'     : f'Taille x [{ds.unit_size}]',
        'increment': f'Increment z [{ds.unit_size}]',
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Taux de division B estimes — {ds.label}', fontsize=12)

    for ax, model in zip(axes, ['age', 'size', 'increment']):
        model_res = results.get(model, {})
        for method, res in model_res.items():
            g   = res['grid']
            Bh  = res['B_hat']
            c   = COLORS.get(method, 'gray')
            lbl = LABELS.get(method, method)
            mask = np.isfinite(Bh) & (Bh >= 0)
            if np.any(mask):
                ax.plot(g[mask], Bh[mask], color=c, lw=2, label=lbl)

        ax.set_xlabel(xlabels[model])
        ax.set_ylabel('B(t)')
        ax.set_title(model_labels[model])
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return fig


def plot_H_fit(ds: CellDataset, results: Dict,
               save_path: Optional[str] = None):
    """
    Compare Ĥ_NA (données) et H = Ψ B̂ (estimé) pour valider l'ajustement.
    Si l'ajustement est bon → les B̂ captent bien la structure des données.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(f'Ajustement H = PsiB — {ds.label}', fontsize=11)

    for ax, model in zip(axes, ['age', 'size', 'increment']):
        model_res = results.get(model, {})
        any_res   = next(iter(model_res.values()), None)
        if any_res is None:
            continue
        g    = any_res['grid']
        H_na = any_res['H_na']
        ax.plot(g, H_na, 'k-', lw=2.5, label='Nelson-Aalen H', zorder=5)
        ax.plot(g, any_res['H_smooth'], 'k:', lw=1.5, alpha=0.6, label='H lissé', zorder=4)

        for method, res in model_res.items():
            Bh  = res['B_hat']
            c   = COLORS.get(method, 'gray')
            lbl = LABELS.get(method, method)
            # Reconstruire H = ΨB
            H_est = DirectProblemSolver(g).compute_H(np.maximum(Bh, 0))
            ax.plot(g, H_est, color=c, lw=1.8, ls='--', label=f'H(B̂) {lbl}')

        ax.set_xlabel({'age': 'Age [min]', 'size': f'Taille [{ds.unit_size}]',
                       'increment': f'Increment [{ds.unit_size}]'}[model])
        ax.set_ylabel('H(t)'); ax.set_title(f'Modele {model}')
        ax.legend(fontsize=7)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return fig


def plot_all_datasets_comparison(all_ds: Dict, all_res: Dict,
                                  model: str = 'age',
                                  method: str = 'tikhonov_1',
                                  save_path: Optional[str] = None):
    """
    Compare B̂ entre tous les datasets pour un modèle donné.
    Permet d'observer l'effet de la condition de croissance sur la forme de B.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = plt.cm.tab10(np.linspace(0, 1, len(all_ds)))

    for (name, ds), color in zip(all_ds.items(), palette):
        res = all_res.get(name, {}).get(model, {}).get(method)
        if res is None:
            continue
        g    = res['grid']
        Bh   = res['B_hat']
        mask = np.isfinite(Bh) & (Bh >= 0)
        if not np.any(mask):
            continue
        # Normaliser t par τ pour comparer les conditions
        ax.plot(g[mask] / ds.tau, Bh[mask] * ds.tau,
                color=color, lw=2,
                label=f'{ds.name}  (τ={ds.tau:.0f} min)')

    ax.set_xlabel('Age / τ  (normalisé par le temps de doublement)')
    ax.set_ylabel('B(t) × τ  (normalisé)')
    ax.set_title(f'Comparaison inter-datasets — Modele {model}, {LABELS[method]}')
    ax.legend(fontsize=8)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Critères de sélection du modèle
# ═══════════════════════════════════════════════════════════════════════════

def model_selection_criteria(ds: CellDataset, results: Dict) -> Dict:
    """
    Calcule des critères pour comparer les 3 modèles biologiquement.

    Critères :
    1. Résidu H : ||Ψ B̂ - Ĥ||² / ||Ĥ||² — fidélité aux données
    2. Monotonie de B : % de pts où B est croissant / décroissant / constant
       - B croissant → timer (accumulation d'un seuil)
       - B constant  → memoryless (exponentiel)
       - B décroissant → biologiquement douteux
    3. Cohérence biologique du taux estimé moyen E[T] = ∫S(t)dt vs données
    4. Corrélation sb/increment comme proxy du meilleur modèle
    """
    from scipy.stats import pearsonr
    criteria = {}

    # Corrélation sb/incrément (diagnostic du modèle)
    r_adder, _ = pearsonr(ds.sb, ds.increment)

    for model, model_res in results.items():
        best = model_res.get('tikhonov_1') or next(iter(model_res.values()), None)
        if best is None:
            continue
        g  = best['grid']
        Bh = best['B_hat']
        mask = np.isfinite(Bh) & (Bh >= 0)

        # Résidu relatif de H
        H_na  = best['H_na']
        H_est = DirectProblemSolver(g).compute_H(np.maximum(Bh, 0))
        H_sel = H_na[mask]
        H_e   = H_est[mask]
        resid = float(np.sqrt(np.trapezoid((H_est[mask] - H_na[mask])**2, g[mask]))
                      / max(float(np.sqrt(np.trapezoid(H_na[mask]**2, g[mask]))), 1e-10))

        # Monotonie de B
        B_valid   = Bh[mask]
        dB        = np.diff(B_valid)
        pct_incr  = float(np.mean(dB > 0))
        pct_decr  = float(np.mean(dB < 0))

        # Forme estimée
        if   pct_incr > 0.7 : shape = 'croissant (timer/sizer)'
        elif pct_decr > 0.7 : shape = 'decroissant (atypique)'
        elif pct_incr > 0.45: shape = 'quasi-croissant (mixte)'
        else                : shape = 'plat (memoryless/adder)'

        # E[T] estimé
        S_est = np.exp(-np.maximum(H_e, 0))
        E_T   = float(np.trapezoid(S_est, g[mask]))

        if model == 'age':
            E_data = float(np.mean(ds.ad))
        elif model == 'increment':
            E_data = float(np.mean(ds.increment))
        else:
            E_data = float(np.mean(ds.sd))

        criteria[model] = {
            'resid_H_rel': round(resid, 4),
            'B_shape'    : shape,
            'E_T_est'    : round(E_T, 3),
            'E_T_data'   : round(E_data, 3),
            'E_T_error'  : round(abs(E_T - E_data) / max(E_data, 1e-6), 4),
            'pct_incr'   : round(pct_incr, 3),
            'pct_decr'   : round(pct_decr, 3),
        }

    # Ajout du diagnostic global
    criteria['_diagnostics'] = {
        'corr_sb_increment': round(r_adder, 4),
        'model_hint': ('Adder' if abs(r_adder) < 0.10 else
                       'Sizer' if r_adder > 0.20 else 'Mixte'),
        'n': ds.n, 'K': round(ds.K, 6), 'tau': round(ds.tau, 2),
    }

    return criteria


# ═══════════════════════════════════════════════════════════════════════════
# Rapport textuel des conclusions
# ═══════════════════════════════════════════════════════════════════════════

def print_conclusions(ds: CellDataset, criteria: Dict):
    """
    Affiche les conclusions biologiques et mathématiques pour un dataset.
    """
    diag = criteria['_diagnostics']
    print(f'\n{"=" * 60}')
    print(f'  CONCLUSIONS — {ds.name}')
    print(f'{"=" * 60}')
    print(f'  n={ds.n}  K={ds.K:.5f}/min  τ={ds.tau:.1f} min')
    print(f'  corr(sb, Z) = {diag["corr_sb_increment"]:.4f}  → {diag["model_hint"]}')
    print()

    best_model  = min(
        (k for k in criteria if not k.startswith('_')),
        key=lambda m: criteria[m].get('resid_H_rel', 999)
    )

    for model in ['age', 'size', 'increment']:
        c = criteria.get(model)
        if not c:
            continue
        marker = '>>>' if model == best_model else '   '
        print(f'  {marker} [{model:10s}]  '
              f'resid(H)={c["resid_H_rel"]:.4f}  '
              f'E[T]_est={c["E_T_est"]:.2f} vs {c["E_T_data"]:.2f}  '
              f'B: {c["B_shape"]}')

    print(f'\n  Meilleur ajustement (resid H) : {best_model.upper()}')

    # Interprétation biologique
    r = diag['corr_sb_increment']
    print('\n  Interpretation biologique :')
    if abs(r) < 0.10:
        print('  - corr(sb,Z) ≈ 0 → modele ADDER : les cellules ajoutent')
        print('    un increment constant Z independant de leur taille initiale.')
        print('    => B(z) est le taux le plus informatif.')
    elif r > 0.20:
        print('  - corr(sb,Z) > 0 → tendance SIZER : la division depend')
        print('    de la taille absolue atteinte.')
        print('    => B(x) (modele taille) le plus adapte.')
    else:
        print('  - corr(sb,Z) intermediaire → modele MIXTE (adder/sizer).')
        print('    Aucun modele univoque ne s\'impose ; le modele age')
        print('    reste une bonne approximation phenomenologique.')
