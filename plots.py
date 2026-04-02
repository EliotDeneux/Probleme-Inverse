"""
plots.py — Toutes les fonctions de visualisation.

Ce fichier est à la RACINE du projet (même niveau que run_all.py).

Organisation :
─────────────
  1. plot_direct_problem      : B, H, S, f théoriques + histogramme
  2. plot_estimation_results  : B̂ vs B_true, erreur absolue
  3. plot_alpha_selection     : résidu / GCV / courbe-L / barplot erreurs
  4. plot_convergence         : erreur L² vs n + taux théoriques
  5. plot_error_heatmap       : heatmap taux × méthodes
  6. plot_picard_criterion    : coefficients spectraux de Picard
  7. plot_global_summary      : vue d'ensemble tous modèles

Toutes les fonctions acceptent un paramètre optionnel save_path pour
sauvegarder la figure en PNG. Elles retournent toujours la figure matplotlib.
"""

from __future__ import annotations
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import EstimationResult, METHODS, theoretical_convergence_rate

# ── Palette et labels cohérents ─────────────────────────────────────────────
_COLORS = {
    'kde'        : '#e41a1c',
    'tsvd'       : '#377eb8',
    'tikhonov_0' : '#4daf4a',
    'tikhonov_p0': '#4daf4a',
    'tikhonov_1' : '#ff7f00',
    'tikhonov_p1': '#ff7f00',
    'tikhonov_2' : '#984ea3',
    'tikhonov_p2': '#984ea3',
}
_LABELS = {
    'kde'        : 'KDE  (f/S)',
    'tsvd'       : 'SVD tronquee',
    'tikhonov_0' : 'Tikhonov p=0',
    'tikhonov_p0': 'Tikhonov p=0',
    'tikhonov_1' : 'Tikhonov p=1',
    'tikhonov_p1': 'Tikhonov p=1',
    'tikhonov_2' : 'Tikhonov p=2',
    'tikhonov_p2': 'Tikhonov p=2',
}
_X_LABELS = {
    'age'      : 'Age a [min]',
    'size'     : 'Taille x [um]',
    'increment': 'Increment z [um]',
}

plt.rcParams.update({
    'figure.dpi'       : 110,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'font.size'        : 10,
    'axes.titlesize'   : 11,
    'font.family'      : 'DejaVu Sans',
})

def _savefig(fig: plt.Figure, save_path: Optional[str]):
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches='tight')


# ═══════════════════════════════════════════════════════════════════════════
# 1. Problème direct
# ═══════════════════════════════════════════════════════════════════════════

def plot_direct_problem(model: str, rate_name: str,
                         data_dir: str = 'data',
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Vérifie la cohérence simulateur / théorie (problème direct).

    Panneau (2 x 2) :
      - Haut-gauche  : taux de division B(t) théorique
      - Haut-droite  : hasard cumulé H(t) = PsiB (opérateur direct)
      - Bas-gauche   : densité f(t) + histogramme empirique + stat KS
      - Bas-droite   : survie S(t) théorique + survie empirique 1-F_n
    """
    from evaluate import verify_direct_problem
    from src import KNOWN_RATES

    result    = verify_direct_problem(model, rate_name, data_dir)
    theory    = result['theory']
    rate_spec = KNOWN_RATES.get((model, rate_name))
    xlabel    = _X_LABELS.get(model, 't')
    g         = theory['grid']

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    desc = rate_spec.description if rate_spec else rate_name
    fig.suptitle(f'Probleme direct  --  modele {model}, taux {rate_name}\n{desc}',
                 fontsize=12, y=1.01)

    # B(t) — taux de division
    ax = axes[0, 0]
    ax.plot(g, theory['B'], 'k-', lw=2.5)
    ax.set_xlabel(xlabel); ax.set_ylabel('B(t)')
    ax.set_title('Taux de division B (a inverser)')

    # H(t) — opérateur Psi appliqué à B
    ax = axes[0, 1]
    ax.plot(g, theory['H'], color='steelblue', lw=2.5)
    ax.set_xlabel(xlabel); ax.set_ylabel('H(t)')
    ax.set_title('Hasard cumule H = Psi B  (operateur direct)')

    # f(t) + histogramme
    ax = axes[1, 0]
    w = result['hist_x'][1] - result['hist_x'][0] if len(result['hist_x']) > 1 else 1.0
    ax.bar(result['hist_x'], result['hist_y'], width=w,
           alpha=0.35, color='steelblue', label='Donnees simulees')
    ax.plot(g, theory['f'], 'k-', lw=2.5, label='f(t) = B(t)S(t) theorique')
    ks = result['ks_stat']
    ax.set_xlabel(xlabel); ax.set_ylabel('Densite')
    ax.set_title(f'Densite f  --  KS stat = {ks:.4f}'
                 + ('' if model != 'size' else '  (eleve attendu: troncature)'))
    ax.legend(fontsize=9)

    # S(t) + survie empirique
    ax = axes[1, 1]
    ax.plot(g, theory['S'], 'k-', lw=2.5, label='S(t) theorique')
    ax.plot(g, 1.0 - result['F_emp'], 'r--', lw=1.8, label='1 - F_n empirique')
    ax.set_xlabel(xlabel); ax.set_ylabel('S(t) = P(T > t)')
    ax.set_title('Survie S')
    ax.legend(fontsize=9)

    plt.tight_layout()
    _savefig(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 2. Résultats d'estimation
# ═══════════════════════════════════════════════════════════════════════════

def plot_estimation_results(results: Dict[str, EstimationResult],
                             title: str = '',
                             save_path: Optional[str] = None) -> plt.Figure:
    """
    Compare B_true et B_hat pour toutes les méthodes.

    Panneau gauche  : B_hat(t) pour chaque méthode + B_true en noir
    Panneau droit   : |B_hat - B_true| en echelle log (erreur absolue)
    """
    if not results:
        fig, ax = plt.subplots(); ax.set_title('Aucun resultat'); return fig

    any_res = next(iter(results.values()))
    B_true  = any_res.B_true
    grid    = any_res.grid
    xlabel  = _X_LABELS.get(any_res.model, 't')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title or 'Estimation de B -- comparaison des methodes', fontsize=12)

    # Panneau gauche : estimation
    ax = axes[0]
    if B_true is not None:
        ax.plot(grid, B_true, color='black', lw=3, ls='-', label='B vrai', zorder=10)
    for method, res in results.items():
        mask = np.isfinite(res.B_hat)
        if not np.any(mask):
            continue
        c   = _COLORS.get(method, 'gray')
        lbl = _LABELS.get(method, method)
        l2  = res.l2_error()
        a   = res.alpha
        extra = f'  (L2={l2:.3f})' if l2 is not None else ''
        ax.plot(grid[mask], res.B_hat[mask], color=c, lw=1.8,
                label=f'{lbl}{extra}')
    ax.set_xlabel(xlabel); ax.set_ylabel('B(t)')
    ax.set_title('Taux de division estimes')
    ax.legend(fontsize=8, loc='best')

    # Panneau droit : erreur absolue
    ax2 = axes[1]
    if B_true is not None:
        for method, res in results.items():
            mask = np.isfinite(res.B_hat) & np.isfinite(B_true)
            if not np.any(mask):
                continue
            err = np.abs(res.B_hat[mask] - B_true[mask])
            c   = _COLORS.get(method, 'gray')
            lbl = _LABELS.get(method, method)
            ax2.semilogy(grid[mask], err + 1e-10, color=c, lw=1.8, label=lbl)
        ax2.set_xlabel(xlabel); ax2.set_ylabel('|B_hat - B|  (log)')
        ax2.set_title('Erreur absolue pointwise')
        ax2.legend(fontsize=8)
    else:
        ax2.text(0.5, 0.5, 'B_true non disponible',
                 transform=ax2.transAxes, ha='center', va='center')

    plt.tight_layout()
    _savefig(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 3. Sélection du paramètre α
# ═══════════════════════════════════════════════════════════════════════════

def plot_alpha_selection(analysis: Dict,
                          title: str = '',
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Analyse des methodes de selection de alpha (Tikhonov p=0).

    Panneau (2 x 2) :
      - Haut-gauche : residus vs alpha + cible de Morozov
      - Haut-droite : GCV(alpha) normalise
      - Bas-gauche  : courbe en L coloriee par log(alpha)
      - Bas-droite  : barplot des erreurs L2 par methode de selection
    """
    fig = plt.figure(figsize=(13, 9))
    gs  = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.35)
    fig.suptitle(title or 'Selection du parametre de regularisation alpha',
                 fontsize=12)

    ag  = analysis['alpha_grid']
    eps = analysis['epsilon']
    m   = len(analysis['grid'])

    # ── Résidu vs alpha (Morozov) ─────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.loglog(ag, analysis['residuals'], 'b-', lw=2,
              label='Residu ||AB_a - H_eps||')
    target = analysis.get('target_disc', 1.1 * eps * np.sqrt(m))
    ax.axhline(target, color='red', ls='--', lw=1.5,
               label=f'Cible Morozov ({target:.3f})')
    ax.axvline(analysis['alpha_disc'], color='red', ls=':', lw=1.5,
               label=f'alpha_disc = {analysis["alpha_disc"]:.2e}')
    ax.set_xlabel('alpha'); ax.set_ylabel('Residu')
    ax.set_title('Principe de discordance (Morozov)')
    ax.legend(fontsize=8)

    # ── GCV ──────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    gcv  = np.asarray(analysis['gcv_vals'], dtype=float)
    gcv_n = gcv / np.nanmax(gcv) if np.nanmax(gcv) > 0 else gcv
    ax.loglog(ag, gcv_n, color='darkgreen', lw=2)
    ax.axvline(analysis['alpha_gcv'], color='darkgreen', ls=':', lw=1.5,
               label=f'alpha_GCV = {analysis["alpha_gcv"]:.2e}')
    ax.set_xlabel('alpha'); ax.set_ylabel('GCV(alpha) / max')
    ax.set_title('Validation croisee generalisee (GCV)')
    ax.legend(fontsize=8)

    # ── Courbe en L ───────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    lx, ly = analysis['lcurve_x'], analysis['lcurve_y']
    sc = ax.scatter(lx, ly, c=np.log10(ag + 1e-30), cmap='viridis', s=25, zorder=3)
    plt.colorbar(sc, ax=ax, label='log10(alpha)')
    # Marquer l'alpha sélectionné par la courbe en L
    idx_lc = np.argmin(np.abs(ag - analysis['alpha_lcurve']))
    ax.scatter([lx[idx_lc]], [ly[idx_lc]], marker='*', s=200, c='red',
               zorder=5, label=f'alpha_L = {analysis["alpha_lcurve"]:.2e}')
    ax.set_xlabel('log10 ||AB_a - H||'); ax.set_ylabel('log10 ||B_a||')
    ax.set_title('Courbe en L (heuristique)')
    ax.legend(fontsize=8)

    # ── Barplot erreurs ───────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    labels, values, colors = [], [], []

    selectors = [
        ('Discordance', analysis['err_disc'],    'red'),
        ('GCV',         analysis['err_gcv'],     'darkgreen'),
        ('Courbe-L',    analysis['err_lcurve'],  'purple'),
    ]
    palette_s = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, (s, a) in enumerate(sorted(analysis['alpha_apriori'].items())):
        err = analysis['err_apriori'].get(s, float('nan'))
        selectors.append((f'A priori s={s}', err, palette_s[i % len(palette_s)]))

    for lbl, err, col in selectors:
        if not np.isnan(err):
            labels.append(lbl)
            values.append(err)
            colors.append(col)

    if values:
        ax.bar(range(len(labels)), values, color=colors, alpha=0.82)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('Erreur L2 relative')
        ax.set_title(f'Erreur selon la selection de alpha  (n={analysis["n"]})')
        ax.set_yscale('log')

    _savefig(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 4. Convergence erreur L² vs n
# ═══════════════════════════════════════════════════════════════════════════

def plot_convergence(conv_results: List[Dict],
                     theoretical_s: Optional[List[float]] = None,
                     title: str = '',
                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Trace l'erreur L2 relative en fonction de n (log-log).

    Superpose les taux de convergence theoriques du cours :
      O(n^{-s/(2s+1)}) pour B dans Y^s.

    La bande coloree represente +/- un ecart-type (n_repeat repetitions).

    Parameters
    ----------
    conv_results   : liste de dicts issus de convergence_study()
    theoretical_s  : valeurs de s pour les droites theoriques
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(title or 'Convergence de l\'erreur L2 en fonction de n', fontsize=12)

    n_ref = None
    for res in conv_results:
        n_vals = np.asarray(res['n_values'])
        l2_mu  = np.asarray(res['l2_mean'])
        l2_std = np.asarray(res['l2_std'])
        method = res['method']
        rate   = res.get('rate', '')
        model  = res.get('model', '')
        color  = _COLORS.get(method, 'gray')
        label  = _LABELS.get(method, method) + f'  [{model}/{rate}]'

        # Estimer la pente empirique (régression log-log)
        valid = np.isfinite(l2_mu) & (l2_mu > 0)
        if np.sum(valid) >= 3:
            slope = float(np.polyfit(np.log(n_vals[valid]), np.log(l2_mu[valid]), 1)[0])
            label += f'  (pente ≈ {slope:.2f})'

        ax.loglog(n_vals, l2_mu, 'o-', color=color, lw=2.2, ms=6, label=label)
        ax.fill_between(n_vals,
                        np.maximum(l2_mu - l2_std, 1e-6),
                        l2_mu + l2_std,
                        alpha=0.15, color=color)
        n_ref = n_vals

    # Taux théoriques
    if theoretical_s is not None and n_ref is not None:
        ls_styles = ['--', ':', '-.', (0, (3, 1, 1, 1))]
        # Ancrer la constante sur le premier résultat disponible
        if conv_results:
            l2_ref = np.nanmean(conv_results[0]['l2_all'][0])
            n0     = conv_results[0]['n_values'][0]
        else:
            l2_ref, n0 = 1.0, 100

        for i, s in enumerate(theoretical_s):
            rate_theory = theoretical_convergence_rate(n_ref, s=s)
            # Normalisation pour ancrer la courbe théorique au premier point
            C_anc = l2_ref / max(rate_theory[0], 1e-15)
            ax.loglog(n_ref, C_anc * rate_theory,
                      ls=ls_styles[i % len(ls_styles)], color='black', lw=1.5,
                      label=f'Theorie  O(n^(-{s}/{2*s+1:.0f}))')

    ax.set_xlabel('Taille de l\'echantillon n')
    ax.set_ylabel('Erreur ||B_hat - B||_L2 / ||B||_L2')
    ax.legend(fontsize=8, ncol=1)
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    _savefig(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 5. Heatmap de comparaison
# ═══════════════════════════════════════════════════════════════════════════

def plot_error_heatmap(comparison: Dict, model: str,
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Heatmap des erreurs L2 relatives : taux (lignes) x methodes (colonnes).

    Couleur verte = faible erreur, rouge = forte erreur.
    Valeurs affichees dans chaque cellule.
    """
    from evaluate import summary_table

    table, rates, methods = summary_table(comparison)
    n_rates, n_meth = table.shape

    fig, ax = plt.subplots(figsize=(max(8, n_meth * 1.8), max(4, n_rates * 1.1)))

    vmax = float(np.nanpercentile(table, 85))
    im   = ax.imshow(table, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=vmax)

    ax.set_xticks(range(n_meth))
    ax.set_xticklabels([_LABELS.get(m, m) for m in methods],
                        rotation=35, ha='right', fontsize=9)
    ax.set_yticks(range(n_rates))
    ax.set_yticklabels(rates, fontsize=9)

    # Valeurs dans les cellules
    for i in range(n_rates):
        for j in range(n_meth):
            val = table[i, j]
            if np.isfinite(val):
                text_color = 'white' if val > vmax * 0.65 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                        fontsize=8.5, color=text_color, fontweight='bold')

    plt.colorbar(im, ax=ax, label='Erreur L2 relative  (plus fonce = pire)')
    ax.set_title(f'Erreurs L2 relatives -- modele {model}', fontsize=11)
    plt.tight_layout()
    _savefig(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 6. Critère de Picard
# ═══════════════════════════════════════════════════════════════════════════

def plot_picard_criterion(result: EstimationResult,
                           save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Visualise le critere de Picard pour diagnostiquer le mal-positude.

    Panneau gauche : sigma_j (valeurs singulieres de Psi) et |c_j| (coeff Picard)
    Panneau droit  : |c_j| / sigma_j — doit decroitre pour H_eps dans D(Psi†)

    Si le rapport |c_j|/sigma_j croît, les hauts modes sont amplifies par
    l'inversion — c'est le symptome du mal-positude qui necessite la regularisation.
    """
    if 'picard_j' not in result.extras:
        return None

    j_arr  = result.extras['picard_j']
    sigma  = result.extras['picard_sigma']
    c_abs  = result.extras['picard_c']
    eps    = result.noise_level

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(f'Critere de Picard -- {result.model}/{result.rate_name}  '
                 f'(n={result.n_cells})', fontsize=12)

    # Panneau gauche : sigma_j et |c_j|
    ax = axes[0]
    ax.semilogy(j_arr, sigma, 'r-', lw=2, label='sigma_j = 2T/[pi(2j+1)]',
                zorder=3)
    ax.semilogy(j_arr, np.abs(c_abs) + 1e-15, 'b.', ms=4,
                label='|c_j| = |<H_eps, f_j>|', alpha=0.7)
    ax.axhline(eps, color='gray', ls='--', lw=1.2,
               label=f'Niveau de bruit eps={eps:.4f}')
    ax.set_xlabel('Indice j')
    ax.set_title('Valeurs singulieres et coefficients de Picard')
    ax.legend(fontsize=8)

    # Panneau droit : rapport |c_j|/sigma_j
    ax = axes[1]
    ratio = np.abs(c_abs) / (sigma + 1e-30)
    ax.semilogy(j_arr, ratio, 'g.', ms=4, alpha=0.75)
    # Seuil : amplification commence quand |c_j|/sigma_j > 1/eps
    ax.axhline(1.0 / eps, color='red', ls='--', lw=1.5,
               label=f'Seuil instabilite 1/eps = {1/eps:.1f}')
    ax.axhline(1.0, color='gray', ls=':', lw=1.2)
    ax.set_xlabel('Indice j')
    ax.set_ylabel('|c_j| / sigma_j')
    ax.set_title('Diagnostic Picard : |c_j| / sigma_j\n'
                 '(doit decroitre pour stabilite)')
    ax.legend(fontsize=8)

    plt.tight_layout()
    _savefig(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 7. Vue d'ensemble globale
# ═══════════════════════════════════════════════════════════════════════════

def plot_global_summary(all_results: Dict[Tuple, Dict[str, EstimationResult]],
                         methods_to_show: Optional[List[str]] = None,
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Grille de panneaux : B_hat pour les methodes principales,
    un panneau par (modele, taux).

    Parameters
    ----------
    all_results     : {(model, rate): {method: EstimationResult}}
    methods_to_show : sous-ensemble de methodes (defaut : kde, tikhonov_0, tikhonov_1)
    """
    if methods_to_show is None:
        methods_to_show = ['tikhonov_0', 'tikhonov_1', 'kde']

    model_rate_order = [
        ('age',       'constant'), ('age',       'weibull2'), ('age',       'step'),
        ('size',      'constant'), ('size',      'linear'),   ('size',      'power'),
        ('increment', 'constant'), ('increment', 'weibull2'), ('increment', 'step'),
    ]
    # Ne garder que les (modèle, taux) présents dans all_results
    pairs = [p for p in model_rate_order if p in all_results]
    if not pairs:
        fig, ax = plt.subplots()
        ax.set_title('Aucun resultat disponible')
        return fig

    n_rows = len(pairs)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 2.9 * n_rows),
                             squeeze=False)

    for row, (model, rate) in enumerate(pairs):
        results = all_results[(model, rate)]
        ax = axes[row, 0]
        xlabel = _X_LABELS.get(model, 't')

        # B_true en premier plan
        any_res = next((r for r in results.values() if r.B_true is not None), None)
        if any_res is not None and any_res.B_true is not None:
            ax.plot(any_res.grid, any_res.B_true, 'k-', lw=3,
                    label='B vrai', zorder=10)

        for method in methods_to_show:
            res = results.get(method)
            if res is None:
                continue
            mask = np.isfinite(res.B_hat)
            if not np.any(mask):
                continue
            c    = _COLORS.get(method, 'gray')
            lbl  = _LABELS.get(method, method)
            l2   = res.l2_error()
            sfx  = f'  (L2={l2:.3f})' if l2 is not None else ''
            ax.plot(res.grid[mask], res.B_hat[mask], color=c, lw=1.6,
                    label=f'{lbl}{sfx}')

        # Annotation
        ax.set_title(f'{model} / {rate}', fontsize=10, loc='left', pad=3)
        ax.set_ylabel('B(t)')
        if row == n_rows - 1:
            ax.set_xlabel(xlabel)
        ax.legend(fontsize=7, loc='upper right', ncol=min(len(methods_to_show)+1, 4))

    fig.suptitle('Estimation de B  --  vue d\'ensemble', fontsize=13, y=1.002)
    plt.tight_layout()
    _savefig(fig, save_path)
    return fig
