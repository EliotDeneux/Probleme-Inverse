"""
run_all.py — Script principal d'exécution de toutes les analyses.

Usage
─────
    python run_all.py                            # tout exécuter (sauf convergence)
    python run_all.py --steps direct             # seulement le problème direct
    python run_all.py --steps inverse alpha
    python run_all.py --model age                # seulement le modèle âge
    python run_all.py --n-max 2000               # sous-échantillonner à 2000 cellules
    python run_all.py --steps convergence        # étude convergence (~5 min)
    python run_all.py --no-plots                 # calculs sans figures

Étapes disponibles (--steps) :
    direct       : problème direct — vérification simulateur vs théorie
    inverse      : estimation de B par toutes les méthodes
    alpha        : analyse des méthodes de sélection de α
    convergence  : étude de la convergence en n  [~5 min]
    summary      : heatmap récapitulative des erreurs

Modèles disponibles (--model) :  age  size  increment  (défaut : tous)

Sorties :
    figures/     : figures PNG
    results/     : métriques JSON
"""

import sys, argparse, json, time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')   # mode non-interactif (pas de fenêtre)
import matplotlib.pyplot as plt
import numpy as np

# ── Résolution des chemins ──────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data' if (ROOT / 'data').exists() else Path('/data')
FIG_DIR  = ROOT / 'figures'
RES_DIR  = ROOT / 'results'
sys.path.insert(0, str(ROOT))

FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

# ── Imports locaux (évaluations et figures depuis la racine) ─────────────────
from src import (
    HazardEstimationPipeline, run_all_methods, METHODS, KNOWN_RATES,
)
from evaluate import (
    compute_errors, verify_direct_problem, compare_all_methods,
    convergence_study, summary_table, analyze_alpha_selection,
)
from plots import (
    plot_direct_problem, plot_estimation_results, plot_alpha_selection,
    plot_convergence, plot_error_heatmap, plot_picard_criterion,
    plot_global_summary,
)

MODEL_RATES = {
    'age'      : ['constant', 'weibull2', 'step'],
    'size'     : ['constant', 'linear',   'power'],
    'increment': ['constant', 'weibull2', 'step'],
}


# ═══════════════════════════════════════════════════════════════════════════
# Utilitaires
# ═══════════════════════════════════════════════════════════════════════════

def _header(title: str):
    print(f'\n{"=" * 62}\n  {title}\n{"=" * 62}')


def _save(fig: plt.Figure, name: str):
    path = FIG_DIR / f'{name}.png'
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'    -> {path.relative_to(ROOT)}')


def _save_json(data, name: str):
    path = RES_DIR / f'{name}.json'

    def _default(x):
        if isinstance(x, (np.floating, np.integer)): return float(x)
        if isinstance(x, np.ndarray):                return x.tolist()
        return str(x)

    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False, default=_default)
    print(f'    -> {path.relative_to(ROOT)}')


# ═══════════════════════════════════════════════════════════════════════════
# Étape 1 — Problème direct
# ═══════════════════════════════════════════════════════════════════════════

def run_direct(models, show_plots: bool = True):
    """
    Vérifie la cohérence simulateur / distributions théoriques.

    Pour chaque (modèle, taux) : histogramme vs f(t) = B(t)S(t), stat KS.
    Note : pour le modèle 'size', le KS élevé est attendu (troncature gauche).
    """
    _header('ETAPE 1 : Probleme direct  (simulateur vs theorie)')
    ks_stats = {}

    for model in models:
        for rate in MODEL_RATES[model]:
            t0 = time.perf_counter()
            print(f'  {model}/{rate}')
            result = verify_direct_problem(model, rate, str(DATA_DIR))
            ks     = result['ks_stat']
            ks_stats[f'{model}/{rate}'] = round(ks, 6)
            note = '  [eleve : troncature attendue]' if model == 'size' else ''
            print(f'    KS = {ks:.5f}{note}  '
                  f'({(time.perf_counter() - t0) * 1e3:.0f} ms)')
            if show_plots:
                fig = plot_direct_problem(model, rate, str(DATA_DIR))
                _save(fig, f'direct_{model}_{rate}')

    _save_json(ks_stats, 'direct_ks_stats')


# ═══════════════════════════════════════════════════════════════════════════
# Étape 2 — Problème inverse
# ═══════════════════════════════════════════════════════════════════════════

def run_inverse(models, n_max=None, show_plots: bool = True):
    """
    Estime B par les 5 méthodes pour chaque (modèle, taux).
    Méthodes : KDE, TSVD, Tikhonov p=0/1/2.
    """
    _header('ETAPE 2 : Probleme inverse  (estimation de B)')
    all_errors  = {}
    all_results = {}

    for model in models:
        for rate in MODEL_RATES[model]:
            t0 = time.perf_counter()
            print(f'\n  {model}/{rate}')

            results = run_all_methods(model, rate,
                                      data_dir=str(DATA_DIR), n_max=n_max)
            all_results[(model, rate)] = results

            errors_row = {}
            for method, res in results.items():
                errs = compute_errors(res)
                l2   = errs.get('l2_rel', float('nan'))
                a    = errs.get('alpha',  float('nan'))
                errors_row[method] = {'l2_rel': round(l2, 4), 'alpha': round(a, 6)}
                print(f'    {method:14s}  L2={l2:.4f}  alpha={a:.3e}')

            all_errors[f'{model}/{rate}'] = errors_row
            print(f'    [{time.perf_counter() - t0:.1f} s]')

            if show_plots:
                fig = plot_estimation_results(
                    results, title=f'Estimation de B  --  {model}/{rate}')
                _save(fig, f'inverse_{model}_{rate}')
                if 'tsvd' in results:
                    fig_p = plot_picard_criterion(results['tsvd'])
                    if fig_p is not None:
                        _save(fig_p, f'picard_{model}_{rate}')

    _save_json(all_errors, 'inverse_errors')

    if show_plots and all_results:
        _save(plot_global_summary(all_results), 'inverse_global_summary')

    return all_errors, all_results


# ═══════════════════════════════════════════════════════════════════════════
# Étape 3 — Sélection de α
# ═══════════════════════════════════════════════════════════════════════════

def run_alpha(models, show_plots: bool = True):
    """Compare Morozov, GCV, courbe-L, α a priori pour chaque dataset."""
    _header('ETAPE 3 : Selection du parametre alpha')
    alpha_summary = {}

    for model in models:
        for rate in MODEL_RATES[model]:
            print(f'  {model}/{rate}')
            analysis = analyze_alpha_selection(model, rate, str(DATA_DIR))

            alpha_summary[f'{model}/{rate}'] = {
                'epsilon'     : round(analysis['epsilon'],      6),
                'n'           : analysis['n'],
                'alpha_disc'  : round(analysis['alpha_disc'],   6),
                'alpha_gcv'   : round(analysis['alpha_gcv'],    6),
                'alpha_lcurve': round(analysis['alpha_lcurve'], 6),
                'err_disc'    : round(analysis['err_disc'],     4),
                'err_gcv'     : round(analysis['err_gcv'],      4),
                'err_lcurve'  : round(analysis['err_lcurve'],   4),
                'err_apriori' : {str(s): round(e, 4)
                                 for s, e in analysis['err_apriori'].items()},
            }
            print(f'    disc={analysis["alpha_disc"]:.3e} (L2={analysis["err_disc"]:.4f})'
                  f'  |  GCV={analysis["alpha_gcv"]:.3e} (L2={analysis["err_gcv"]:.4f})')

            if show_plots:
                fig = plot_alpha_selection(
                    analysis, title=f'Selection de alpha  --  {model}/{rate}')
                _save(fig, f'alpha_{model}_{rate}')

    _save_json(alpha_summary, 'alpha_comparison')


# ═══════════════════════════════════════════════════════════════════════════
# Étape 4 — Convergence en n
# ═══════════════════════════════════════════════════════════════════════════

def run_convergence(models, show_plots: bool = True):
    """
    Étudie la décroissance de l'erreur L2 en fonction de n.
    Durée estimée : ~5 min (8 valeurs de n × 4 répétitions × 3 méthodes × n_models).
    """
    _header('ETAPE 4 : Convergence en n  [peut prendre plusieurs minutes]')
    n_values      = [200, 500, 1000, 2000, 5000, 10000]
    methods_conv  = ['tikhonov_0', 'tikhonov_1', 'kde']
    alpha_sel_map = {'tikhonov_0': 'discrepancy',
                     'tikhonov_1': 'discrepancy',
                     'kde'       : 'silverman'}

    for model in models:
        rate = MODEL_RATES[model][0]   # 'constant'
        print(f'\n  {model}/{rate}')
        conv_list = []

        for method in methods_conv:
            print(f'  Methode : {method}')
            conv = convergence_study(
                model, rate,
                method=method,
                alpha_selection=alpha_sel_map[method],
                n_values=n_values,
                n_repeat=4,
                data_dir=str(DATA_DIR),
            )
            conv_list.append(conv)
            _save_json({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                        for k, v in conv.items() if k != 'l2_all'},
                       f'convergence_{model}_{rate}_{method}')

        if show_plots:
            fig = plot_convergence(
                conv_list, theoretical_s=[0.5, 1.0],
                title=f'Convergence en n  --  {model}/{rate}')
            _save(fig, f'convergence_{model}')


# ═══════════════════════════════════════════════════════════════════════════
# Étape 5 — Heatmap récapitulative
# ═══════════════════════════════════════════════════════════════════════════

def run_summary(models, n_max=None, show_plots: bool = True):
    """Heatmap (taux × méthodes) des erreurs L2 pour chaque modèle."""
    _header('ETAPE 5 : Resume global  (heatmap des erreurs)')
    all_summaries = {}

    for model in models:
        print(f'  Modele : {model}')
        comparison = compare_all_methods(model, data_dir=str(DATA_DIR), n_max=n_max)
        table, rates, methods = summary_table(comparison)
        all_summaries[model] = {
            'rates'  : rates,
            'methods': methods,
            'table'  : [[round(v, 4) if np.isfinite(v) else None for v in row]
                        for row in table.tolist()],
        }
        if show_plots:
            _save(plot_error_heatmap(comparison, model), f'heatmap_{model}')

    _save_json(all_summaries, 'summary_errors')


# ═══════════════════════════════════════════════════════════════════════════
# Point d'entrée
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Analyses problemes directs/inverses - division cellulaire',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--steps', nargs='+',
                   choices=['direct', 'inverse', 'alpha', 'convergence', 'summary'],
                   default=['direct', 'inverse', 'alpha', 'summary'])
    p.add_argument('--model', nargs='+',
                   choices=['age', 'size', 'increment'],
                   default=['age', 'size', 'increment'])
    p.add_argument('--n-max',    type=int, default=None)
    p.add_argument('--no-plots', action='store_true')
    return p.parse_args()


def main():
    args   = parse_args()
    models = args.model
    steps  = args.steps
    show   = not args.no_plots
    n_max  = args.n_max

    t0 = time.perf_counter()
    print(f'\n  Problemes inverses -- Division cellulaire')
    print(f'  Modeles  : {models}')
    print(f'  Etapes   : {steps}')
    print(f'  n_max    : {n_max or "10 000 (complet)"}')
    print(f'  Figures  : {"oui" if show else "non"}')
    print(f'  Donnees  : {DATA_DIR}')
    print(f'  Racine   : {ROOT}')

    if 'direct'      in steps: run_direct(models, show)
    if 'inverse'     in steps: run_inverse(models, n_max, show)
    if 'alpha'       in steps: run_alpha(models, show)
    if 'convergence' in steps: run_convergence(models, show)
    if 'summary'     in steps: run_summary(models, n_max, show)

    total = time.perf_counter() - t0
    print(f'\n{"=" * 62}')
    print(f'  Termine en {total:.1f} s')
    print(f'  Figures  -> {FIG_DIR}')
    print(f'  Resultats -> {RES_DIR}')
    print(f'{"=" * 62}\n')


if __name__ == '__main__':
    main()
