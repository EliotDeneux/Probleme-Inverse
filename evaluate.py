"""
evaluate.py — Évaluation des méthodes et analyses quantitatives.

Ce module est à la RACINE du projet (même niveau que run_all.py).
Il fournit toutes les fonctions d'analyse des résultats :
  - Métriques d'erreur (L², L∞)
  - Vérification du problème direct (simulateur vs théorie)
  - Étude de convergence en n
  - Comparaison de toutes les méthodes
  - Analyse des méthodes de sélection de α

Référence théorique : cours Doumic 2025.

Taux de convergence théoriques (Sections 3.3, 4.1, 5.2) :
─────────────────────────────────────────────────────────
  ε = 1/√n,  B ∈ Y^s :
    Tikhonov p=0  (qualif. 1)  →  O(n^{-s/(2s+1)}),  saturé à O(n^{-1/3}) si s > 1
    Tikhonov p=1  (qualif. 3)  →  O(n^{-s/(2s+1)}),  saturé à O(n^{-3/7}) si s > 3
    SVD tronquée  (qualif. ∞)  →  O(n^{-s/(2s+1)}),  jamais saturé
    KDE  (f ∈ W^{m,∞})         →  O(n^{-m/(2m+1)})
"""

from __future__ import annotations
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Assure que src/ est trouvable depuis la racine du projet
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import (
    HazardEstimationPipeline, EstimationResult, run_all_methods,
    KNOWN_RATES, METHODS, theoretical_convergence_rate,
    load_observations, DirectProblemSolver, NelsonAalanEstimator,
    TikhonovRegularizer, DiscrepancyPrinciple, GeneralizedCrossValidation,
    LCurveMethod, alpha_apriori,
)


# ═══════════════════════════════════════════════════════════════════════════
# Métriques d'erreur
# ═══════════════════════════════════════════════════════════════════════════

def compute_errors(result: EstimationResult) -> Dict:
    """
    Calcule les métriques d'erreur pour un EstimationResult.

    Les deux erreurs relatives servent à différents diagnostics :
      - l2_rel  (intégrale) : erreur globale, robuste aux pics locaux
      - linf_rel (maximum)  : pire erreur locale, sensible aux oscillations

    Returns
    -------
    dict avec clés 'l2_abs', 'l2_rel', 'linf_abs', 'linf_rel', 'alpha', 'n'.
    Retourne {} si B_true n'est pas disponible.
    """
    if result.B_true is None:
        return {}

    # Masque : exclure les NaN, infinis, et zones où B_true ≈ 0
    mask = (np.isfinite(result.B_hat)
            & np.isfinite(result.B_true)
            & (result.B_true > 1e-10))

    if not np.any(mask):
        return {}

    g  = result.grid[mask]
    Bh = result.B_hat[mask]
    Bt = result.B_true[mask]

    l2_abs  = float(np.sqrt(np.trapezoid((Bh - Bt)**2, g)))
    l2_true = float(np.sqrt(np.trapezoid(Bt**2, g)))
    l2_rel  = l2_abs / max(l2_true, 1e-15)

    linf_abs = float(np.max(np.abs(Bh - Bt)))
    linf_rel = linf_abs / max(float(np.max(Bt)), 1e-15)

    return {
        'l2_abs'  : l2_abs,
        'l2_rel'  : l2_rel,
        'linf_abs': linf_abs,
        'linf_rel': linf_rel,
        'alpha'   : result.alpha or float('nan'),
        'n'       : result.n_cells,
        'method'  : result.method,
        'model'   : result.model,
        'rate'    : result.rate_name,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Vérification du problème direct
# ═══════════════════════════════════════════════════════════════════════════

def verify_direct_problem(model: str, rate_name: str,
                           data_dir: str = 'data') -> Dict:
    """
    Vérifie la cohérence entre les données simulées et la théorie.

    Compare l'histogramme empirique de T avec la densité théorique f(t),
    et la CDF empirique avec la CDF théorique F(t).

    Note : pour le modèle 'size', le KS est élevé (~0.8) car F(x) est
    la CDF marginale de X_ud, alors que chaque cellule est tirée
    conditionnellement à X_ub — le test KS n'est pas adapté pour ce modèle.
    La comparaison visuelle (histogramme vs f) reste pertinente.

    Returns
    -------
    dict avec clés : grid, theory{B,H,S,f,F}, hist_x, hist_y, F_emp, ks_stat, T_data.
    """
    obs      = load_observations(model, rate_name, data_dir)
    T        = obs['T']
    rate_spec = KNOWN_RATES.get((model, rate_name))

    if rate_spec is None:
        raise ValueError(f"Taux '{rate_name}' inconnu pour le modèle '{model}'.")

    grid   = DirectProblemSolver.grid_from_data(T, n_points=300)
    solver = DirectProblemSolver(grid)

    B_vals = rate_spec.func(grid)
    theory = solver.compute_all(B_vals)

    # Histogramme normalisé des observations
    t_min, t_max = grid[0], grid[-1]
    counts, edges = np.histogram(
        T[(T >= t_min) & (T <= t_max)],
        bins=50, density=True,
        range=(t_min, t_max),
    )
    centers = 0.5 * (edges[:-1] + edges[1:])

    # CDF empirique sur la grille théorique
    F_emp = np.array([np.mean(T <= t) for t in grid])

    # Statistique de Kolmogorov-Smirnov
    ks_stat = float(np.max(np.abs(F_emp - theory['F'])))

    return {
        'grid'    : grid,
        'theory'  : theory,
        'hist_x'  : centers,
        'hist_y'  : counts,
        'F_emp'   : F_emp,
        'ks_stat' : ks_stat,
        'T_data'  : T,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Comparaison de toutes les méthodes
# ═══════════════════════════════════════════════════════════════════════════

def compare_all_methods(model: str, data_dir: str = 'data',
                        n_max: Optional[int] = None,
                        ) -> Dict[str, Dict[str, Dict]]:
    """
    Lance toutes les méthodes d'estimation sur tous les taux d'un modèle.

    Parameters
    ----------
    model    : 'age', 'size' ou 'increment'
    data_dir : chemin vers le dossier de données
    n_max    : sous-échantillonnage optionnel

    Returns
    -------
    {rate_name: {method_name: error_dict}}
    """
    rate_names = [k[1] for k in KNOWN_RATES if k[0] == model]
    all_results: Dict[str, Dict] = {}

    for rate in rate_names:
        print(f'  {model}/{rate}')
        results = run_all_methods(model, rate, data_dir=data_dir, n_max=n_max)
        all_results[rate] = {m: compute_errors(r) for m, r in results.items()}

    return all_results


def summary_table(comparison_dict: Dict
                  ) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Construit le tableau (taux × méthodes) des erreurs L² relatives.

    Returns
    -------
    table   : np.ndarray de forme (n_rates, n_methods)
    rates   : liste des noms de taux (lignes)
    methods : liste des noms de méthodes (colonnes)
    """
    rates   = list(comparison_dict.keys())
    methods = METHODS

    table = np.full((len(rates), len(methods)), np.nan)
    for i, rate in enumerate(rates):
        for j, method in enumerate(methods):
            errs = comparison_dict[rate].get(method, {})
            table[i, j] = errs.get('l2_rel', np.nan)

    return table, rates, methods


# ═══════════════════════════════════════════════════════════════════════════
# Étude de convergence en n
# ═══════════════════════════════════════════════════════════════════════════

def convergence_study(model: str, rate_name: str,
                      method: str          = 'tikhonov_0',
                      alpha_selection: str = 'discrepancy',
                      n_values: Optional[List[int]] = None,
                      n_repeat: int        = 5,
                      data_dir: str        = 'data',
                      seed: int            = 0,
                      ) -> Dict:
    """
    Étudie la décroissance de l'erreur L² relative en fonction de n.

    Pour chaque n ∈ n_values, sous-échantillonne n_repeat fois les données
    (n_total = 10 000 disponibles) et calcule l'erreur moyenne et son écart-type.

    Complexité : O(n_values × n_repeat) estimations (environ 5–10 min au total).

    Returns
    -------
    dict avec clés :
      'n_values'   : array des tailles testées
      'l2_mean'    : moyenne des erreurs L² relatives
      'l2_std'     : écart-type des erreurs
      'l2_all'     : array (n_values, n_repeat) de toutes les erreurs
      'alpha_mean' : moyenne des α sélectionnés
      'method', 'model', 'rate' : métadonnées
    """
    if n_values is None:
        n_values = [100, 200, 500, 1000, 2000, 5000, 10000]

    rng       = np.random.default_rng(seed)
    pipeline  = HazardEstimationPipeline()
    l2_all    = np.full((len(n_values), n_repeat), np.nan)
    alpha_all = np.full((len(n_values), n_repeat), np.nan)

    for i, n in enumerate(n_values):
        print(f'    n={n:6d} ', end='', flush=True)
        for r in range(n_repeat):
            np.random.seed(int(rng.integers(1_000_000)))
            try:
                res = pipeline.run(
                    model, rate_name,
                    method=method,
                    alpha_selection=alpha_selection,
                    data_dir=data_dir,
                    n_max=n,
                )
                errs = compute_errors(res)
                l2_all[i, r]    = errs.get('l2_rel', np.nan)
                alpha_all[i, r] = errs.get('alpha', np.nan)
            except Exception as e:
                print(f'[err:{e}] ', end='')
        mu = float(np.nanmean(l2_all[i]))
        print(f'L²_rel = {mu:.4f}')

    return {
        'n_values'  : np.array(n_values),
        'l2_mean'   : np.nanmean(l2_all, axis=1),
        'l2_std'    : np.nanstd(l2_all,  axis=1),
        'l2_all'    : l2_all,
        'alpha_mean': np.nanmean(alpha_all, axis=1),
        'method'    : method,
        'model'     : model,
        'rate'      : rate_name,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Analyse de la sélection de α (Tikhonov p=0)
# ═══════════════════════════════════════════════════════════════════════════

def analyze_alpha_selection(model: str, rate_name: str,
                             data_dir: str = 'data') -> Dict:
    """
    Compare les méthodes de sélection de α sur un dataset donné.

    Méthodes comparées :
      - Principe de discordance de Morozov (a posteriori)
      - Validation croisée généralisée (GCV)
      - Courbe en L (heuristique)
      - α a priori théorique pour différentes régularités s

    Returns
    -------
    dict avec :
      alpha_grid, residuals, gcv_vals  : courbes de sélection
      lcurve_x, lcurve_y               : données de la courbe en L
      alpha_disc, alpha_gcv, alpha_lcurve : α sélectionnés
      alpha_apriori                    : {s: alpha} pour s ∈ {0.5,1,1.5,2}
      err_disc, err_gcv, err_lcurve    : erreurs L² associées
      err_apriori                      : {s: erreur}
      n, epsilon                       : taille et niveau de bruit
      grid, H_eps                      : grille et données lissées
    """
    obs   = load_observations(model, rate_name, data_dir)
    T     = obs['T']
    X_ub  = obs.get('X_ub', None)
    n     = len(T)
    eps   = 1.0 / np.sqrt(n)

    grid   = DirectProblemSolver.grid_from_data(T)
    solver = DirectProblemSolver(grid)
    A      = solver.integration_matrix

    # Nelson-Aalen + mollification
    na    = NelsonAalanEstimator().fit(T, entry_times=X_ub)
    H_eps = na.smooth(grid, sigma_grid=2.0)

    # Tikhonov p=0 comme référence pour les courbes de sélection
    tikh  = TikhonovRegularizer(A, p=0).fit(H_eps)
    ag    = np.logspace(-6, 0, 80)

    # Courbes de sélection
    _, residuals = DiscrepancyPrinciple().curve(tikh, ag)
    _, gcv_vals  = GeneralizedCrossValidation().curve(tikh, ag)
    _, lr, ln    = LCurveMethod().compute_curve(tikh, ag)

    # Paramètres sélectionnés
    alpha_disc   = DiscrepancyPrinciple().select(tikh, eps)
    alpha_gcv    = GeneralizedCrossValidation().select(tikh, ag)
    alpha_lcurve = LCurveMethod().select(tikh, ag)
    alpha_apr    = {s: alpha_apriori(eps, s=s) for s in [0.5, 1.0, 1.5, 2.0]}

    # Taux vrai pour calculer les erreurs
    rate_spec = KNOWN_RATES.get((model, rate_name))
    B_true    = rate_spec.func(grid) if rate_spec else None

    def _l2_err(a: float) -> float:
        if B_true is None:
            return float('nan')
        B_hat = tikh.predict(a)
        mask  = np.isfinite(B_hat) & np.isfinite(B_true)
        num   = float(np.sqrt(np.trapezoid((B_hat[mask] - B_true[mask])**2, grid[mask])))
        den   = float(np.sqrt(np.trapezoid(B_true[mask]**2, grid[mask])))
        return num / max(den, 1e-15)

    return {
        'alpha_grid'  : ag,
        'residuals'   : residuals,
        'gcv_vals'    : gcv_vals,
        'lcurve_x'    : lr,
        'lcurve_y'    : ln,
        'alpha_disc'  : alpha_disc,
        'alpha_gcv'   : alpha_gcv,
        'alpha_lcurve': alpha_lcurve,
        'alpha_apriori': alpha_apr,
        'err_disc'    : _l2_err(alpha_disc),
        'err_gcv'     : _l2_err(alpha_gcv),
        'err_lcurve'  : _l2_err(alpha_lcurve),
        'err_apriori' : {s: _l2_err(a) for s, a in alpha_apr.items()},
        'n'           : n,
        'epsilon'     : eps,
        'grid'        : grid,
        'H_eps'       : H_eps,
        'target_disc' : 1.1 * eps * np.sqrt(len(grid)),
    }
