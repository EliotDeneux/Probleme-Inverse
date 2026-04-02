"""
Pipeline complet d'estimation du taux de division B pour chaque modèle.

Intègre les modules src/ en un pipeline cohérent :
    données → Ĥ (Nelson-Aalen) → régularisation → B̂

Architecture (cours Doumic 2025) :
──────────────────────────────────
    Données T₁,...,Tₙ     ← observations (âges, tailles division, incréments)
         ↓
    Ĥ_NA(t)              ← estimateur Nelson-Aalen (zε ≈ H)
         ↓
    Lissage optionnel     ← mollificateur (réduction bruit, Section 5.1.2)
         ↓
    Régularisation        ← TSVD | Tikhonov classique | Tikhonov généralisé
         ↓
    B̂(t) = (ΨΨ+α²I)⁻¹Ψ*Ĥ  ← estimée du taux de hasard

Méthode KDE directe (alternative) :
    B̂(t) = f̂_h(t) / Ŝ(t)

Modèles gérés :
    'age'       : T = A_ud (âge division), données i.i.d.
    'increment' : T = Z_ud = X_ud - X_ub,  données i.i.d.
    'size'      : T = X_ud avec entrée X_ub (troncature gauche)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np
from pathlib import Path

from src.direct_problem import DirectProblemSolver, KNOWN_RATES, DivisionRateSpec
from src.density_estimation import NelsonAalanEstimator, KDEHazardEstimator
from src.linear_inverse import TikhonovRegularizer, TruncatedSVD
from src.parameter_selection import (
    DiscrepancyPrinciple, GeneralizedCrossValidation,
    LCurveMethod, alpha_apriori,
)


# ═══════════════════════════════════════════════════════════════════════════
# Structure de résultat
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EstimationResult:
    """
    Résultat d'une estimation du taux de hasard B̂.

    Attributes
    ----------
    grid       : grille de discrétisation
    B_hat      : taux estimé B̂(t) sur la grille
    B_true     : taux théorique B(t) si disponible
    alpha      : paramètre de régularisation utilisé
    method     : nom de la méthode
    model      : modèle de division ('age', 'size', 'increment')
    rate_name  : nom du taux de référence
    n_cells    : taille de l'échantillon
    noise_level: ε estimé (≈ 1/√n)
    extras     : données supplémentaires (résidus, courbe L, etc.)
    """
    grid      : np.ndarray
    B_hat     : np.ndarray
    B_true    : Optional[np.ndarray] = None
    alpha     : Optional[float]      = None
    method    : str                  = ''
    model     : str                  = ''
    rate_name : str                  = ''
    n_cells   : int                  = 0
    noise_level: float               = 0.0
    extras    : Dict                 = field(default_factory=dict)

    def l2_error(self) -> Optional[float]:
        """||B̂ - B||_{L²} / ||B||_{L²}  (erreur relative)."""
        if self.B_true is None:
            return None
        mask = np.isfinite(self.B_hat) & np.isfinite(self.B_true)
        num  = np.trapz((self.B_hat[mask] - self.B_true[mask])**2, self.grid[mask])
        den  = np.trapz(self.B_true[mask]**2, self.grid[mask])
        return float(np.sqrt(num / max(den, 1e-15)))

    def linf_error(self) -> Optional[float]:
        """||B̂ - B||_{L∞} / max(B)  (erreur relative infinie)."""
        if self.B_true is None:
            return None
        mask = np.isfinite(self.B_hat) & np.isfinite(self.B_true)
        return float(np.max(np.abs(self.B_hat[mask] - self.B_true[mask]))
                     / max(np.max(self.B_true[mask]), 1e-15))


# ═══════════════════════════════════════════════════════════════════════════
# Fonction de chargement des données
# ═══════════════════════════════════════════════════════════════════════════

def load_observations(model: str, rate_name: str,
                      data_dir: str = '/data',
                      n_max: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Charge un fichier .npz et retourne les observations selon le modèle.

    Returns
    -------
    dict avec clés :
        'T'    : variable de division (âge, incrément, ou taille division)
        'X_ub' : taille à la naissance (uniquement modèle 'size')
        'raw'  : données brutes complètes
    """
    path = Path(data_dir) / model / f"{rate_name}.npz"
    d    = np.load(path)

    # Sous-échantillonnage éventuel
    n = len(d['cell_id'])
    if n_max is not None and n_max < n:
        idx = np.random.choice(n, n_max, replace=False)
    else:
        idx = np.arange(n)

    raw = {k: d[k][idx] for k in d.files}

    if model == 'age':
        T = raw['division_age']
        return {'T': T, 'raw': raw}
    elif model == 'increment':
        T = raw['increment']
        return {'T': T, 'raw': raw}
    elif model == 'size':
        T    = raw['division_size']
        X_ub = raw['birth_size']
        return {'T': T, 'X_ub': X_ub, 'raw': raw}
    else:
        raise ValueError(f"Modèle inconnu : '{model}'")


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline principal d'estimation
# ═══════════════════════════════════════════════════════════════════════════

class HazardEstimationPipeline:
    """
    Pipeline complet pour estimer B(·) à partir des données simulées.

    Méthodes disponibles :
    ──────────────────────
    'kde'                 : B̂ = f̂_h / Ŝ  (KDE direct, Section 5.2)
    'tsvd'                : SVD tronquée de Ψ sur Ĥ_NA  (Section 3.2)
    'tikhonov_0'          : Tikhonov classique p=0  (Section 4.1.1)
    'tikhonov_1'          : Tikhonov généralisé p=1 (Section 4.1.3)
    'tikhonov_2'          : Tikhonov généralisé p=2 (Section 4.1.3)

    Sélection de α :
    ────────────────
    'discrepancy'         : principe de discordance de Morozov
    'gcv'                 : validation croisée généralisée
    'lcurve'              : courbe en L
    'apriori_s{s}'        : formule théorique α = (ε/δ)^{1/(2s+1)}

    Parameters
    ----------
    n_grid    : nb de points de la grille [0, T_max]
    quantile  : T_max = quantile de cette fraction des données
    J_tsvd    : nb de modes pour la SVD tronquée
    """

    def __init__(self, n_grid: int = 200, quantile: float = 0.98,
                 J_tsvd: int = 80):
        self.n_grid   = n_grid
        self.quantile = quantile
        self.J_tsvd   = J_tsvd

    def run(self, model: str, rate_name: str,
            method: str         = 'tikhonov_0',
            alpha_selection: str = 'discrepancy',
            data_dir: str       = '/data',
            n_max: Optional[int] = None,
            fixed_alpha: Optional[float] = None,
            ) -> EstimationResult:
        """
        Exécute le pipeline complet pour un (modèle, taux, méthode) donné.

        Parameters
        ----------
        model, rate_name : identifiants des données simulées
        method           : méthode d'estimation (voir ci-dessus)
        alpha_selection  : méthode de sélection de α
        data_dir         : répertoire des données
        n_max            : sous-échantillonnage optionnel
        fixed_alpha      : α imposé (ignore alpha_selection)
        """
        # ── 1. Chargement des données ─────────────────────────────────────
        data   = load_observations(model, rate_name, data_dir, n_max)
        T_obs  = data['T']
        X_ub   = data.get('X_ub', None)
        n      = len(T_obs)
        eps    = 1.0 / np.sqrt(n)   # niveau de bruit estimé

        # ── 2. Construction de la grille ──────────────────────────────────
        grid = DirectProblemSolver.grid_from_data(T_obs, self.n_grid, self.quantile)

        # ── 3. Calcul du taux vrai sur la grille ──────────────────────────
        B_true_vals: Optional[np.ndarray] = None
        rate_spec = KNOWN_RATES.get((model, rate_name))
        if rate_spec is not None:
            B_true_vals = rate_spec.func(grid)

        # ── 4. Estimation ─────────────────────────────────────────────────
        if method == 'kde':
            result = self._run_kde(
                T_obs, X_ub, grid, model, rate_name, rate_spec,
                B_true_vals, n, eps,
            )
        else:
            result = self._run_regularization(
                T_obs, X_ub, grid, model, rate_name, rate_spec,
                B_true_vals, n, eps, method, alpha_selection, fixed_alpha,
            )

        return result

    # ── Méthode KDE directe ───────────────────────────────────────────────

    def _run_kde(self, T_obs, X_ub, grid, model, rate_name,
                 rate_spec, B_true_vals, n, eps) -> EstimationResult:
        estimator = KDEHazardEstimator(bandwidth='silverman', kernel='gaussian')
        estimator.fit(T_obs, entry_times=X_ub)
        B_hat = estimator.predict_B(grid)

        return EstimationResult(
            grid=grid, B_hat=B_hat, B_true=B_true_vals,
            alpha=estimator.kde.bandwidth_value,
            method='kde', model=model, rate_name=rate_name,
            n_cells=n, noise_level=eps,
            extras={'h_kde': estimator.kde.bandwidth_value},
        )

    # ── Méthodes de régularisation ────────────────────────────────────────

    def _run_regularization(self, T_obs, X_ub, grid, model, rate_name,
                            rate_spec, B_true_vals, n, eps, method,
                            alpha_selection, fixed_alpha) -> EstimationResult:
        """Pipeline Nelson-Aalen → lissage → régularisation."""
        # ── Nelson-Aalen ──────────────────────────────────────────────────
        na = NelsonAalanEstimator()
        na.fit(T_obs, entry_times=X_ub)
        H_raw = na.predict(grid)

        # Lissage léger (mollificateur) pour réduire les artefacts en escalier
        H_eps = na.smooth(grid, sigma_grid=2.0)

        # ── Opérateur direct discrétisé ───────────────────────────────────
        solver = DirectProblemSolver(grid)
        A      = solver.integration_matrix

        # ── Sélection de la méthode ───────────────────────────────────────
        if method == 'tsvd':
            return self._run_tsvd(
                H_eps, grid, A, eps, model, rate_name,
                B_true_vals, n, alpha_selection, fixed_alpha,
                extras={'H_raw': H_raw, 'H_eps': H_eps},
            )
        else:
            p = int(method.split('_')[1])   # 'tikhonov_0' → p=0
            return self._run_tikhonov(
                H_eps, grid, A, eps, p, model, rate_name,
                B_true_vals, n, alpha_selection, fixed_alpha,
                extras={'H_raw': H_raw, 'H_eps': H_eps},
            )

    def _run_tsvd(self, H_eps, grid, A, eps, model, rate_name,
                  B_true_vals, n, alpha_selection, fixed_alpha,
                  extras) -> EstimationResult:
        tsvd = TruncatedSVD(J_max=self.J_tsvd)
        tsvd.fit(H_eps, grid)

        if fixed_alpha is not None:
            alpha = fixed_alpha
        elif alpha_selection == 'discrepancy':
            # Pour TSVD : choisir N tel que résidu ≈ ε√m
            # (approché via Tikhonov classique pour la dichotomie)
            tikh = TikhonovRegularizer(A, p=0).fit(H_eps)
            alpha = DiscrepancyPrinciple().select(tikh, eps)
        elif alpha_selection == 'gcv':
            tikh  = TikhonovRegularizer(A, p=0).fit(H_eps)
            alpha = GeneralizedCrossValidation().select(tikh)
        elif alpha_selection.startswith('apriori'):
            s = float(alpha_selection.split('s')[-1])
            alpha = alpha_apriori(eps, s=s, p=0)
        else:
            alpha = eps

        B_hat = tsvd.predict(alpha)
        j_vals, sigmas, picard_c = tsvd.picard_plot_data()

        return EstimationResult(
            grid=grid, B_hat=B_hat, B_true=B_true_vals,
            alpha=alpha, method='tsvd', model=model, rate_name=rate_name,
            n_cells=n, noise_level=eps,
            extras={**extras, 'picard_j': j_vals, 'picard_sigma': sigmas,
                    'picard_c': picard_c},
        )

    def _run_tikhonov(self, H_eps, grid, A, eps, p, model, rate_name,
                      B_true_vals, n, alpha_selection, fixed_alpha,
                      extras) -> EstimationResult:
        tikh = TikhonovRegularizer(A, p=p).fit(H_eps)

        if fixed_alpha is not None:
            alpha = fixed_alpha
        elif alpha_selection == 'discrepancy':
            alpha = DiscrepancyPrinciple().select(tikh, eps)
        elif alpha_selection == 'gcv':
            alpha = GeneralizedCrossValidation().select(tikh)
        elif alpha_selection == 'lcurve':
            alpha = LCurveMethod().select(tikh)
        elif alpha_selection.startswith('apriori'):
            s = float(alpha_selection.split('s')[-1])
            alpha = alpha_apriori(eps, s=s, p=p)
        else:
            alpha = eps**(1.0 / (2 * p + 2))   # heuristique

        B_hat = tikh.predict(alpha)

        # Informations supplémentaires pour analyse
        alpha_grid = np.logspace(-6, 1, 50)
        residuals  = np.array([tikh.residual(a) for a in alpha_grid])
        gcv_vals   = GeneralizedCrossValidation().values(tikh, alpha_grid)
        _, lr, ln  = LCurveMethod().compute_curve(tikh, alpha_grid)

        return EstimationResult(
            grid=grid, B_hat=B_hat, B_true=B_true_vals,
            alpha=alpha, method=f'tikhonov_p{p}', model=model,
            rate_name=rate_name, n_cells=n, noise_level=eps,
            extras={**extras,
                    'alpha_grid': alpha_grid,
                    'residuals' : residuals,
                    'gcv_vals'  : gcv_vals,
                    'lcurve_x'  : lr,
                    'lcurve_y'  : ln,
                    'filter_r'  : tikh.filter_function(alpha),
                    'sigma_A'   : tikh._s,
                    },
        )


# ═══════════════════════════════════════════════════════════════════════════
# Fonction de haut niveau : toutes méthodes × tous modèles
# ═══════════════════════════════════════════════════════════════════════════

METHODS = ['kde', 'tsvd', 'tikhonov_0', 'tikhonov_1', 'tikhonov_2']
ALPHA_SELECTIONS = {
    'kde'         : 'silverman',     # sélection bande passante KDE
    'tsvd'        : 'discrepancy',
    'tikhonov_0'  : 'discrepancy',
    'tikhonov_1'  : 'discrepancy',
    'tikhonov_2'  : 'gcv',
}


def run_all_methods(model: str, rate_name: str,
                    data_dir: str = '/data',
                    n_max: Optional[int] = None,
                    ) -> Dict[str, EstimationResult]:
    """
    Lance toutes les méthodes sur un (modèle, taux) et retourne les résultats.

    Returns
    -------
    dict {nom_méthode: EstimationResult}
    """
    pipeline = HazardEstimationPipeline()
    results  = {}
    for method in METHODS:
        alpha_sel = ALPHA_SELECTIONS[method]
        try:
            res = pipeline.run(
                model, rate_name,
                method         = method,
                alpha_selection= alpha_sel,
                data_dir       = data_dir,
                n_max          = n_max,
            )
            results[method] = res
        except Exception as exc:
            print(f"  [WARN] {model}/{rate_name}/{method}: {exc}")
    return results
