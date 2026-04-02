"""
Module src — Méthodes de problèmes inverses pour la division cellulaire.

Modules :
    direct_problem      : opérateur direct Ψ, calcul de H, S, f
    density_estimation  : KDE, Nelson-Aalen, estimateur de hasard
    linear_inverse      : SVD analytique, Tikhonov, TSVD
    parameter_selection : discordance, GCV, courbe en L
    hazard_estimation   : pipeline complet d'estimation
"""
from src.direct_problem import DirectProblemSolver, KNOWN_RATES, DivisionRateSpec
from src.density_estimation import (
    KernelDensityEstimator, NelsonAalanEstimator, KDEHazardEstimator,
)
from src.linear_inverse import AnalyticSVD, TruncatedSVD, TikhonovRegularizer
from src.parameter_selection import (
    DiscrepancyPrinciple, GeneralizedCrossValidation, LCurveMethod,
    alpha_apriori, theoretical_convergence_rate,
)
from src.hazard_estimation import (
    HazardEstimationPipeline, EstimationResult, run_all_methods,
    load_observations, METHODS,
)

__all__ = [
    'DirectProblemSolver', 'KNOWN_RATES', 'DivisionRateSpec',
    'KernelDensityEstimator', 'NelsonAalanEstimator', 'KDEHazardEstimator',
    'AnalyticSVD', 'TruncatedSVD', 'TikhonovRegularizer',
    'DiscrepancyPrinciple', 'GeneralizedCrossValidation', 'LCurveMethod',
    'alpha_apriori', 'theoretical_convergence_rate',
    'HazardEstimationPipeline', 'EstimationResult', 'run_all_methods',
    'load_observations', 'METHODS',
]
