"""
Problème direct : calcul des distributions théoriques à partir d'un taux B.

Contexte mathématique (cours Doumic 2025, Section 2.1 + 3.4) :
─────────────────────────────────────────────────────────────
L'opérateur direct central est l'intégration :

    Ψ : B ──→ H,    (ΨB)(t) = ∫₀ᵗ B(s) ds    (hasard cumulé)

C'est l'opérateur paradigmatique du cours (eq. 2.10), dont la SVD analytique
est calculée en Section 3.4.1 :
    σⱼ = 2T/[π(2j+1)],  j = 0,1,2,...

Relations fondamentales entre quantités :
    H(t) = (ΨB)(t) = ∫₀ᵗ B(s) ds          (hasard cumulé)
    S(t) = exp(-H(t))                        (survie = P(T > t))
    f(t) = B(t) · S(t)                       (densité de division)
    F(t) = 1 - S(t)                          (CDF)
    B(t) = f(t) / S(t)  ← PROBLÈME INVERSE   (taux de hasard)

Modèles de division :
    Âge       : T = A_ud, variable = âge à la division
    Incrément : T = Z_ud = X_ud - X_ub, même structure que l'âge
    Taille    : T = X_ud, conditionné sur X_ub (troncature gauche)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional
import numpy as np
from scipy.integrate import cumulative_trapezoid


# ═══════════════════════════════════════════════════════════════════════════
# Spécification des taux de division
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DivisionRateSpec:
    """
    Spécification complète d'un taux de division B.

    Attributes
    ----------
    name        : identifiant court (ex: 'constant', 'weibull2')
    func        : B(t) – accepte un array numpy et retourne un array
    params      : dictionnaire de paramètres numériques
    description : formule lisible
    domain      : 'age' | 'size' | 'increment'
    """
    name       : str
    func       : Callable[[np.ndarray], np.ndarray]
    params     : Dict
    description: str
    domain     : str


def make_constant(lam: float, domain: str) -> DivisionRateSpec:
    """B(t) = λ  (taux constant, division ~ Exponentielle(λ))."""
    return DivisionRateSpec(
        name='constant',
        func=lambda t, l=lam: np.full_like(np.asarray(t, float), l),
        params={'lambda': lam},
        description=f'B(t) = {lam}',
        domain=domain,
    )


def make_weibull2(sigma: float, domain: str) -> DivisionRateSpec:
    """B(t) = 2t/σ²  (taux croissant linéaire, division ~ Weibull(2, σ))."""
    return DivisionRateSpec(
        name='weibull2',
        func=lambda t, s=sigma: 2 * np.asarray(t, float) / s**2,
        params={'sigma': sigma},
        description=f'B(t) = 2t/{sigma}²',
        domain=domain,
    )


def make_step(lam: float, t0: float, domain: str) -> DivisionRateSpec:
    """B(t) = λ · 𝟏(t ≥ t₀)  (seuil minimal avant division)."""
    return DivisionRateSpec(
        name='step',
        func=lambda t, l=lam, t_=t0: np.where(np.asarray(t, float) >= t_, l, 0.0),
        params={'lambda': lam, 't0': t0},
        description=f'B(t) = {lam}·𝟏(t≥{t0})',
        domain=domain,
    )


def make_linear(alpha: float) -> DivisionRateSpec:
    """B(x) = αx  (taux croissant, seuil de taille absolu)."""
    return DivisionRateSpec(
        name='linear',
        func=lambda x, a=alpha: a * np.asarray(x, float),
        params={'alpha': alpha},
        description=f'B(x) = {alpha}x',
        domain='size',
    )


def make_power(beta: float, gamma: float) -> DivisionRateSpec:
    """B(x) = β x^γ  (puissance intermédiaire)."""
    return DivisionRateSpec(
        name='power',
        func=lambda x, b=beta, g=gamma: b * np.asarray(x, float)**g,
        params={'beta': beta, 'gamma': gamma},
        description=f'B(x) = {beta}·x^{gamma}',
        domain='size',
    )


# ── Catalogue des taux correspondant aux fichiers simulés ─────────────────
# Paramètres identiques à simulate_division.py pour reproductibilité.
KNOWN_RATES: Dict[tuple, DivisionRateSpec] = {
    # Modèle âge (t en minutes)
    ('age', 'constant') : make_constant(lam=0.02,  domain='age'),
    ('age', 'weibull2') : make_weibull2(sigma=60.0, domain='age'),
    ('age', 'step')     : make_step(lam=0.05, t0=20.0, domain='age'),
    # Modèle taille (x en µm)
    ('size', 'constant'): make_constant(lam=2.0, domain='size'),
    ('size', 'linear')  : make_linear(alpha=4.0),
    ('size', 'power')   : make_power(beta=3.0, gamma=0.5),
    # Modèle incrément (z en µm)
    ('increment', 'constant'): make_constant(lam=2.0,  domain='increment'),
    ('increment', 'weibull2'): make_weibull2(sigma=0.7, domain='increment'),
    ('increment', 'step')    : make_step(lam=4.0, t0=0.2, domain='increment'),
}


# ═══════════════════════════════════════════════════════════════════════════
# Solveur du problème direct
# ═══════════════════════════════════════════════════════════════════════════

class DirectProblemSolver:
    """
    Résout le problème direct : étant donné B, calcule toutes les distributions.

    Opérateur direct (discrétisé) :
        Ψ  : B → H = A · B   (A = matrice d'intégration trapézoïdale)
        Ψ* : H → B via A^T   (opérateur adjoint)

    Parameters
    ----------
    grid : np.ndarray
        Grille de discrétisation [t₀, t₁, ..., t_{m-1}] de l'espace des variables.
    """

    def __init__(self, grid: np.ndarray):
        self.grid = np.asarray(grid, dtype=float)
        self.m = len(grid)
        self.T = grid[-1] - grid[0]
        self._A: Optional[np.ndarray] = None  # matrice d'intégration (lazy)

    # ── Calculs continus (via intégration numérique) ────────────────────────

    def compute_H(self, B_vals: np.ndarray) -> np.ndarray:
        """H(t) = ∫₀ᵗ B(s) ds  (hasard cumulé, opérateur Ψ)."""
        H = np.zeros_like(B_vals, dtype=float)
        H[1:] = cumulative_trapezoid(B_vals, self.grid)
        return H

    def compute_S(self, B_vals: np.ndarray) -> np.ndarray:
        """S(t) = exp(-H(t))  (survie)."""
        return np.exp(-self.compute_H(B_vals))

    def compute_f(self, B_vals: np.ndarray) -> np.ndarray:
        """f(t) = B(t) · S(t)  (densité de division)."""
        return B_vals * self.compute_S(B_vals)

    def compute_all(self, B_vals: np.ndarray) -> Dict[str, np.ndarray]:
        """Calcule H, S, f, F en un seul appel."""
        B = np.asarray(B_vals, dtype=float)
        H = self.compute_H(B)
        S = np.exp(-H)
        return {
            'grid': self.grid,
            'B': B,
            'H': H,
            'S': S,
            'f': B * S,
            'F': 1.0 - S,
        }

    def compute_mean(self, B_vals: np.ndarray) -> float:
        """E[T] = ∫₀^∞ S(t) dt  (espérance de la variable de division)."""
        return float(np.trapz(self.compute_S(B_vals), self.grid))

    def verify_normalization(self, B_vals: np.ndarray) -> float:
        """∫₀^∞ f(t) dt  (doit être proche de 1 si la grille couvre le support)."""
        return float(np.trapz(self.compute_f(B_vals), self.grid))

    # ── Opérateur discret (matrice) ─────────────────────────────────────────

    @property
    def integration_matrix(self) -> np.ndarray:
        """
        Matrice d'intégration trapézoïdale A ∈ ℝ^{m×m}.

        (AB)[i] ≈ ∫₀^{grid[i]} B(s) ds  (règle des trapèzes)

        Construction :
            Chaque segment [t_k, t_{k+1}] de longueur h_k contribue à toutes
            les lignes i ≥ k+1 par h_k/2 aux colonnes k et k+1.

        Cette matrice triangulaire inférieure est l'opérateur Ψ discrétisé,
        dont la SVD est calculée analytiquement ou numériquement.
        """
        if self._A is not None:
            return self._A

        m = len(self.grid)
        h = np.diff(self.grid)      # h[k] = grid[k+1] - grid[k]
        A = np.zeros((m, m))

        # Contribution du segment [t_k, t_{k+1}] à toutes les lignes i > k
        for k in range(m - 1):
            A[k + 1:, k]     += h[k] / 2.0
            A[k + 1:, k + 1] += h[k] / 2.0

        self._A = A
        return A

    def apply_forward(self, B_vals: np.ndarray) -> np.ndarray:
        """H = A · B  (application matricielle de Ψ)."""
        return self.integration_matrix @ np.asarray(B_vals, dtype=float)

    def apply_adjoint(self, H_vals: np.ndarray) -> np.ndarray:
        """Ψ*H = A^T · H  (opérateur adjoint : Ψ*H(s) = ∫_s^T H(t) dt)."""
        return self.integration_matrix.T @ np.asarray(H_vals, dtype=float)

    # ── Utilitaire : construction de grille adaptée aux données ────────────

    @staticmethod
    def grid_from_data(data: np.ndarray, n_points: int = 200,
                       quantile_max: float = 0.99) -> np.ndarray:
        """
        Construit une grille uniforme [0, t_max] où t_max est le quantile
        `quantile_max` des données (évite les queues de distribution).
        """
        t_max = float(np.quantile(data, quantile_max))
        return np.linspace(0.0, t_max, n_points)
