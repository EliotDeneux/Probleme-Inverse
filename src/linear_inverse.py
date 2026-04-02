"""
Méthodes de régularisation pour l'opérateur d'intégration.

Contexte mathématique (cours Doumic 2025, Chapitres 3-4) :
──────────────────────────────────────────────────────────
Problème inverse : estimer B = H' à partir d'une observation bruitée H_ε ≈ H = ΨB.

L'opérateur Ψ : L²([0,T]) → L²([0,T]),  (ΨB)(t) = ∫₀ᵗ B(s)ds  est linéaire,
compact, injectif à image dense — le problème est donc MAL POSÉ (Hadamard).

SVD analytique de Ψ sur [0,T] (cours Section 3.4.1, solution de l'exercice) :
    σⱼ = 2T/[π(2j+1)]        (valeurs singulières, décroissent en O(j⁻¹))
    eⱼ(t) = √(2/T) cos(λⱼt)  (vecteurs singuliers droits, λⱼ = π(2j+1)/(2T))
    fⱼ(t) = √(2/T) sin(λⱼt)  (vecteurs singuliers gauches)

    Vérification : Ψeⱼ = σⱼfⱼ  et  Ψ*fⱼ = σⱼeⱼ

Degré de mal-positude : σⱼ = O(j⁻¹) → degré 1 (Section 3.3, Déf. 9).

Estimateurs régularisés génériques (Th. 5) :
    B̂_α(t) = Σⱼ r(α, σⱼ)/σⱼ · ⟨H_ε, fⱼ⟩_{L²} · eⱼ(t)

    r(α, σ) = 𝟏_{σ≥α}              → SVD tronquée    (Section 3.2, eq. 3.3)
    r(α, σ) = σ²/(σ²+α²)           → Tikhonov classique (Section 4.1, p=0)
    r(α, σ) = 1/(1+α^{4p+2}σ^{-4p-2}) → Tikhonov généralisé (Th. 9, qualification 2p+1)

Convergence (prior B ∈ Yˢ, bruit ε ~ 1/√n) :
    ||B̂_α - B||_{L²} = O(ε^{2s/(2s+1)}) = O(n^{-s/(2s+1)})
    Tikhonov classique (qualification 1) : optimal pour s ≤ 1 → O(ε^{2/3})
    Tikhonov généralisé p=1 (qualif. 3)  : optimal pour s ≤ 3 → O(ε^{6/7})
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# SVD analytique de l'opérateur d'intégration sur [0, T]
# ═══════════════════════════════════════════════════════════════════════════

class AnalyticSVD:
    """
    SVD analytique de Ψ : B → ∫₀^· B sur [0, T].

    Calcule les J+1 premiers éléments du système singulier (eⱼ, fⱼ, σⱼ).
    Voir cours Section 3.4.1 pour la dérivation complète.

    Parameters
    ----------
    T       : longueur du domaine [0, T]
    J_max   : nombre de modes (les σⱼ décroissent, tronquer évite le bruit)
    """

    def __init__(self, T: float, J_max: int = 100):
        self.T     = T
        self.J_max = J_max
        j          = np.arange(J_max + 1)
        # σⱼ = 2T/[π(2j+1)] — décroissent en O(j⁻¹)
        self.sigma    = 2.0 * T / (np.pi * (2 * j + 1))
        self.lambda_j = np.pi * (2 * j + 1) / (2.0 * T)   # λⱼ = 1/σⱼ · ... (non, λⱼ = π(2j+1)/(2T))

    def e_j(self, grid: np.ndarray, j: int) -> np.ndarray:
        """eⱼ(t) = √(2/T) cos(λⱼ t)  (vecteur singulier droit)."""
        return np.sqrt(2.0 / self.T) * np.cos(self.lambda_j[j] * grid)

    def f_j(self, grid: np.ndarray, j: int) -> np.ndarray:
        """fⱼ(t) = √(2/T) sin(λⱼ t)  (vecteur singulier gauche)."""
        return np.sqrt(2.0 / self.T) * np.sin(self.lambda_j[j] * grid)

    def inner_product_L2(self, u: np.ndarray, v: np.ndarray,
                         grid: np.ndarray) -> float:
        """⟨u, v⟩_{L²([0,T])} ≈ ∫₀ᵀ u(t)v(t) dt  (trapèzes)."""
        return float(np.trapz(u * v, grid))

    def picard_coefficients(self, H_eps: np.ndarray,
                            grid: np.ndarray) -> np.ndarray:
        """
        Coefficients de Picard ĉⱼ = ⟨H_ε, fⱼ⟩_{L²}.

        Le critère de Picard (Corol. 1) : H_ε ∈ D(Ψ†) ⟺ Σⱼ ĉⱼ²/σⱼ² < ∞.
        En pratique, ĉⱼ/σⱼ doit décroître avec j pour que l'inversion soit
        numériquement stable.
        """
        coeffs = np.zeros(self.J_max + 1)
        for j in range(self.J_max + 1):
            fj     = self.f_j(grid, j)
            coeffs[j] = self.inner_product_L2(H_eps, fj, grid)
        return coeffs

    def reconstruct(self, coeffs: np.ndarray, filter_r: np.ndarray,
                    grid: np.ndarray) -> np.ndarray:
        """
        B̂(t) = Σⱼ r(α, σⱼ)/σⱼ · ĉⱼ · eⱼ(t)

        Parameters
        ----------
        coeffs   : coefficients de Picard ĉⱼ = ⟨H_ε, fⱼ⟩
        filter_r : valeurs du filtre r(α, σⱼ) pour j = 0,...,J_max
        grid     : grille de reconstruction
        """
        B_hat = np.zeros(len(grid))
        for j in range(self.J_max + 1):
            if filter_r[j] == 0.0:
                continue
            B_hat += (filter_r[j] / self.sigma[j]) * coeffs[j] * self.e_j(grid, j)
        return B_hat


# ═══════════════════════════════════════════════════════════════════════════
# SVD tronquée (filtre de troncature)
# ═══════════════════════════════════════════════════════════════════════════

class TruncatedSVD:
    """
    SVD tronquée via la SVD analytique connue.

    Filtre : r(α, σ) = 𝟏_{σ≥α}  (cours Section 3.2, eq. 3.3)

    Qualification = ∞ (pas de saturation de l'ordre optimal).
    Convergence : O(ε^{2s/(2s+1)}) pour B ∈ Yˢ, optimal.

    Le paramètre de régularisation α est ici le seuil de troncature sur
    les valeurs singulières (équivalent à N_α modes conservés).
    """

    def __init__(self, J_max: int = 100):
        self.J_max  = J_max
        self.svd_   : Optional[AnalyticSVD] = None
        self.coeffs_: Optional[np.ndarray]  = None
        self.grid_  : Optional[np.ndarray]  = None

    def fit(self, H_eps: np.ndarray, grid: np.ndarray) -> 'TruncatedSVD':
        """Calcule les coefficients de Picard à partir de H_ε et de la grille."""
        T          = grid[-1] - grid[0]
        self.svd_  = AnalyticSVD(T, J_max=self.J_max)
        self.grid_ = grid
        self.coeffs_ = self.svd_.picard_coefficients(H_eps, grid)
        return self

    def predict(self, alpha: float) -> np.ndarray:
        """
        B̂_α via filtre de troncature r(α, σⱼ) = 𝟏_{σⱼ ≥ α}.

        Nombre de modes conservés : N_α = max{j : σⱼ ≥ α}.
        Plus α petit → plus de modes → moins de biais mais plus de variance.
        """
        filter_r = (self.svd_.sigma >= alpha).astype(float)
        return self.svd_.reconstruct(self.coeffs_, filter_r, self.grid_)

    def predict_n_modes(self, N: int) -> np.ndarray:
        """Version avec N modes explicitement fixés (N_α ↔ 1/σ_{N} ↔ α)."""
        filter_r = np.zeros(self.J_max + 1)
        filter_r[:min(N, self.J_max + 1)] = 1.0
        return self.svd_.reconstruct(self.coeffs_, filter_r, self.grid_)

    def sigma_values(self) -> np.ndarray:
        """Retourne les valeurs singulières σⱼ (utile pour L-curve)."""
        return self.svd_.sigma if self.svd_ is not None else np.array([])

    def picard_plot_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Retourne (j, σⱼ, |ĉⱼ|) pour le critère de Picard."""
        j  = np.arange(len(self.coeffs_))
        return j, self.svd_.sigma, np.abs(self.coeffs_)


# ═══════════════════════════════════════════════════════════════════════════
# Régularisation de Tikhonov (numérique, via matrice)
# ═══════════════════════════════════════════════════════════════════════════

class TikhonovRegularizer:
    """
    Régularisation de Tikhonov via la discrétisation matricielle.

    Problème variationnel (cours Section 4.1, eq. 4.1) :
        B̂_α = argmin_{B} ½||AB - H_ε||² + (α²/2)||LB||²

    Équation normale (Th. 7, eq. 4.2) :
        (AᵀA + α² LᵀL) B̂_α = Aᵀ H_ε

    Où :
        A  = matrice d'intégration (opérateur Ψ discrétisé)
        L  = opérateur de régularisation :
             p=0 : L = I (Tikhonov classique, qualification 1)
             p=1 : L = D (différences finies d'ordre 1, qualification 3)
             p=2 : L = D² (différences finies d'ordre 2, qualification 5)

    Via SVD de A (Th. 9, Section 4.1.3) :
        r_p(α, σ) = σ²/(σ² + α² σ^{-4p}) = 1/(1 + α² σ^{-4p-2})
        qualification = 2p+1

    Parameters
    ----------
    A    : matrice d'intégration (m×m)
    p    : ordre de régularisation (0=classique, 1=dérivée 1ère, 2=dérivée 2ème)
    """

    def __init__(self, A: np.ndarray, p: int = 0):
        self.A   = A
        self.p   = p
        self.m   = A.shape[0]
        self._setup_regularization_operator()
        # Pré-factorisation SVD de A pour tous les α
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        self._U  = U
        self._s  = s
        self._Vt = Vt
        self._AtA = A.T @ A
        self._Atz  = None   # Aᵀ H_ε (calculé à fit)

    def _setup_regularization_operator(self):
        """
        Construit L^T L pour l'ordre p choisi.
        p=0 : L = I → LᵀL = I
        p=1 : L = D₁ (différences d'ordre 1) → LᵀL = D₁ᵀD₁
        p=2 : L = D₂ (différences d'ordre 2) → LᵀL = D₂ᵀD₂
        """
        m = self.m
        if self.p == 0:
            self._LtL = np.eye(m)
        elif self.p == 1:
            # D₁ : différences finies centrées (matrice (m-1)×m)
            D = np.zeros((m - 1, m))
            for i in range(m - 1):
                D[i, i]     = -1.0
                D[i, i + 1] =  1.0
            self._LtL = D.T @ D
        elif self.p == 2:
            D2 = np.zeros((m - 2, m))
            for i in range(m - 2):
                D2[i, i], D2[i, i+1], D2[i, i+2] = 1.0, -2.0, 1.0
            self._LtL = D2.T @ D2
        else:
            raise ValueError(f"Ordre p={self.p} non supporté (0, 1 ou 2).")

    def fit(self, H_eps: np.ndarray) -> 'TikhonovRegularizer':
        """Stocke Aᵀ H_ε (membre de droite de l'équation normale)."""
        self._Atz = self.A.T @ H_eps
        self._H_eps = H_eps
        return self

    def predict(self, alpha: float) -> np.ndarray:
        """
        Résout (AᵀA + α² LᵀL) B̂_α = Aᵀ H_ε.

        Paramètre α contrôle le compromis biais/variance (eq. 2.13) :
            - Grand α : forte régularisation → grand biais, petite variance
            - Petit  α : faible régularisation → petit biais, grande variance
        """
        if self._Atz is None:
            raise RuntimeError("Appelez fit() d'abord.")
        M = self._AtA + alpha**2 * self._LtL
        return np.linalg.solve(M, self._Atz)

    def predict_via_svd(self, alpha: float) -> np.ndarray:
        """
        Formule explicite via SVD de A (Section 4.1.3, uniquement p=0).
        B̂_α = Σⱼ [σⱼ²/(σⱼ²+α²)] · (1/σⱼ) · ⟨H_ε, uⱼ⟩ · vⱼ
             = V · diag(σⱼ/(σⱼ²+α²)) · Uᵀ H_ε
        """
        if self.p != 0:
            raise ValueError("La formule SVD est seulement disponible pour p=0.")
        filter_coeffs = self._s / (self._s**2 + alpha**2)
        return self._Vt.T @ (filter_coeffs * (self._U.T @ self._H_eps))

    def residual(self, alpha: float) -> float:
        """||A B̂_α - H_ε||₂  (résidu pour critère de discordance)."""
        B_hat = self.predict(alpha)
        return float(np.linalg.norm(self.A @ B_hat - self._H_eps))

    def filter_function(self, alpha: float) -> np.ndarray:
        """
        Valeurs du filtre r_p(α, σⱼ) sur les valeurs singulières de A.
        Permet de visualiser l'effet de la régularisation (Section 4.1.3).
        """
        sigma = self._s
        if self.p == 0:
            return sigma**2 / (sigma**2 + alpha**2)
        else:
            # Approximation : utilise les valeurs singulières de A
            return sigma**2 / (sigma**2 + alpha**2 * sigma**(-4 * self.p))

    def effective_degrees_of_freedom(self, alpha: float) -> float:
        """
        df(α) = tr(A(AᵀA + α²LᵀL)⁻¹Aᵀ) = Σⱼ σⱼ²/(σⱼ²+α²)  [pour p=0].
        Utilisé par le critère GCV.
        """
        if self.p == 0:
            return float(np.sum(self._s**2 / (self._s**2 + alpha**2)))
        else:
            M_inv = np.linalg.inv(self._AtA + alpha**2 * self._LtL)
            H_mat = self.A @ M_inv @ self.A.T
            return float(np.trace(H_mat))
