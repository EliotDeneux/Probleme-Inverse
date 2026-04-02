"""
Estimation de densité et du hasard cumulé à partir de données.

Méthodes implémentées (cours Doumic 2025, Chapitre 5) :
────────────────────────────────────────────────────────
1. KDE (Kernel Density Estimator, Section 5.2)
   f̂_h(t) = (1/nh) Σᵢ K((t - Tᵢ)/h)
   Sélection de h : règle de Silverman, Scott, ou validation croisée.

2. Estimateur de Nelson-Aalen du hasard cumulé Ĥ(t)
   Pour des données i.i.d. : Ĥ(t) = Σ_{Tᵢ≤t} 1/(n - rang(Tᵢ) + 1)
   Version avec troncature gauche (modèle taille) : at-risk corrigé.

3. Mollificateur (Section 5.1.2)
   Ĥ_σ = ρ_σ ★ Ĥ_NA  (lissage gaussien avant différentiation)

4. Estimateur direct du taux de hasard (KDE)
   B̂(t) = f̂_h(t) / Ŝ(t)  où  Ŝ(t) = 1 - F̂ₙ(t)

Lien avec le problème inverse :
   Ĥ_NA constitue la « mesure bruitée » zε de H = ΨB.
   Noise level : ε ≈ C/√n  (TCL, cf. Section 5.2.1)
   → Ĥ sert d'entrée aux méthodes de régularisation (src/linear_inverse.py).
"""

from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# Noyaux et sélection de fenêtre
# ═══════════════════════════════════════════════════════════════════════════

def _gaussian_kernel(u: np.ndarray) -> np.ndarray:
    """K(u) = (1/√(2π)) exp(-u²/2)  (noyau gaussien standard)."""
    return np.exp(-0.5 * u**2) / np.sqrt(2.0 * np.pi)


def _epanechnikov_kernel(u: np.ndarray) -> np.ndarray:
    """K(u) = (3/4)(1 - u²) · 𝟏(|u| ≤ 1)  (noyau optimal au sens MISE)."""
    return np.where(np.abs(u) <= 1.0, 0.75 * (1.0 - u**2), 0.0)


_KERNELS = {
    'gaussian'    : _gaussian_kernel,
    'epanechnikov': _epanechnikov_kernel,
}


def bandwidth_silverman(data: np.ndarray) -> float:
    """
    Règle empirique de Silverman (ordre optimal pour distribution gaussienne).
    h* = 0.9 · min(σ, IQR/1.34) · n^{-1/5}

    Taux optimal : h = O(n^{-1/(2m+1)}) pour f ∈ Wm,∞, ici m=1 → n^{-1/5}.
    """
    n   = len(data)
    std = np.std(data, ddof=1)
    iqr = float(np.percentile(data, 75) - np.percentile(data, 25))
    s   = min(std, iqr / 1.34) if iqr > 0 else std
    return 0.9 * s * n**(-0.2)


def bandwidth_scott(data: np.ndarray) -> float:
    """Règle de Scott : h* = 1.06 · σ · n^{-1/5}."""
    return 1.06 * np.std(data, ddof=1) * len(data)**(-0.2)


def bandwidth_cross_validation(data: np.ndarray, kernel: str = 'gaussian',
                                n_h: int = 30) -> float:
    """
    Sélection de h par validation croisée leave-one-out.

    Maximise la log-vraisemblance LOO :
        ℓ(h) = Σᵢ log f̂_{h,-i}(Tᵢ)
    où f̂_{h,-i} est le KDE calculé sans l'observation i.

    Complexité : O(n² · n_h) → adapté pour n ≤ 5000.
    """
    K = _KERNELS[kernel]
    n = len(data)
    h_grid = np.logspace(np.log10(0.01 * np.std(data)),
                         np.log10(2.0 * np.std(data)), n_h)
    scores = np.full(n_h, -np.inf)

    for ih, h in enumerate(h_grid):
        # LOO : f̂_{-i}(Tᵢ) = (1/((n-1)h)) Σ_{j≠i} K((Tᵢ-Tⱼ)/h)
        # Vecteur de taille (n, n) → diagonale exclue
        U   = (data[:, None] - data[None, :]) / h    # (n, n)
        KU  = K(U)                                   # (n, n)
        np.fill_diagonal(KU, 0.0)
        loo = KU.sum(axis=1) / ((n - 1) * h)        # (n,)
        loo = np.maximum(loo, 1e-15)
        scores[ih] = np.mean(np.log(loo))

    return float(h_grid[np.argmax(scores)])


# ═══════════════════════════════════════════════════════════════════════════
# Estimateur à noyau de densité
# ═══════════════════════════════════════════════════════════════════════════

class KernelDensityEstimator:
    """
    Estimateur à noyau f̂_h (cours Section 5.2).

    f̂_h(t) = (1/(nh)) Σᵢ K((t - Tᵢ)/h)

    Le biais est d'ordre h^m (pour f ∈ W^{m,∞}) et la variance d'ordre 1/(nh).
    L'erreur MISE optimale est O(n^{-m/(2m+1)}) pour h* ∝ n^{-1/(2m+1)}.

    Parameters
    ----------
    kernel    : 'gaussian' | 'epanechnikov'
    bandwidth : float, 'silverman', 'scott', ou 'cv'
    """

    def __init__(self, kernel: str = 'gaussian',
                 bandwidth: str | float = 'silverman'):
        self.kernel    = kernel
        self.bandwidth = bandwidth
        self.h_: Optional[float] = None
        self.data_: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray) -> 'KernelDensityEstimator':
        self.data_ = np.asarray(data, dtype=float)
        n = len(self.data_)

        if   self.bandwidth == 'silverman':
            self.h_ = bandwidth_silverman(self.data_)
        elif self.bandwidth == 'scott':
            self.h_ = bandwidth_scott(self.data_)
        elif self.bandwidth == 'cv':
            # LOO-CV seulement sur un sous-échantillon si n > 2000
            sub = self.data_ if n <= 2000 else self.data_[
                np.random.choice(n, 2000, replace=False)]
            self.h_ = bandwidth_cross_validation(sub, kernel=self.kernel)
        else:
            self.h_ = float(self.bandwidth)
        return self

    def predict(self, grid: np.ndarray) -> np.ndarray:
        """
        Évalue f̂_h sur la grille (vectorisé, O(m·n) en mémoire).
        Implémentation directe de la formule du cours (Section 5.2).
        """
        if self.data_ is None:
            raise RuntimeError("Appelez fit() d'abord.")
        K = _KERNELS[self.kernel]
        # U[i, j] = (grid[i] - data[j]) / h
        U = (grid[:, None] - self.data_[None, :]) / self.h_
        return K(U).mean(axis=1) / self.h_

    @property
    def bandwidth_value(self) -> float:
        return self.h_


# ═══════════════════════════════════════════════════════════════════════════
# Estimateur de Nelson-Aalen du hasard cumulé
# ═══════════════════════════════════════════════════════════════════════════

class NelsonAalanEstimator:
    """
    Estimateur non-paramétrique du hasard cumulé H(t) = ∫₀ᵗ B(s) ds.

    Pour données i.i.d. (modèles âge et incrément) :
        Ĥ(t) = Σ_{Tᵢ ≤ t}  dᵢ / nᵢ
    où dᵢ = nb de divisions en Tᵢ,  nᵢ = nb d'individus à risque en Tᵢ.

    Pour données avec troncature gauche (modèle taille) :
        nᵢ = #{k : X_ub_k < Tᵢ ≤ X_ud_k}  (individus « à risque »).

    Propriété : Ĥ est un estimateur sans biais de H (cf. théorie de la survie).

    Lien problème inverse : Ĥ est la « mesure bruitée » zε de H = ΨB,
    avec bruit de niveau ε ≈ C/√n par le théorème central limite.
    """

    def __init__(self):
        self.jump_times_: Optional[np.ndarray] = None
        self.H_jumps_: Optional[np.ndarray]   = None

    def fit(self, observations: np.ndarray,
            entry_times: Optional[np.ndarray] = None) -> 'NelsonAalanEstimator':
        """
        Parameters
        ----------
        observations : T_i = âges, incréments, ou tailles à la division (X_ud)
        entry_times  : X_ub (tailles à la naissance) pour le modèle taille ;
                       None pour les modèles âge et incrément.
        """
        n   = len(observations)
        T   = np.asarray(observations, dtype=float)
        T_u = np.unique(T)

        increments = np.zeros(len(T_u))

        if entry_times is None:
            # ── Données i.i.d. (âge, incrément) ──────────────────────────
            # Trier et compter : nᵢ = n - #{j : Tⱼ < tᵢ}
            T_sorted = np.sort(T)
            for idx, t in enumerate(T_u):
                d_t = np.sum(T == t)               # événements en t
                n_t = np.sum(T_sorted >= t)        # à risque en t
                increments[idx] = d_t / n_t if n_t > 0 else 0.0
        else:
            # ── Troncature gauche (modèle taille) ────────────────────────
            # Entrée au moment X_ub, événement au moment X_ud
            L = np.asarray(entry_times, dtype=float)   # tailles naissance
            for idx, t in enumerate(T_u):
                d_t = np.sum(T == t)
                n_t = np.sum((L < t) & (T >= t))
                increments[idx] = d_t / n_t if n_t > 0 else 0.0

        # Hasard cumulé en escalier (commence à 0)
        self.jump_times_ = np.concatenate([[0.0], T_u])
        self.H_jumps_    = np.concatenate([[0.0], np.cumsum(increments)])
        return self

    def predict(self, grid: np.ndarray) -> np.ndarray:
        """Interpole Ĥ (en escalier) sur la grille fournie."""
        return np.interp(grid, self.jump_times_, self.H_jumps_,
                         left=0.0, right=self.H_jumps_[-1])

    def smooth(self, grid: np.ndarray, sigma_grid: float = 5.0) -> np.ndarray:
        """
        Mollification gaussienne : Ĥ_σ = ρ_σ ★ Ĥ_NA  (Section 5.1.2).

        Paramètre sigma_grid exprimé en nombre de pas de grille.
        Réduit le bruit Ĥ avant différentiation numérique / régularisation.
        """
        H_raw = self.predict(grid)
        return gaussian_filter1d(H_raw, sigma=sigma_grid)

    def noise_level_estimate(self, n: int) -> float:
        """
        Estimation du niveau de bruit ε en norme L² (Section 5.2.1).

        Par le TCL : ε ≈ C/√n  (équivalence déterministe–stochastique,
        cours eq. finale Sec. 5.2.1).
        """
        return 1.0 / np.sqrt(n)


# ═══════════════════════════════════════════════════════════════════════════
# Estimateur direct KDE du taux de hasard
# ═══════════════════════════════════════════════════════════════════════════

class KDEHazardEstimator:
    """
    Estimation directe du taux de hasard B par la formule :
        B̂(t) = f̂_h(t) / Ŝ(t)
    où f̂_h est le KDE de f et Ŝ(t) = 1 - F̂_n(t) la survie empirique.

    Méthode simple et rapide, mais instable dans les queues (Ŝ → 0).
    Sert de référence de comparaison pour les méthodes de régularisation.

    Parameters
    ----------
    bandwidth : sélection de h pour f̂
    kernel    : noyau pour le KDE
    tail_clip : fraction de données à ignorer pour stabilité numérique
    """

    def __init__(self, bandwidth: str | float = 'silverman',
                 kernel: str = 'gaussian', tail_clip: float = 0.02):
        self.kde       = KernelDensityEstimator(kernel=kernel, bandwidth=bandwidth)
        self.tail_clip = tail_clip
        self.data_     : Optional[np.ndarray] = None
        self.entry_    : Optional[np.ndarray] = None

    def fit(self, observations: np.ndarray,
            entry_times: Optional[np.ndarray] = None) -> 'KDEHazardEstimator':
        self.data_  = np.asarray(observations, dtype=float)
        self.entry_ = entry_times
        self.kde.fit(self.data_)
        return self

    def predict_f(self, grid: np.ndarray) -> np.ndarray:
        """Densité estimée f̂_h(t) par KDE."""
        return self.kde.predict(grid)

    def predict_S(self, grid: np.ndarray) -> np.ndarray:
        """
        Survie empirique Ŝ(t) = 1 - F̂_n(t).

        Pour le modèle taille (entry_times fourni), la survie est calculée
        conditionnellement aux observations à risque.
        """
        if self.entry_ is None:
            S = np.array([np.mean(self.data_ > t) for t in grid])
        else:
            # Kaplan-Meier approché : produit des 1 - dᵢ/nᵢ
            na = NelsonAalanEstimator().fit(self.data_, self.entry_)
            H  = na.predict(grid)
            S  = np.exp(-H)
        return np.maximum(S, 1e-8)

    def predict_B(self, grid: np.ndarray) -> np.ndarray:
        """
        Taux de hasard B̂(t) = f̂_h(t) / Ŝ(t).

        La grille est tronquée au quantile (1 - tail_clip) pour éviter
        l'explosion numérique en queue de distribution.
        """
        t_max = np.quantile(self.data_, 1.0 - self.tail_clip)
        mask  = grid <= t_max
        f = self.predict_f(grid)
        S = self.predict_S(grid)
        B = np.where(mask, f / S, np.nan)
        return B
