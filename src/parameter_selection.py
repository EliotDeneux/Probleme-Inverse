"""
Méthodes de sélection du paramètre de régularisation α.

Cours Doumic 2025, Sections 2.3 et 3.3 :
──────────────────────────────────────────
Le problème central est le compromis biais/variance (eq. 2.13) :
    ||B̂_{ε,α} - B||_Y ≤ ε||R_α|| + ||R_α Ψ B - B||_Y
                          variance     biais

On cherche α*(ε) tel que les deux termes soient équilibrés.

1. Principe de discordance de Morozov (a posteriori, Déf. 7)
   ─────────────────────────────────────────────────────────
   Trouver α tel que ||A B̂_α - H_ε||₂ ≈ τ · ε · √m
   Justification : si B_α est correct, le résidu doit être de l'ordre
   du bruit ε. Méthode a posteriori → ne requiert pas de prior sur B.

   Note (Prop. 5, Bakushinskii) : une méthode a posteriori qui converge
   pour tout z ∈ Im(Ψ) impliquerait Ψ† borné — impossible ici. Donc
   la convergence n'est garantie que si α dépend AUSSI de ε.

2. Validation croisée généralisée (GCV, automatique)
   ──────────────────────────────────────────────────
   GCV(α) = ||A B̂_α - H_ε||² / (m - df(α))²
   où df(α) = tr(A(AᵀA+α²I)⁻¹Aᵀ) est le nb de degrés de liberté effectifs.
   Minimiser GCV(α) donne α sans connaître ε.

3. Courbe en L (heuristique, [5] du cours)
   ────────────────────────────────────────
   Tracer log||B̂_α||_Y vs log||AB̂_α - H_ε||_Z.
   Le paramètre optimal est au « coin » de la courbe en L.
   Méthode non convergente en général (Prop. 5), mais utile en pratique.

4. Paramètre a priori optimal (Prop. 8)
   ─────────────────────────────────────
   Si B ∈ Im(Ψ*) (s=1/2) : α* = O(ε^{1/2})  → convergence O(ε^{1/2})
   Si B ∈ Im(Ψ*Ψ) (s=1)  : α* = O(ε^{1/3})  → convergence O(ε^{2/3})
   Si B ∈ Yˢ              : α* = (ε/δ)^{1/(2s+1)}
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import brentq
from typing import Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# Utilitaires communs
# ═══════════════════════════════════════════════════════════════════════════

def alpha_grid_log(alpha_min: float = 1e-6, alpha_max: float = 1.0,
                   n_points: int = 60) -> np.ndarray:
    """Grille logarithmique de valeurs de α pour les balayages."""
    return np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_points)


def residual_curve(tikhonov, alpha_grid: np.ndarray) -> np.ndarray:
    """||AB̂_α - H_ε||₂ pour chaque α de la grille."""
    return np.array([tikhonov.residual(a) for a in alpha_grid])


# ═══════════════════════════════════════════════════════════════════════════
# 1. Principe de discordance de Morozov
# ═══════════════════════════════════════════════════════════════════════════

class DiscrepancyPrinciple:
    """
    Critère a posteriori de Morozov (cours Section 2.3, Déf. 7).

    Trouve α* tel que : ||AB̂_{α*} - H_ε||₂ = τ · ε · √m

    Avec τ > 1 (typiquement 1.1) : le résidu doit être légèrement supérieur
    au niveau de bruit pour ne pas sur-ajuster.

    Propriété importante (Prop. 4) : si α → 0 quand ε → 0 ET ε||R_α|| → 0,
    la méthode est convergente. Le principe de discordance satisfait ces
    conditions (à condition de connaître ε).

    Parameters
    ----------
    tau       : facteur de sur-correction (τ > 1, défaut 1.1)
    alpha_min : borne inférieure pour la recherche (évite α=0)
    alpha_max : borne supérieure pour la recherche
    """

    def __init__(self, tau: float = 1.1,
                 alpha_min: float = 1e-8,
                 alpha_max: float = 10.0):
        self.tau       = tau
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

    def select(self, tikhonov, noise_level: float) -> float:
        """
        Trouve α* par dichotomie sur la condition de discordance.

        Parameters
        ----------
        tikhonov    : instance TikhonovRegularizer après fit()
        noise_level : ε (niveau de bruit, ε ≈ 1/√n pour données statistiques)
        """
        m      = tikhonov.A.shape[0]
        target = self.tau * noise_level * np.sqrt(m)

        def f_residual(log_alpha):
            return tikhonov.residual(10**log_alpha) - target

        # Si le résidu minimum est déjà > target, retourner alpha_min
        if tikhonov.residual(self.alpha_min) >= target:
            return self.alpha_min

        try:
            log_alpha_star = brentq(
                f_residual,
                np.log10(self.alpha_min),
                np.log10(self.alpha_max),
                xtol=1e-4, rtol=1e-4,
            )
            return float(10**log_alpha_star)
        except ValueError:
            # Pas de zéro trouvé : retourner la valeur minimisant |résidu - target|
            ag = alpha_grid_log(self.alpha_min, self.alpha_max)
            res = residual_curve(tikhonov, ag)
            return float(ag[np.argmin(np.abs(res - target))])

    def curve(self, tikhonov, alpha_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Retourne (alpha_grid, résidus) pour visualisation."""
        return alpha_grid, residual_curve(tikhonov, alpha_grid)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Validation croisée généralisée (GCV)
# ═══════════════════════════════════════════════════════════════════════════

class GeneralizedCrossValidation:
    """
    Critère GCV (Generalized Cross-Validation).

    GCV(α) = m · ||AB̂_α - H_ε||² / (m - df(α))²
           = m · ||AB̂_α - H_ε||² / tr(I - H_α)²

    où H_α = A(AᵀA + α²I)⁻¹Aᵀ est la matrice chapeau.

    Propriété : GCV minimise (asymptotiquement) l'erreur de prédiction
    en validation croisée leave-one-out, sans connaître ε.
    Méthode pratique recommandée quand ε n'est pas disponible.
    """

    def select(self, tikhonov, alpha_grid: Optional[np.ndarray] = None) -> float:
        """Trouve α* minimisant GCV(α)."""
        if alpha_grid is None:
            alpha_grid = alpha_grid_log()

        gcv_vals = self.values(tikhonov, alpha_grid)
        return float(alpha_grid[np.argmin(gcv_vals)])

    def values(self, tikhonov, alpha_grid: np.ndarray) -> np.ndarray:
        """Calcule GCV(α) pour chaque α de la grille."""
        m    = tikhonov.A.shape[0]
        gcv  = np.zeros(len(alpha_grid))
        for i, alpha in enumerate(alpha_grid):
            res = tikhonov.residual(alpha)
            df  = tikhonov.effective_degrees_of_freedom(alpha)
            denom = (m - df)**2
            gcv[i] = (m * res**2) / denom if denom > 1e-10 else np.inf
        return gcv

    def curve(self, tikhonov,
              alpha_grid: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Retourne (alpha_grid, GCV(α)) pour visualisation."""
        if alpha_grid is None:
            alpha_grid = alpha_grid_log()
        return alpha_grid, self.values(tikhonov, alpha_grid)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Courbe en L
# ═══════════════════════════════════════════════════════════════════════════

class LCurveMethod:
    """
    Méthode heuristique de la courbe en L.

    Tracer dans l'espace log-log :
        x(α) = log||AB̂_α - H_ε||₂   (discordance = fidélité aux données)
        y(α) = log||B̂_α||₂           (norme solution = régularisation)

    La courbe présente un « coin » (forte courbure) séparant :
        - Partie gauche (α grand) : sur-régularisé, biais dominant
        - Partie droite (α petit) : sous-régularisé, variance dominante

    Le coin correspond à l'α optimal (selon cette heuristique).

    Mise en garde : Prop. 5 (Bakushinskii) montre que cette méthode
    ne peut pas être universellement convergente. Elle reste utile
    en pratique mais sans garantie théorique.
    """

    def compute_curve(self, tikhonov,
                      alpha_grid: Optional[np.ndarray] = None
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcule les coordonnées (log_residual, log_norm) de la courbe en L.

        Returns
        -------
        alpha_grid    : grille de α
        log_residuals : log||AB̂_α - H_ε||₂
        log_norms     : log||B̂_α||₂
        """
        if alpha_grid is None:
            alpha_grid = alpha_grid_log()

        log_res  = np.zeros(len(alpha_grid))
        log_norm = np.zeros(len(alpha_grid))

        for i, alpha in enumerate(alpha_grid):
            B_hat       = tikhonov.predict(alpha)
            res         = tikhonov.residual(alpha)
            log_res[i]  = np.log10(max(res, 1e-15))
            log_norm[i] = np.log10(max(np.linalg.norm(B_hat), 1e-15))

        return alpha_grid, log_res, log_norm

    def select(self, tikhonov,
               alpha_grid: Optional[np.ndarray] = None) -> float:
        """
        Sélectionne α au point de courbure maximale de la courbe en L.

        Courbure de la courbe paramétrée (x(α), y(α)) :
            κ(α) = (x' y'' - x'' y') / (x'² + y'²)^{3/2}
        """
        alpha_grid, log_res, log_norm = self.compute_curve(tikhonov, alpha_grid)
        n = len(alpha_grid)
        if n < 3:
            return float(alpha_grid[n // 2])

        # Dérivées numériques par différences centrées
        dx  = np.gradient(log_res,  np.arange(n))
        dy  = np.gradient(log_norm, np.arange(n))
        ddx = np.gradient(dx,       np.arange(n))
        ddy = np.gradient(dy,       np.arange(n))

        # Courbure (valeur absolue)
        numer = np.abs(dx * ddy - ddx * dy)
        denom = (dx**2 + dy**2)**1.5
        curv  = np.where(denom > 1e-15, numer / denom, 0.0)

        return float(alpha_grid[np.argmax(curv)])


# ═══════════════════════════════════════════════════════════════════════════
# 4. Paramètre a priori optimal (théorique)
# ═══════════════════════════════════════════════════════════════════════════

def alpha_apriori(epsilon: float, delta: float = 1.0,
                  s: float = 1.0, p: int = 0) -> float:
    """
    Paramètre α optimal a priori pour B ∈ Yˢ (cours Sections 3.3 et 4.1.3).

    D'après la Prop. 8 et le Th. 9 :
        α*(ε, δ, s) = (ε/δ)^{1/(2s+1)}   [pour s ≤ 2p+1 = qualification]
        α*(ε, δ, s) = (ε/δ)^{1/(4p+3)}   [si s > qualification, saturation]

    Convergence correspondante :
        ||B̂_{α*} - B||_Y = O(ε^{2s/(2s+1)})

    Parameters
    ----------
    epsilon : niveau de bruit ε ≈ 1/√n
    delta   : norme a priori ||B||_{Yˢ} ≤ δ
    s       : régularité de B (exposant de Sobolev)
    p       : ordre de Tikhonov (0=classique, 1=généralisé p=1, ...)
    """
    qualification = 2 * p + 1
    s_eff = min(s, qualification)  # saturation pour s > qualification
    return (epsilon / delta) ** (1.0 / (2 * s_eff + 1))


def theoretical_convergence_rate(n_values: np.ndarray, s: float = 1.0,
                                 C: float = 1.0) -> np.ndarray:
    """
    Taux de convergence théorique O(n^{-s/(2s+1)}) (cours Section 5.2.1).

    Utilisé pour comparer empiriquement avec les résultats numériques.
    ε = 1/√n → ||B̂ - B|| = O(ε^{2s/(2s+1)}) = O(n^{-s/(2s+1)}).
    """
    return C * n_values ** (-s / (2 * s + 1))
