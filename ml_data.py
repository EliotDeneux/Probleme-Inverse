"""
ml_data.py — Generation de donnees synthetiques pour l'apprentissage d'operateur.

STRATEGIE DE GENERATION
════════════════════════
Pour chaque exemple d'entrainement :

  1. Tirer B depuis une famille parametrique aleatoire (8 familles, poids varies)
  2. Calculer H_true = Psi B  sur la grille [0,1] (integration trapezoidale)
  3. Tirer n_obs observations de f(t) = B(t)*S(t) par inversion de CDF
  4. Calculer H_NA (Nelson-Aalen bruyte) sur la meme grille
  5. Lisser H_NA par mollification gaussienne
  6. Normaliser : H_norm = H / max(H),  B_norm = B / mean(B)
     -> le reseau est invariant a l'echelle

FAMILLES DE TAUX B
══════════════════
Choisies pour couvrir les formes biologiquement plausibles et les cas
limites mathematiques :

  A. Constant     B(t) = lambda          (adder, memoryless, Exp)
  B. Weibull      B(t) = k*l*(l*t)^{k-1} (timer, k in [0.7, 5])
  C. Seuil        B(t) = eps + l*1(t>=t0) (delai minimal + Exp)
  D. Lineaire     B(t) = a*t + b          (sizer-like)
  E. Sigmoide     B(t) = L/(1+exp(-k(t-t0))) (switch progressif)
  F. Mix gaussien B(t) = base + sum_i A_i*gauss_i(t) (forme quelconque)
  G. Puissance    B(t) = beta*t^gamma + b (intermediaire)
  H. Spline rand. interpolation + lissage (formes non parametriques)

Toutes verifient : int_0^inf B = +inf (division certaine)

CURRICULUM LEARNING
════════════════════
Phase 1 : n_obs >= 2000 (bruit faible) -> le reseau apprend la structure
Phase 2 : n_obs >= 300  (bruit fort)   -> le reseau apprend la robustesse

Voir ml_train.py pour l'utilisation dans la boucle d'entrainement.
"""

from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import Dict, Optional, Tuple

# ── Grille normalisee ────────────────────────────────────────────────────
M_GRID: int        = 128
GRID  : np.ndarray = np.linspace(0.0, 1.0, M_GRID)


def _build_psi(m: int) -> np.ndarray:
    """Matrice d'integration trapezoidale Psi sur [0, 1], taille (m, m)."""
    g = np.linspace(0.0, 1.0, m)
    h = np.diff(g)
    A = np.zeros((m, m))
    for k in range(m - 1):
        A[k + 1:, k]     += h[k] / 2.0
        A[k + 1:, k + 1] += h[k] / 2.0
    return A


# Pre-calculee une seule fois au chargement du module
PSI: np.ndarray = _build_psi(M_GRID)


# ═══════════════════════════════════════════════════════════════════════════
# Familles de taux B sur [0, 1]
# ═══════════════════════════════════════════════════════════════════════════

def _B_constant(rng: np.random.Generator, t: np.ndarray) -> np.ndarray:
    """B(t) = lambda ~ U[0.5, 15].  Cas memoryless (exponentiel)."""
    return np.full_like(t, rng.uniform(0.5, 15.0))


def _B_weibull(rng: np.random.Generator, t: np.ndarray) -> np.ndarray:
    """
    B(t) = k * lambda * (lambda*t)^{k-1},  k ~ U[0.7, 5].
    k > 1 : taux croissant (timer biologique, accumulation de regulateur).
    k = 1 : exponentiel.  k < 1 : taux decroissant (cas atypique).
    """
    k   = rng.uniform(0.7, 5.0)
    lam = rng.uniform(0.5, 3.0)
    return k * lam ** k * np.maximum(t, 1e-9) ** (k - 1.0)


def _B_step(rng: np.random.Generator, t: np.ndarray) -> np.ndarray:
    """B(t) = eps + lambda * 1(t >= t0) — delai minimal avant division."""
    t0  = rng.uniform(0.05, 0.55)
    lam = rng.uniform(1.0, 12.0)
    eps = rng.uniform(0.01, 0.5)
    return np.where(t >= t0, lam, eps)


def _B_linear(rng: np.random.Generator, t: np.ndarray) -> np.ndarray:
    """B(t) = a*t + b,  a > 0,  b > 0 — croissant lineaire (sizer-like)."""
    a = rng.uniform(2.0, 30.0)
    b = rng.uniform(0.1, 3.0)
    return a * t + b


def _B_sigmoid(rng: np.random.Generator, t: np.ndarray) -> np.ndarray:
    """B(t) = L / (1 + exp(-k*(t-t0))) — switch progressif."""
    L   = rng.uniform(2.0, 15.0)
    k   = rng.uniform(5.0, 40.0)
    t0  = rng.uniform(0.1, 0.7)
    return L / (1.0 + np.exp(np.clip(-k * (t - t0), -200, 200)))


def _B_gauss_mix(rng: np.random.Generator, t: np.ndarray) -> np.ndarray:
    """B(t) = base + sum_i A_i * exp(-0.5*((t-mu_i)/s_i)^2) — forme generale."""
    n_comp = rng.integers(1, 4)
    B      = np.full_like(t, rng.uniform(0.2, 2.0))
    for _ in range(n_comp):
        A  = rng.uniform(2.0, 10.0)
        mu = rng.uniform(0.05, 0.85)
        sg = rng.uniform(0.03, 0.20)
        B  = B + A * np.exp(-0.5 * ((t - mu) / sg) ** 2)
    return B


def _B_power(rng: np.random.Generator, t: np.ndarray) -> np.ndarray:
    """B(t) = beta * t^gamma + b — intermédiaire entre constant et lineaire."""
    beta  = rng.uniform(1.0, 20.0)
    gamma = rng.uniform(0.3, 3.0)
    b     = rng.uniform(0.05, 1.0)
    return beta * np.maximum(t, 1e-9) ** gamma + b


def _B_spline(rng: np.random.Generator, t: np.ndarray) -> np.ndarray:
    """
    Interpolation de points de controle aleatoires + lissage gaussien.
    Couvre les formes non parametriques que les autres familles ne captent pas.
    """
    n_ctrl = rng.integers(4, 9)
    t_ctrl = np.linspace(0.0, 1.0, n_ctrl)
    # 50% : monotone croissante (biologique) | 50% : forme quelconque
    if rng.random() < 0.5:
        v_ctrl = np.cumsum(rng.exponential(2.0, n_ctrl)) + rng.uniform(0.1, 1.0)
    else:
        v_ctrl = rng.uniform(0.2, 12.0, n_ctrl)
    v_ctrl = np.clip(v_ctrl, 0.1, 20.0)
    B      = np.interp(t, t_ctrl, v_ctrl)
    return np.maximum(gaussian_filter1d(B, sigma=2.5), 0.05)


_FAMILIES = [
    _B_constant, _B_weibull, _B_step,    _B_linear,
    _B_sigmoid,  _B_gauss_mix, _B_power, _B_spline,
]
# Poids : plus de poids aux formes biologiquement observees
_WEIGHTS = np.array([0.10, 0.20, 0.12, 0.12, 0.12, 0.12, 0.11, 0.11])
_WEIGHTS = _WEIGHTS / _WEIGHTS.sum()


# ═══════════════════════════════════════════════════════════════════════════
# Simulation d'observations
# ═══════════════════════════════════════════════════════════════════════════

def sample_B(rng: np.random.Generator) -> np.ndarray:
    """Tire un taux B sur GRID depuis une famille parametrique aleatoire."""
    fam = rng.choice(len(_FAMILIES), p=_WEIGHTS)
    return np.clip(_FAMILIES[fam](rng, GRID), 1e-3, 100.0)


def B_to_H(B: np.ndarray) -> np.ndarray:
    """H = Psi B  via la matrice d'integration pre-calculee."""
    return PSI @ B


def simulate_T(B: np.ndarray, n: int,
               rng: np.random.Generator) -> np.ndarray:
    """
    Simule n observations i.i.d. de f(t) = B(t)*S(t) par inversion de CDF.

    F(t) = 1 - exp(-H(t)) est calculee sur GRID,
    puis on tire u ~ U[0, F(T_max)] et on interpole F^{-1}(u).
    """
    H     = B_to_H(B)
    F     = 1.0 - np.exp(-np.maximum(H, 0.0))
    F_max = float(F[-1])
    if F_max < 1e-6:
        return np.full(n, 0.5)
    u = rng.uniform(0.0, F_max * 0.9999, n)
    return np.interp(u, F, GRID)


def nelson_aalen_on_grid(T: np.ndarray,
                          entry: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Calcule H_NA (Nelson-Aalen bruite) sur GRID.

    entry : tailles a la naissance normalisees (modele taille, troncature gauche).
            None pour les modeles age et increment (donnees iid).
    """
    T_uniq = np.unique(T)
    jumps  = np.zeros(len(T_uniq))

    for idx, t in enumerate(T_uniq):
        d_t = int(np.sum(T == t))
        if entry is None:
            n_t = int(np.sum(T >= t))
        else:
            n_t = int(np.sum((entry < t) & (T >= t)))
        jumps[idx] = d_t / n_t if n_t > 0 else 0.0

    t_jump = np.concatenate([[0.0], T_uniq])
    H_jump = np.concatenate([[0.0], np.cumsum(jumps)])
    H_raw  = np.interp(GRID, t_jump, H_jump)
    # Mollification gaussienne : reduit l'effet escalier de H_NA
    return gaussian_filter1d(H_raw, sigma=2.5)


def normalize_pair(H_obs: np.ndarray, B: np.ndarray
                   ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Normalise (H, B) pour l'apprentissage.

    H_norm = H / H_scale   (H_scale = max H)   -> H dans [0, 1]
    B_norm = B / B_scale   (B_scale = mean B)  -> B centree autour de 1

    La denormalisation a l'inference utilise les scales conservees.
    """
    H_scale = float(max(H_obs.max(), 1e-6))
    B_scale = float(max(B.mean(),    1e-6))
    return H_obs / H_scale, B / B_scale, {'H': H_scale, 'B': B_scale}


# ═══════════════════════════════════════════════════════════════════════════
# Generation du dataset complet
# ═══════════════════════════════════════════════════════════════════════════

def generate_dataset(n_samples: int,
                     n_obs_range: Tuple[int, int] = (300, 6000),
                     phase: int    = 2,
                     seed: int     = 0,
                     verbose: bool = True,
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genere n_samples paires (H_norm, B_norm) pour l'entrainement.

    Parameters
    ----------
    n_samples    : nombre de paires a generer
    n_obs_range  : (n_min, n_max) observations simulees par paire
    phase        : 1 = peu bruyte (n_obs >= 2000), 2 = bruit realiste
    seed         : graine pour la reproductibilite
    verbose      : afficher la progression

    Returns
    -------
    X : (n_samples, M_GRID)  float32  — H normalises (entrees)
    Y : (n_samples, M_GRID)  float32  — B normalises (cibles)

    Exemple d'utilisation :
    >>> X_train, Y_train = generate_dataset(20000, seed=1)
    >>> X_val,   Y_val   = generate_dataset(2000,  seed=99)
    """
    rng   = np.random.default_rng(seed)
    X     = np.zeros((n_samples, M_GRID), dtype=np.float32)
    Y     = np.zeros((n_samples, M_GRID), dtype=np.float32)

    n_min = max(n_obs_range[0], 2000) if phase == 1 else n_obs_range[0]
    n_max = n_obs_range[1]

    log_every = max(1, n_samples // 10)
    for i in range(n_samples):
        if verbose and (i + 1) % log_every == 0:
            print(f'    {i+1:7d} / {n_samples}')

        B       = sample_B(rng)
        n_obs   = int(rng.integers(n_min, n_max + 1))
        T       = simulate_T(B, n_obs, rng)
        H_obs   = nelson_aalen_on_grid(T)
        H_n, B_n, _ = normalize_pair(H_obs, B)
        X[i]    = H_n.astype(np.float32)
        Y[i]    = B_n.astype(np.float32)

    return X, Y
