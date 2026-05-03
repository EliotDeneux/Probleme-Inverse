#!/usr/bin/env python3
"""
test_generalized_gamma.py — Test du problème direct pour B ~ gamma généralisé.

La loi gamma généralisée (= Weibull dans ce contexte) a pour taux de hasard :

    B(t) = (k / σ) · (t / σ)^{k−1}

ce qui donne le hasard cumulé :

    H(t) = (t / σ)^k

et donc la densité de division :

    f(t) = B(t) · exp(−H(t)) = (k/σ)(t/σ)^{k−1} · exp(−(t/σ)^k)

C'est exactement une loi de Weibull(k, σ).
  • k = 1  →  taux constant  (exponentielle, cas déjà présent)
  • k = 2  →  Weibull2       (cas déjà présent avec σ)
  • k > 1  →  taux croissant  (cellule de plus en plus "prête" à se diviser)
  • k < 1  →  taux décroissant

Usage :
    python test_generalized_gamma.py              # paramètres par défaut
    python test_generalized_gamma.py --k 3 --sigma 50 --n 20000
    python test_generalized_gamma.py --model age --k 1.5 --sigma 70
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest, weibull_min

# ── Résolution du chemin du projet ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


# ═══════════════════════════════════════════════════════════════════════════
# 1. Taux de hasard gamma généralisé (Weibull)
# ═══════════════════════════════════════════════════════════════════════════

def make_generalized_gamma_rate(k: float, sigma: float, unit_label: str = ""):
    """
    Construit un objet Rate pour B(t) = (k/σ)(t/σ)^{k−1}.

    Paramètres
    ----------
    k     : paramètre de forme (shape). k=1 → exp, k=2 → Weibull2 déjà connu.
    sigma : paramètre d'échelle (scale).

    Simulation : T ~ Weibull(k, σ), soit T = σ · (-ln U)^{1/k}, U ~ Unif(0,1).
    Équivalent numpy : rng.weibull(k, n) * sigma.

    Vérifications mathématiques :
      (C1) ∫₀^∞ B(s) ds = ∫₀^∞ (k/σ)(s/σ)^{k−1} ds = ∞  ✓  (k > 0)
      (C2) E[T] = σ · Γ(1 + 1/k) < ∞                       ✓
    """
    import math
    E_t = sigma * math.gamma(1 + 1 / k)

    return dict(
        name=f"generalized_gamma_k{k}_s{sigma}",
        description=f"B(t) = (k/σ)(t/σ)^{{k−1}},  k={k}, σ={sigma}  [Weibull({k}, {sigma})]",
        params={"k": k, "sigma": sigma, f"E_t_{unit_label}": round(E_t, 4)},
        B=lambda t, _k=k, _s=sigma: np.where(
            np.asarray(t, float) > 0,
            (_k / _s) * (np.clip(np.asarray(t, float), 1e-12, None) / _s) ** (_k - 1),
            0.0 if _k >= 1 else np.inf,
        ),
        sample_t=lambda n, rng, _k=k, _s=sigma: rng.weibull(_k, n) * _s,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 2. Problème direct : calcul des quantités théoriques
# ═══════════════════════════════════════════════════════════════════════════

def compute_theory(B_func, grid: np.ndarray) -> dict:
    """
    Calcule H, S, f, F à partir de B sur la grille donnée.

    Relations :
        H(t) = ∫₀ᵗ B(s) ds          (trapèzes)
        S(t) = exp(−H(t))
        f(t) = B(t) · S(t)
        F(t) = 1 − S(t)
    """
    from scipy.integrate import cumulative_trapezoid

    B = np.asarray([B_func(np.array([t]))[0] for t in grid])
    H = np.zeros_like(B)
    H[1:] = cumulative_trapezoid(B, grid)
    S = np.exp(-H)
    return {"grid": grid, "B": B, "H": H, "S": S, "f": B * S, "F": 1.0 - S}


# ═══════════════════════════════════════════════════════════════════════════
# 3. Test complet du problème direct
# ═══════════════════════════════════════════════════════════════════════════

def verify_direct_generalized_gamma(
    k: float,
    sigma: float,
    n: int = 10_000,
    model: str = "age",
    seed: int = 42,
    show_plot: bool = True,
) -> dict:
    """
    Simule n cellules avec B(t) = (k/σ)(t/σ)^{k−1} et vérifie la cohérence
    avec la théorie via :
      - Comparaison histogramme empirique vs f(t) théorique
      - Comparaison survie empirique vs S(t) théorique
      - Statistique de Kolmogorov-Smirnov (KS)
      - Test KS de scipy (p-value)

    Retourne
    --------
    dict avec clés : ks_stat, ks_pvalue, norm_check, mean_empirique, mean_theorique
    """
    rng = np.random.default_rng(seed)
    rate = make_generalized_gamma_rate(k, sigma, unit_label={"age": "min",
                                                              "increment": "um",
                                                              "size": "um"}.get(model, ""))

    # ── Simulation ──────────────────────────────────────────────────────────
    T = rate["sample_t"](n, rng)

    # ── Grille de travail ───────────────────────────────────────────────────
    t_max = float(np.quantile(T, 0.99))
    grid  = np.linspace(0.0, t_max, 300)

    # ── Quantités théoriques ────────────────────────────────────────────────
    theory = compute_theory(rate["B"], grid)

    # Vérification de normalisation : ∫ f(t) dt doit être ≈ 1
    norm_check = float(np.trapezoid(theory["f"], grid))

    # ── Statistique KS manuelle ─────────────────────────────────────────────
    F_emp  = np.array([np.mean(T <= t) for t in grid])
    ks_stat = float(np.max(np.abs(F_emp - theory["F"])))

    # ── Test KS de scipy (Weibull = weibull_min avec loc=0) ─────────────────
    # weibull_min(c=k, scale=sigma) correspond exactement à notre loi
    ks_result = kstest(T, weibull_min(c=k, scale=sigma).cdf)

    # ── Moyennes ────────────────────────────────────────────────────────────
    import math
    mean_theorique = sigma * math.gamma(1 + 1 / k)
    mean_empirique = float(np.mean(T))

    # ── Affichage console ────────────────────────────────────────────────────
    print(f"\n{'═'*58}")
    print(f"  Test problème direct — Gamma généralisé (Weibull)")
    print(f"  Modèle : {model}  |  k={k}  σ={sigma}  n={n:,}")
    if k < 1:
        print(f"  ⚠ k < 1 : B(t) → ∞ en t=0. KS manuel faussé, utiliser KS scipy.")
    print(f"{'─'*58}")
    print(f"  ∫ f(t) dt (doit être ≈ 1)  : {norm_check:.5f}")
    print(f"  E[T] théorique             : {mean_theorique:.4f}")
    print(f"  E[T] empirique             : {mean_empirique:.4f}")
    print(f"  KS stat (manuel)           : {ks_stat:.5f}  (attendu ≈ 1/√n = {1/n**0.5:.4f})")
    print(f"  KS stat (scipy)            : {ks_result.statistic:.5f}")
    print(f"  KS p-value                 : {ks_result.pvalue:.4f}  ", end="")
    if ks_result.pvalue > 0.05:
        print("✓  (p > 0.05 : simulateur cohérent avec la théorie)")
    else:
        print("✗  (p ≤ 0.05 : écart significatif détecté !)")
    print(f"{'═'*58}\n")

    # ── Figure 2×2 ──────────────────────────────────────────────────────────
    if show_plot:
        xlabel = {"age": "Âge à la division [min]",
                  "increment": "Incrément de taille [µm]",
                  "size": "Taille à la division [µm]"}.get(model, "t")

        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        fig.suptitle(
            f"Problème direct — Gamma généralisé (Weibull)\n"
            f"Modèle {model}  |  k={k},  σ={sigma},  n={n:,}",
            fontsize=12,
        )

        # B(t) théorique
        ax = axes[0, 0]
        ax.plot(grid, theory["B"], "k-", lw=2.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("B(t)")
        ax.set_title(f"Taux de division B(t) = (k/σ)(t/σ)^{{k−1}}")
        ax.axhline(0, color="gray", lw=0.5)

        # H(t) = opérateur direct ΨB
        ax = axes[0, 1]
        ax.plot(grid, theory["H"], color="steelblue", lw=2.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("H(t)")
        ax.set_title("Hasard cumulé H(t) = (t/σ)^k")

        # f(t) + histogramme empirique
        ax = axes[1, 0]
        T_clip = T[(T >= grid[0]) & (T <= grid[-1])]
        ax.hist(T_clip, bins=50, density=True, alpha=0.35,
                color="steelblue", label="Données simulées")
        ax.plot(grid, theory["f"], "k-", lw=2.5,
                label="f(t) = B(t)·S(t) théorique")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Densité")
        ax.set_title(
            f"Densité f  —  KS={ks_result.statistic:.4f}  "
            f"(p={ks_result.pvalue:.3f})"
        )
        ax.legend(fontsize=9)

        # Survie : théorique vs empirique
        ax = axes[1, 1]
        ax.plot(grid, theory["S"], "k-", lw=2.5, label="S(t) théorique")
        ax.plot(grid, 1.0 - F_emp, "r--", lw=1.8, label="1 − Fₙ empirique")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("S(t) = P(T > t)")
        ax.set_title("Fonction de survie S(t)")
        ax.legend(fontsize=9)

        plt.tight_layout()
        out_path = Path("figures") / f"direct_generalized_gamma_k{k}_s{sigma}.png"
        out_path.parent.mkdir(exist_ok=True)
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        print(f"  Figure sauvegardée → {out_path}")
        plt.show()

    return {
        "ks_stat"        : ks_stat,
        "ks_pvalue"      : ks_result.pvalue,
        "ks_scipy"       : ks_result.statistic,
        "norm_check"     : norm_check,
        "mean_empirique" : mean_empirique,
        "mean_theorique" : mean_theorique,
        "k"              : k,
        "sigma"          : sigma,
        "n"              : n,
        "model"          : model,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. Balayage de plusieurs valeurs de k (optionnel)
# ═══════════════════════════════════════════════════════════════════════════

def sweep_k_values(
    k_values: list,
    sigma: float = 60.0,
    n: int = 10_000,
    model: str = "age",
    seed: int = 42,
) -> None:
    """
    Teste plusieurs valeurs de k et affiche un tableau comparatif des KS.
    Utile pour voir comment le taux change de forme selon k.
    """
    print(f"\n{'═'*58}")
    print(f"  Balayage de k  —  σ={sigma}  n={n:,}  modèle={model}")
    print(f"{'─'*58}")
    print(f"  {'k':>6}  {'KS stat':>10}  {'p-value':>10}  {'E[T] emp':>10}  {'E[T] th':>10}")
    print(f"{'─'*58}")

    for k in k_values:
        res = verify_direct_generalized_gamma(
            k=k, sigma=sigma, n=n, model=model, seed=seed, show_plot=False
        )
        ok = "✓" if res["ks_pvalue"] > 0.05 else "✗"
        print(
            f"  {k:>6.2f}  {res['ks_stat']:>10.5f}  "
            f"{res['ks_pvalue']:>10.4f}  "
            f"{res['mean_empirique']:>10.3f}  "
            f"{res['mean_theorique']:>10.3f}  {ok}"
        )

    print(f"{'═'*58}\n")


# ═══════════════════════════════════════════════════════════════════════════
# 5. Point d'entrée
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Test du problème direct — taux gamma généralisé (Weibull)"
    )
    p.add_argument("--k",     type=float, default=2.5,
                   help="Paramètre de forme k (défaut: 2.5)")
    p.add_argument("--sigma", type=float, default=60.0,
                   help="Paramètre d'échelle σ (défaut: 60.0)")
    p.add_argument("--n",     type=int,   default=10_000,
                   help="Nombre de cellules simulées (défaut: 10000)")
    p.add_argument("--model", choices=["age", "increment", "size"],
                   default="age",
                   help="Modèle de division (défaut: age)")
    p.add_argument("--sweep", action="store_true",
                   help="Balayer plusieurs valeurs de k (0.5, 1, 1.5, 2, 3, 5)")
    p.add_argument("--no-plot", action="store_true",
                   help="Ne pas afficher la figure")
    return p.parse_args()


def main():
    args = parse_args()

    if args.sweep:
        sweep_k_values(
            k_values=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0],
            sigma=args.sigma,
            n=args.n,
            model=args.model,
        )
    else:
        verify_direct_generalized_gamma(
            k=args.k,
            sigma=args.sigma,
            n=args.n,
            model=args.model,
            show_plot=not args.no_plot,
        )


if __name__ == "__main__":
    main()
