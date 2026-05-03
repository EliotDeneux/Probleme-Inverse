#!/usr/bin/env python3
"""
sweep_gamma.py — Balayage des paramètres (k, σ) du gamma généralisé.

Pour chaque combinaison (k, σ) × modèle, on simule n cellules,
on estime B par Tikhonov p=0 avec sélection GCV, et on calcule
l'erreur L² relative ‖B̂ − B‖ / ‖B‖.

À la fin, le script affiche le meilleur (k, σ) par modèle et
met à jour automatiquement simulate_division.py, src_direct_problem.py
et run_all.py avec ces paramètres optimaux.

Usage
─────
    python sweep_gamma.py                  # balayage complet (tous modèles)
    python sweep_gamma.py --model age      # seulement âge
    python sweep_gamma.py --n 5000         # sous-échantillonner à 5000
    python sweep_gamma.py --no-update      # afficher sans modifier les fichiers

Grille de balayage par défaut
──────────────────────────────
    k     : [1.5, 2.0, 2.5, 3.0, 4.0]
    σ âge : [40, 60, 80]   min
    σ incr: [0.5, 1.0, 1.5] µm
    σ size: [1.0, 1.5, 2.0] µm
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── Imports projet ────────────────────────────────────────────────────────────
from simulate_division import (
    _make_generalized_gamma, _make_size_generalized_gamma,
    simulate_age_model, simulate_incr_model, simulate_size_model,
    N_CELLS,
)
from src_direct_problem import DirectProblemSolver, make_generalized_gamma
from src import NelsonAalanEstimator, TikhonovRegularizer, GeneralizedCrossValidation


# ═══════════════════════════════════════════════════════════════════════════
# Grilles de balayage
# ═══════════════════════════════════════════════════════════════════════════

K_VALUES = [1.5, 2.0, 2.5, 3.0, 4.0]

SIGMA_GRID = {
    'age'      : [40.0, 60.0, 80.0],   # minutes
    'increment': [0.5,  1.0,  1.5],    # µm
    'size'     : [1.0,  1.5,  2.0],    # µm
}


# ═══════════════════════════════════════════════════════════════════════════
# Estimation de l'erreur L² pour un (k, σ, modèle) donné
# ═══════════════════════════════════════════════════════════════════════════

def estimate_l2(k: float, sigma: float, model: str,
                n: int = 10_000, seed: int = 42) -> float:
    """
    Simule n cellules avec B = gamma_généralisé(k, σ),
    estime B̂ par Tikhonov p=0 + GCV,
    retourne l'erreur L² relative ‖B̂ − B‖ / ‖B‖.
    """
    rng = np.random.default_rng(seed)

    # ── Simulation ──────────────────────────────────────────────────────────
    if model == 'age':
        rate = _make_generalized_gamma(k, sigma, unit_label="min")
        df   = simulate_age_model(rate, n, rng)
        T    = df['division_age'].to_numpy(float)
        X_ub = None
    elif model == 'increment':
        rate = _make_generalized_gamma(k, sigma, unit_label="um")
        df   = simulate_incr_model(rate, n, rng)
        T    = df['increment'].to_numpy(float)
        X_ub = None
    elif model == 'size':
        rate = _make_size_generalized_gamma(k, sigma)
        df   = simulate_size_model(rate, n, rng)
        T    = df['division_size'].to_numpy(float)
        X_ub = df['birth_size'].to_numpy(float)
    else:
        raise ValueError(f"Modèle inconnu : {model}")

    # ── Pipeline inverse ─────────────────────────────────────────────────────
    grid  = DirectProblemSolver.grid_from_data(T, n_points=200, quantile_max=0.98)
    A     = DirectProblemSolver(grid).integration_matrix
    eps   = 1.0 / np.sqrt(n)

    na    = NelsonAalanEstimator().fit(T, entry_times=X_ub)
    H_eps = na.smooth(grid, sigma_grid=2.0)

    tikh  = TikhonovRegularizer(A, p=0).fit(H_eps)
    ag    = np.logspace(-6, 0, 60)
    alpha = GeneralizedCrossValidation().select(tikh, ag)
    B_hat = tikh.predict(alpha)

    # ── B vrai sur la grille ─────────────────────────────────────────────────
    spec  = make_generalized_gamma(k, sigma, domain=model)
    B_true = spec.func(grid)

    # ── Erreur L² relative ───────────────────────────────────────────────────
    mask = np.isfinite(B_hat) & np.isfinite(B_true) & (B_true > 0)
    num  = np.trapezoid((B_hat[mask] - B_true[mask]) ** 2, grid[mask])
    den  = np.trapezoid(B_true[mask] ** 2, grid[mask])
    return float(np.sqrt(num / max(den, 1e-15)))


# ═══════════════════════════════════════════════════════════════════════════
# Balayage complet
# ═══════════════════════════════════════════════════════════════════════════

def run_sweep(models: list, n: int = 10_000, seed: int = 42) -> dict:
    """
    Balaye toutes les combinaisons (k, σ) pour chaque modèle.
    Retourne un dict {model: {'results': [...], 'best': {...}}}.
    """
    all_results = {}

    for model in models:
        print(f"\n{'═'*58}")
        print(f"  Modèle : {model.upper()}")
        print(f"{'─'*58}")
        print(f"  {'k':>5}  {'σ':>6}  {'L² err':>10}  {'temps':>8}")
        print(f"{'─'*58}")

        rows = []
        sigma_vals = SIGMA_GRID[model]

        for k, sigma in product(K_VALUES, sigma_vals):
            t0  = time.perf_counter()
            err = estimate_l2(k, sigma, model, n=n, seed=seed)
            dt  = time.perf_counter() - t0
            rows.append({'k': k, 'sigma': sigma, 'l2_err': err})
            print(f"  {k:>5.1f}  {sigma:>6.2f}  {err:>10.4f}  {dt:>7.1f}s")

        best = min(rows, key=lambda r: r['l2_err'])
        print(f"{'─'*58}")
        print(f"  ✓ Meilleur : k={best['k']},  σ={best['sigma']},  L²={best['l2_err']:.4f}")

        all_results[model] = {'results': rows, 'best': best}

    return all_results


# ═══════════════════════════════════════════════════════════════════════════
# Mise à jour automatique des fichiers du projet
# ═══════════════════════════════════════════════════════════════════════════

def update_project_files(sweep_results: dict) -> None:
    """
    Met à jour simulate_division.py, src_direct_problem.py et run_all.py
    avec les paramètres optimaux trouvés par le balayage.
    """
    print(f"\n{'═'*58}")
    print("  Mise à jour des fichiers projet")
    print(f"{'─'*58}")

    # ── simulate_division.py ─────────────────────────────────────────────────
    sim_path = ROOT / 'simulate_division.py'
    content  = sim_path.read_text(encoding='utf-8')

    for model, res in sweep_results.items():
        best  = res['best']
        k, s  = best['k'], best['sigma']
        label = "min" if model == "age" else "um"
        name  = f"generalized_gamma_k{k}_s{s}".replace('.', 'p')

        # Remplace l'ancienne instance gamma généralisé dans la liste du modèle
        if model in ('age', 'increment'):
            old_pattern = f"_make_generalized_gamma(k="
            # On cherche et remplace la ligne correspondante dans AGE_RATES/INCR_RATES
            rates_var = 'AGE_RATES' if model == 'age' else 'INCR_RATES'
            # Remplacement ligne par ligne
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if old_pattern in line and rates_var in '\n'.join(lines[max(0,i-10):i]):
                    lines[i] = (
                        f"    _make_generalized_gamma(k={k}, sigma={s}, "
                        f"unit_label=\"{label}\"),  "
                        f"# meilleur balayage : L²={best['l2_err']:.4f}"
                    )
                    break
            content = '\n'.join(lines)
        elif model == 'size':
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if '_make_size_generalized_gamma(k=' in line:
                    lines[i] = (
                        f"size_gg = _make_size_generalized_gamma(k={k}, sigma={s})  "
                        f"# meilleur balayage : L²={best['l2_err']:.4f}"
                    )
                    break
            content = '\n'.join(lines)

    sim_path.write_text(content, encoding='utf-8')
    print(f"  ✓ simulate_division.py mis à jour")

    # ── src_direct_problem.py ────────────────────────────────────────────────
    spec_path = ROOT / 'src_direct_problem.py'
    content   = spec_path.read_text(encoding='utf-8')
    lines     = content.split('\n')

    for model, res in sweep_results.items():
        best  = res['best']
        k, s  = best['k'], best['sigma']
        old_key = f"('{model}', 'generalized_gamma_"
        new_name = f"generalized_gamma_k{k}_s{s}".replace('.', 'p')
        new_line = (
            f"    ('{model}', '{new_name}') : "
            f"make_generalized_gamma(k={k}, sigma={s}, domain='{model}'),"
            f"  # L²={best['l2_err']:.4f}"
        )
        for i, line in enumerate(lines):
            if old_key in line:
                lines[i] = new_line
                break

    spec_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f"  ✓ src_direct_problem.py mis à jour")

    # ── run_all.py ───────────────────────────────────────────────────────────
    run_path = ROOT / 'run_all.py'
    content  = run_path.read_text(encoding='utf-8')
    lines    = content.split('\n')

    for i, line in enumerate(lines):
        if "'age'" in line and 'generalized_gamma' in line and 'MODEL_RATES' not in line:
            k  = sweep_results['age']['best']['k']
            s  = sweep_results['age']['best']['sigma']
            nm = f"generalized_gamma_k{k}_s{s}".replace('.', 'p')
            lines[i] = f"    'age'      : ['constant', 'weibull2', 'step', '{nm}'],"
        elif "'size'" in line and 'generalized_gamma' in line:
            k  = sweep_results['size']['best']['k']
            s  = sweep_results['size']['best']['sigma']
            nm = f"generalized_gamma_k{k}_s{s}".replace('.', 'p')
            lines[i] = f"    'size'     : ['constant', 'linear',   'power', '{nm}'],"
        elif "'increment'" in line and 'generalized_gamma' in line:
            k  = sweep_results['increment']['best']['k']
            s  = sweep_results['increment']['best']['sigma']
            nm = f"generalized_gamma_k{k}_s{s}".replace('.', 'p')
            lines[i] = f"    'increment': ['constant', 'weibull2', 'step',  '{nm}'],"

    run_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f"  ✓ run_all.py mis à jour")


# ═══════════════════════════════════════════════════════════════════════════
# Point d'entrée
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Balayage (k, σ) du gamma généralisé"
    )
    p.add_argument('--model', nargs='+',
                   choices=['age', 'increment', 'size'],
                   default=['age', 'increment', 'size'])
    p.add_argument('--n',     type=int, default=10_000,
                   help="Nombre de cellules par simulation (défaut: 10000)")
    p.add_argument('--seed',  type=int, default=42)
    p.add_argument('--no-update', action='store_true',
                   help="Ne pas modifier les fichiers projet")
    return p.parse_args()


def main():
    args = parse_args()
    t0   = time.perf_counter()

    print(f"\n{'═'*58}")
    print(f"  Balayage gamma généralisé — (k, σ)")
    print(f"  k ∈ {K_VALUES}")
    print(f"  Modèles : {args.model}")
    print(f"  n = {args.n:,}  |  seed = {args.seed}")
    print(f"{'═'*58}")

    results = run_sweep(args.model, n=args.n, seed=args.seed)

    # ── Résumé final ─────────────────────────────────────────────────────────
    print(f"\n{'═'*58}")
    print("  RÉSUMÉ DES MEILLEURS PARAMÈTRES")
    print(f"{'─'*58}")
    for model, res in results.items():
        b = res['best']
        print(f"  {model:12s}  k={b['k']:.1f}  σ={b['sigma']:.2f}  L²={b['l2_err']:.4f}")
    print(f"{'═'*58}")

    # ── Sauvegarde JSON ──────────────────────────────────────────────────────
    out = ROOT / 'results' / 'sweep_gamma.json'
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Résultats sauvegardés → {out}")

    # ── Mise à jour des fichiers ─────────────────────────────────────────────
    if not args.no_update:
        update_project_files(results)
    else:
        print("  (--no-update : fichiers non modifiés)")

    print(f"\n  Temps total : {time.perf_counter() - t0:.1f} s\n")


if __name__ == '__main__':
    main()
