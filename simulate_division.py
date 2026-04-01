#!/usr/bin/env python3
"""
Simulation de populations de cellules en division – trois modèles.
═══════════════════════════════════════════════════════════════════

Modèles :
  • Âge       – taux B(a)  dépend de l'âge à la division
  • Taille    – taux B(x)  dépend de la taille à la division
  • Incrément – taux B(z)  dépend de l'incrément de taille Xud − Xub

Croissance exponentielle : Xud = Xub · exp(K · Aud)

Conditions sur B pour tout modèle :
  (C1) ∫₀^∞ B(s) ds = ∞        [division certaine]
  (C2) ∫₀^∞ exp(−∫₀^a B(s)ds) da < ∞  [espérance finie]

Sorties : /data/{model}/{rate}.parquet  +  /data/metadata.json

Format Parquet (snappy) : compromis optimal entre taille disque et
vitesse de lecture (pandas, polars, pyarrow, R/arrow, Julia…).
"""

from __future__ import annotations
import json, time, sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════════
# Paramètres globaux
# ═══════════════════════════════════════════════════════════════════════════════
SEED     = 42
N_CELLS  = 10_000
K        = np.log(2) / 70   # taux de croissance [min⁻¹]  (doublement ≈ 70 min, E. coli)

# Taille à la naissance : log-normale  (médiane = 1 µm, CV ≈ 10 %)
BIRTH_MU  = 0.0
BIRTH_SIG = 0.1

DATA_DIR = Path("data") # ou Path("./data")


def birth_sizes(n: int, rng: np.random.Generator) -> np.ndarray:
    """Tire n tailles de naissance i.i.d. log-normales."""
    return rng.lognormal(BIRTH_MU, BIRTH_SIG, n)


# ═══════════════════════════════════════════════════════════════════════════════
# Structure générique d'un taux de division
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class Rate:
    name: str
    description: str
    params: dict                                                  # paramètres numériques
    B: Callable[[np.ndarray], np.ndarray]                        # fonction de hasard
    # sample_t : échantillonne depuis f(t) = B(t) exp(−∫₀ᵗ B)
    #   utilisé pour les modèles âge et incrément
    sample_t: Callable[[int, np.random.Generator], np.ndarray] | None = None
    # sample_x : échantillonne depuis f(x ; xb) = B(x) exp(−∫_{xb}^x B)
    #   utilisé pour le modèle taille ; xb est un tableau (une valeur par cellule)
    sample_x: Callable[[np.ndarray, np.random.Generator], np.ndarray] | None = None


# ═══════════════════════════════════════════════════════════════════════════════
# ── Taux pour les modèles Âge et Incrément ──────────────────────────────────
#
# Les deux modèles ont la même structure mathématique :
#   f(t) = B(t) exp(−∫₀ᵗ B(s) ds)
# Seule l'interprétation de t (âge ou incrément) diffère.
#
# Conditions :
#   (C1) ∫₀^∞ B(s) ds = ∞     ← vérifiée pour chaque choix ci-dessous
#   (C2) ∫₀^∞ exp(−∫₀^a B) da < ∞  ← idem
# ═══════════════════════════════════════════════════════════════════════════════

# ── 1. Taux constant : B(t) = λ ─────────────────────────────────────────────
#
# H(t) = λt  →  F(t) = 1 − e^{−λt}  →  t ~ Exp(λ)
#
# Vérifications :
#   (C1) λ > 0  →  ∫₀^∞ λ ds = ∞  ✓
#   (C2) ∫₀^∞ e^{−λa} da = 1/λ  < ∞  ✓

def _make_constant(lam: float, unit_label: str) -> Rate:
    return Rate(
        name="constant",
        description=f"B(t) = λ  [taux constant, t ~ Exp(λ)]",
        params={"lambda": lam, f"E_t_{unit_label}": round(1 / lam, 4)},
        B=lambda t, l=lam: np.full_like(np.asarray(t, float), l),
        sample_t=lambda n, rng, l=lam: rng.exponential(1 / l, n),
    )


# ── 2. Weibull k=2 : B(t) = 2t/σ² ──────────────────────────────────────────
#
# H(t) = t²/σ²  →  F(t) = 1 − exp(−t²/σ²)  →  t ~ Weibull(2, σ)
#
# Vérifications :
#   (C1) ∫₀^∞ (2s/σ²) ds = ∞  ✓
#   (C2) ∫₀^∞ exp(−a²/σ²) da = σ√π/2 < ∞  ✓
#
# Ce choix capture le fait que la probabilité de division augmente avec
# le temps/l'incrément : le taux de hasard est croissant.

def _make_weibull2(sigma: float, unit_label: str) -> Rate:
    E_t = sigma * np.sqrt(np.pi) / 2   # = σ Γ(3/2)
    return Rate(
        name="weibull2",
        description=f"B(t) = 2t/σ²  [Weibull k=2, taux hasard linéaire]",
        params={"sigma": sigma, f"E_t_{unit_label}": round(E_t, 4)},
        B=lambda t, s=sigma: 2 * np.asarray(t, float) / s**2,
        sample_t=lambda n, rng, s=sigma: rng.weibull(2, n) * s,
    )


# ── 3. Seuil (delayed exponential) : B(t) = λ · 𝟏(t ≥ t₀) ─────────────────
#
# H(t) = λ (t − t₀)₊  →  t ~ t₀ + Exp(λ)
# La cellule ne peut pas se diviser avant d'avoir atteint t₀.
#
# Vérifications :
#   (C1) ∫₀^∞ B(s) ds = ∞  ✓
#   (C2) ∫₀^{t₀} 1 da + ∫_{t₀}^∞ e^{−λ(a−t₀)} da = t₀ + 1/λ < ∞  ✓

def _make_step(lam: float, t0: float, unit_label: str) -> Rate:
    return Rate(
        name="step",
        description=f"B(t) = λ · 𝟏(t ≥ t₀)  [seuil minimal avant division]",
        params={"lambda": lam, "t0": t0, f"E_t_{unit_label}": round(t0 + 1 / lam, 4)},
        B=lambda t, l=lam, t_=t0: np.where(np.asarray(t, float) >= t_, l, 0.0),
        sample_t=lambda n, rng, l=lam, t_=t0: t_ + rng.exponential(1 / l, n),
    )


# ── Instanciation pour le modèle Âge ────────────────────────────────────────
# Paramètres calibrés : E. coli, temps en minutes, tailles en µm
AGE_RATES = [
    _make_constant(lam=0.02,  unit_label="min"),          # E[A] = 50 min
    _make_weibull2(sigma=60.0, unit_label="min"),         # E[A] ≈ 53 min
    _make_step(lam=0.05, t0=20.0, unit_label="min"),     # E[A] = 40 min
]

# ── Instanciation pour le modèle Incrément ───────────────────────────────────
# Incrément Z = Xud − Xub ~ 0.4–0.7 µm (cohérent avec croissance ×1.5 par division)
INCR_RATES = [
    _make_constant(lam=2.0,   unit_label="um"),           # E[Z] = 0.50 µm
    _make_weibull2(sigma=0.7,  unit_label="um"),          # E[Z] ≈ 0.62 µm
    _make_step(lam=4.0, t0=0.2, unit_label="um"),        # E[Z] = 0.45 µm
]


# ═══════════════════════════════════════════════════════════════════════════════
# ── Taux pour le modèle Taille ───────────────────────────────────────────────
#
# f(x ; xb) = B(x) exp(−∫_{xb}^x B(s) ds)   pour x > xb
#
# Conditions (pour tout xb > 0) :
#   (C1) ∫_{xb}^∞ B(x) dx = ∞
#   (C2) ∫_{xb}^∞ exp(−∫_{xb}^x B(s) ds) dx < ∞
#
# L'inversion analytique de H(x ; xb) = ∫_{xb}^x B(s) ds est possible
# pour les trois familles ci-dessous : H(x ; xb) = E ~ Exp(1)  →  x = H⁻¹(E ; xb)
# ═══════════════════════════════════════════════════════════════════════════════

# ── 1. Taux constant : B(x) = β ─────────────────────────────────────────────
#
# H(x ; xb) = β (x − xb)  →  Xud = xb + Exp(β) ← incrément exponentiel
#
# Vérifications :
#   (C1) ∫_{xb}^∞ β dx = ∞  ✓
#   (C2) ∫_{xb}^∞ e^{−β(x−xb)} dx = 1/β < ∞  ✓

_beta1 = 2.0   # µm⁻¹  →  E[incrément] = 0.5 µm
size_constant = Rate(
    name="constant",
    description="B(x) = β  [taux constant, incrément ~ Exp(β)]",
    params={"beta": _beta1, "E_increment_um": round(1 / _beta1, 4)},
    B=lambda x: np.full_like(np.asarray(x, float), _beta1),
    sample_x=lambda xb, rng: xb + rng.exponential(1 / _beta1, len(xb)),
)

# ── 2. Taux linéaire : B(x) = α x ────────────────────────────────────────────
#
# H(x ; xb) = α(x² − xb²)/2  →  Xud = sqrt(xb² + 2 E / α),  E ~ Exp(1)
# Interprétation : la probabilité de division augmente avec la taille
# (mécanisme de seuil de taille absolu).
#
# Vérifications :
#   (C1) ∫_{xb}^∞ αx dx = ∞  ✓
#   (C2) ∫_{xb}^∞ e^{−α(x²−xb²)/2} dx = e^{αxb²/2} · (√π/2) · erfc(xb√(α/2)) / √(α/2) < ∞  ✓

_alpha2 = 4.0   # µm⁻²
size_linear = Rate(
    name="linear",
    description="B(x) = αx  [taux croissant, seuil de taille marqué]",
    params={"alpha": _alpha2},
    B=lambda x: _alpha2 * np.asarray(x, float),
    sample_x=lambda xb, rng: np.sqrt(xb**2 + 2 * rng.exponential(1, len(xb)) / _alpha2),
)

# ── 3. Taux puissance : B(x) = β x^γ ─────────────────────────────────────────
#
# H(x ; xb) = β(x^{γ+1} − xb^{γ+1}) / (γ+1)
#   →  Xud = (xb^{γ+1} + (γ+1) E / β)^{1/(γ+1)},  E ~ Exp(1)
# Intermédiaire entre constant (γ→0) et linéaire (γ=1).
#
# Vérifications (γ > −1, ici γ = 0.5) :
#   (C1) ∫_{xb}^∞ β x^γ dx = ∞  ✓
#   (C2) E[Xud^{γ+1} − xb^{γ+1}] = (γ+1)/β < ∞  ✓

_beta3, _gamma3 = 3.0, 0.5   # B(x) = 3√x
_gp1 = _gamma3 + 1           # exposant  = 1.5
size_power = Rate(
    name="power",
    description="B(x) = β √x  [puissance γ=0.5, intermédiaire constant/linéaire]",
    params={"beta": _beta3, "gamma": _gamma3},
    B=lambda x: _beta3 * np.asarray(x, float) ** _gamma3,
    sample_x=lambda xb, rng: (
        xb**_gp1 + _gp1 * rng.exponential(1, len(xb)) / _beta3
    ) ** (1 / _gp1),
)

SIZE_RATES = [size_constant, size_linear, size_power]


# ═══════════════════════════════════════════════════════════════════════════════
# Fonctions de simulation
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_age_model(rate: Rate, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Modèle en âge.
    Aud est tiré depuis f(a) = B(a) exp(−∫₀^a B(s) ds).
    Taille à la division : Xud = Xub exp(K Aud).
    """
    xb  = birth_sizes(n, rng)
    aud = rate.sample_t(n, rng)
    xd  = xb * np.exp(K * aud)
    return pd.DataFrame({
        "cell_id"      : np.arange(n, dtype=np.int32),
        "birth_size"   : xb.astype(np.float32),
        "division_age" : aud.astype(np.float32),
        "division_size": xd.astype(np.float32),
        "increment"    : (xd - xb).astype(np.float32),
    })


def simulate_size_model(rate: Rate, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Modèle en taille.
    Xud est tiré depuis f(x ; xb) = B(x) exp(−∫_{xb}^x B(s) ds).
    Âge à la division : Aud = log(Xud / Xub) / K.
    """
    xb  = birth_sizes(n, rng)
    xd  = rate.sample_x(xb, rng)
    aud = np.log(xd / xb) / K
    return pd.DataFrame({
        "cell_id"      : np.arange(n, dtype=np.int32),
        "birth_size"   : xb.astype(np.float32),
        "division_age" : aud.astype(np.float32),
        "division_size": xd.astype(np.float32),
        "increment"    : (xd - xb).astype(np.float32),
    })


def simulate_incr_model(rate: Rate, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Modèle en incrément.
    Z = Xud − Xub est tiré depuis f(z) = B(z) exp(−∫₀^z B(s) ds).
    Âge à la division : Aud = log(1 + Z/Xub) / K.
    """
    xb  = birth_sizes(n, rng)
    z   = rate.sample_t(n, rng)
    xd  = xb + z
    aud = np.log(xd / xb) / K
    return pd.DataFrame({
        "cell_id"      : np.arange(n, dtype=np.int32),
        "birth_size"   : xb.astype(np.float32),
        "division_age" : aud.astype(np.float32),
        "division_size": xd.astype(np.float32),
        "increment"    : z.astype(np.float32),
    })


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline principal
# ═══════════════════════════════════════════════════════════════════════════════

def df_to_npz(df: pd.DataFrame, path: Path) -> None:
    """
    Sauvegarde un DataFrame comme archive NumPy compressée (.npz).
    Chaque colonne devient un tableau ; float32 et int32 pour minimiser la taille.
    Lecture : data = np.load(path) ; data['birth_size'], data['division_age'], …
    """
    np.savez_compressed(
        path,
        cell_id       = df["cell_id"].to_numpy(np.int32),
        birth_size    = df["birth_size"].to_numpy(np.float32),
        division_age  = df["division_age"].to_numpy(np.float32),
        division_size = df["division_size"].to_numpy(np.float32),
        increment     = df["increment"].to_numpy(np.float32),
    )


def insert_into_sqlite(df: pd.DataFrame, con: sqlite3.Connection,
                       model: str, rate: str) -> None:
    """
    Insère les données dans la table 'cells' de la base SQLite globale.
    Les colonnes model et rate permettent les requêtes croisées.
    """
    rows = [
        (int(r.cell_id), model, rate,
         float(r.birth_size), float(r.division_age),
         float(r.division_size), float(r.increment))
        for r in df.itertuples(index=False)
    ]
    con.executemany(
        "INSERT INTO cells VALUES (?,?,?,?,?,?,?)", rows
    )
    con.commit()


def init_sqlite(path: Path) -> sqlite3.Connection:
    """Crée la base SQLite avec la table principale et les index utiles."""
    con = sqlite3.connect(path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS cells (
            cell_id       INTEGER,
            model         TEXT,
            rate          TEXT,
            birth_size    REAL,
            division_age  REAL,
            division_size REAL,
            increment     REAL
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_model_rate ON cells (model, rate)")
    con.commit()
    return con


def main() -> None:
    rng = np.random.default_rng(SEED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ── Base SQLite globale ───────────────────────────────────────────────────
    db_path = DATA_DIR / "cells.db"
    if db_path.exists():
        db_path.unlink()   # repart d'une base vierge à chaque lancement
    con = init_sqlite(db_path)

    # ── Métadonnées JSON ──────────────────────────────────────────────────────
    meta: dict = {
        "description": "Simulation de division cellulaire – trois modèles (âge, taille, incrément)",
        "formats": {
            "npz" : "archive NumPy compressée par modèle/taux — np.load(path)",
            "sqlite": "base SQL unifiée — toutes simulations dans cells.db",
        },
        "seed"    : SEED,
        "n_cells" : N_CELLS,
        "K_per_min": K,
        "birth_lognormal": {
            "mu": BIRTH_MU, "sigma": BIRTH_SIG,
            "median_um": 1.0,
            "mean_um": round(np.exp(BIRTH_MU + BIRTH_SIG**2 / 2), 4),
        },
        "columns": {
            "cell_id"      : "identifiant entier de la cellule",
            "birth_size"   : "Xub – taille à la naissance [µm]",
            "division_age" : "Aud – âge à la division [min]",
            "division_size": "Xud – taille à la division [µm]",
            "increment"    : "Zud = Xud − Xub – incrément de taille [µm]",
        },
        "models": {},
    }

    simulations = [
        ("age",       AGE_RATES,  simulate_age_model),
        ("size",      SIZE_RATES, simulate_size_model),
        ("increment", INCR_RATES, simulate_incr_model),
    ]

    total_bytes = 0
    print(f"\n{'═'*60}")
    print(f"  Simulation de division cellulaire — N = {N_CELLS:,} cellules")
    print(f"{'═'*60}\n")

    for model_name, rates, sim_fn in simulations:
        model_dir = DATA_DIR / model_name
        model_dir.mkdir(exist_ok=True)
        meta["models"][model_name] = {}
        print(f"  Modèle : {model_name.upper()}")

        for rate in rates:
            t0  = time.perf_counter()
            df  = sim_fn(rate, N_CELLS, rng)

            # ── .npz par (modèle, taux) ───────────────────────────────────────
            npz_path = model_dir / f"{rate.name}.npz"
            df_to_npz(df, npz_path)

            # ── SQLite global ─────────────────────────────────────────────────
            insert_into_sqlite(df, con, model_name, rate.name)

            elapsed = time.perf_counter() - t0
            size_kb = npz_path.stat().st_size / 1024
            total_bytes += npz_path.stat().st_size

            # Statistiques descriptives arrondies
            stats_raw = df.drop(columns="cell_id").describe().to_dict()
            stats = {col: {k: round(float(v), 4) for k, v in vs.items()}
                     for col, vs in stats_raw.items()}

            meta["models"][model_name][rate.name] = {
                "description": rate.description,
                "params"     : rate.params,
                "file_npz"   : f"{model_name}/{rate.name}.npz",
                "n_rows"     : len(df),
                "columns"    : list(df.columns),
                "stats"      : stats,
            }

            print(f"    [{rate.name:12s}]  npz={size_kb:5.1f} KB   ({elapsed*1000:.0f} ms)")

        print()

    con.close()
    db_kb = db_path.stat().st_size / 1024

    # ── Métadonnées JSON ──────────────────────────────────────────────────────
    meta_path = DATA_DIR / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)

    n_files = sum(len(r) for _, r, _ in simulations)
    print(f"{'─'*60}")
    print(f"  SQLite global  → {db_path}  ({db_kb:.0f} KB)")
    print(f"  Métadonnées    → {meta_path}")
    print(f"  Total .npz     → {total_bytes/1024:.0f} KB  ({n_files} fichiers)")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
