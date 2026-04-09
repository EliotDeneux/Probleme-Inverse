"""
ml_train.py — Boucle d'entrainement du Neural Operator.

PERTE PHYSICS-INFORMED
═══════════════════════
L(theta) = lambda_B * ||B_hat - B||^2         (terme supervise)
         + lambda_H * ||Psi*B_hat - H||^2     (terme physique)
         + lambda_m * Pen_mono                (penalite de monotonicite)

Terme supervise : erreur directe sur B — guide le reseau vers la vraie solution.

Terme physique : force B_hat a etre coherent avec l'operateur Psi,
i.e. que l'integrale de B_hat reproduise H observe.
C'est un terme de regularisation physiquement motive qui reduit les
oscillations sans penaliser la norme de B (contrairement a Tikhonov).
En pratique il ameliore la precision dans les zones peu observees.

Penalite de monotonicite :
    Pen_mono = mean( max(-dB_hat/dt, 0)^2 )
Incorpore le prior biologique que B est generalement croissant
(le risque de division augmente avec l'age / la taille).

CURRICULUM LEARNING
════════════════════
Epochs < warmup_epochs : lambda_H = 1.5 * lambda_H  (physique renforcee)
Epochs >= warmup_epochs : lambda_H nominal           (equilibre supervise/physique)
-> Le reseau apprend d'abord la contrainte physique, puis affine sur B.

AUGMENTATION DE DONNEES (online, par mini-batch)
══════════════════════════════════════════════════
  1. Bruit additif sur H : H += eps * N(0,1),  eps ~ U[0, 0.01]
     -> Robustesse au bruit de mesure
  2. Reechantillonnage d'echelle : H *= r,  r ~ U[0.85, 1.15]
     -> Invariance a l'echelle (B ne change pas)
  3. Flip temporel (proba 0.1) : H(t) -> H(1)-H(1-t),  B(t) -> B(1-t)
     -> Diversite des formes sans simulation supplementaire

EARLY STOPPING
═══════════════
Si la perte de validation ne s'ameliore pas pendant `patience` epochs,
l'entrainement s'arrete et les meilleurs poids sont restaures.
"""

from __future__ import annotations
import sys, time
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ml_core import NeuralOperator, AdamW
from ml_data  import M_GRID, PSI, GRID


# ═══════════════════════════════════════════════════════════════════════════
# Augmentation de donnees (appliquee online par mini-batch)
# ═══════════════════════════════════════════════════════════════════════════

def augment_batch(X: np.ndarray, Y: np.ndarray,
                  rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augmentation legere sur un mini-batch (batch, m).

    Modifie X (H entrees) et Y (B cibles) en place — copies preservees.
    """
    X = X.copy().astype(np.float64)
    Y = Y.copy().astype(np.float64)
    batch = len(X)

    # 1. Bruit gaussien sur H
    eps_noise = rng.uniform(0.0, 0.01, (batch, 1))
    X += eps_noise * rng.standard_normal(X.shape)
    X  = np.clip(X, 0.0, None)

    # 2. Reechantillonnage d'echelle
    X *= rng.uniform(0.85, 1.15, (batch, 1))

    # 3. Flip temporel (p = 0.10)
    flip = rng.random(batch) < 0.10
    if flip.any():
        H_max         = X[flip, -1:]
        X[flip]       = H_max - X[flip, ::-1]
        Y[flip]       = Y[flip, ::-1]
        X[flip]       = np.clip(X[flip], 0.0, None)

    return X.astype(np.float32), Y.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Perte physics-informed
# ═══════════════════════════════════════════════════════════════════════════

def compute_loss(B_hat      : np.ndarray,
                 B_true     : np.ndarray,
                 H_in       : np.ndarray,
                 lambda_B   : float = 1.0,
                 lambda_H   : float = 0.4,
                 lambda_mono: float = 0.05,
                 ) -> Tuple[float, np.ndarray, Dict[str, float]]:
    """
    Calcule la perte composite et son gradient dL/dB_hat.

    Parameters
    ----------
    B_hat, B_true, H_in : (batch, m) en float64

    Returns
    -------
    loss      : valeur scalaire
    grad_Bhat : (batch, m) gradient dL/dB_hat
    parts     : {'B': ..., 'H': ..., 'mono': ...} pour le monitoring
    """
    batch, m = B_hat.shape

    # ── Terme supervise ──────────────────────────────────────────────────
    diff_B = B_hat - B_true
    loss_B = float(np.mean(diff_B ** 2))
    grad_B = (2.0 * lambda_B / (batch * m)) * diff_B

    # ── Terme physique : ||Psi*B_hat - H||^2 ────────────────────────────
    # H_hat[i] = PSI @ B_hat[i]  ->  vectorise via B_hat @ PSI^T
    H_hat  = B_hat @ PSI.T
    diff_H = H_hat - H_in
    loss_H = float(np.mean(diff_H ** 2))
    # dL_H/dB_hat = 2 * (diff_H @ PSI)  (chaine regle)
    grad_H = (2.0 * lambda_H / (batch * m)) * (diff_H @ PSI)

    # ── Penalite de monotonicite ─────────────────────────────────────────
    dB          = np.diff(B_hat, axis=1)         # (batch, m-1)
    neg_dB      = np.minimum(dB, 0.0)
    loss_mono   = float(np.mean(neg_dB ** 2))
    g_neg       = (2.0 * lambda_mono / (batch * (m - 1))) * neg_dB
    grad_mono   = np.zeros_like(B_hat)
    grad_mono[:, :-1] += g_neg
    grad_mono[:,  1:] -= g_neg

    total = lambda_B * loss_B + lambda_H * loss_H + lambda_mono * loss_mono
    grad  = grad_B + grad_H + grad_mono

    return total, grad, {'B': loss_B, 'H': loss_H, 'mono': loss_mono}


# ═══════════════════════════════════════════════════════════════════════════
# Boucle d'entrainement
# ═══════════════════════════════════════════════════════════════════════════

def train(net: NeuralOperator,
          X_train: np.ndarray, Y_train: np.ndarray,
          X_val  : np.ndarray, Y_val  : np.ndarray,
          n_epochs    : int   = 100,
          batch_size  : int   = 256,
          lr_max      : float = 4e-4,
          T_warmup    : int   = 300,
          lambda_H    : float = 0.4,
          lambda_mono : float = 0.05,
          patience    : int   = 15,
          augment     : bool  = True,
          save_path   : Optional[str] = None,
          verbose     : bool  = True,
          ) -> Dict:
    """
    Entraine le NeuralOperator sur les paires (H, B) synthetiques.

    Parameters
    ----------
    net              : NeuralOperator a entrainer (modifie in-place)
    X_train/val      : (n, m) float32 — H normalises
    Y_train/val      : (n, m) float32 — B normalises (cibles)
    n_epochs         : epochs maximales
    batch_size       : taille des mini-batches
    lr_max           : learning rate de crete (apres warmup)
    T_warmup         : steps de warmup lineaire
    lambda_H         : poids du terme physique (recommande : 0.3 a 0.5)
    lambda_mono      : poids de la penalite de monotonicite (0.02 a 0.1)
    patience         : early stopping (epochs sans amelioration)
    augment          : activer l'augmentation de donnees
    save_path        : si fourni, sauvegarde le meilleur modele ici
    verbose          : afficher la progression toutes les 10 epochs

    Returns
    -------
    history : {
        'train_loss', 'val_loss', 'lr',
        'val_B', 'val_H', 'val_mono',   <- decomposition de la perte
        'best_val_loss', 'n_epochs_done', 'total_time_s'
    }

    Exemple d'utilisation typique :
    >>> net = NeuralOperator(m=128, d=256, n_layers=6)
    >>> history = train(net, X_tr, Y_tr, X_v, Y_v, save_path='model.pkl')
    """
    n          = len(X_train)
    n_steps_ep = max(1, n // batch_size)
    T_total    = n_epochs * n_steps_ep
    warmup_eps = max(1, T_warmup // n_steps_ep)

    optim = AdamW(net,
                  lr_max=lr_max, lr_min=lr_max / 100,
                  T_warmup=T_warmup, T_total=T_total,
                  weight_decay=1e-4)

    rng = np.random.default_rng(42)

    history = {
        'train_loss': [], 'val_loss': [], 'lr': [],
        'val_B': [],      'val_H'  : [], 'val_mono': [],
    }
    best_val   = np.inf
    best_w     = [p.copy() for p, _ in net.all_params()]
    no_improve = 0
    t0         = time.perf_counter()

    if verbose:
        print(f'  Entrainement : {n} exemples,  {n_steps_ep} steps/epoch')
        print(f'  Architecture : m={net.m}, d={net.d}, L={net.n_layers}, '
              f'params={net.n_params():,}')

    for epoch in range(n_epochs):
        # Curriculum : lambda_H fort pendant le warmup
        lH_epoch = lambda_H * 1.5 if epoch < warmup_eps else lambda_H

        perm    = rng.permutation(n)
        ep_loss = []

        for i in range(n_steps_ep):
            idx = perm[i * batch_size: (i + 1) * batch_size]
            Xb  = X_train[idx].astype(np.float64)
            Yb  = Y_train[idx].astype(np.float64)

            if augment:
                Xb, Yb = augment_batch(Xb, Yb, rng)

            B_hat = net.forward(Xb, training=True)
            loss, grad, _ = compute_loss(
                B_hat, Yb, Xb,
                lambda_B=1.0, lambda_H=lH_epoch, lambda_mono=lambda_mono)
            net.backward(grad)
            lr = optim.step()
            ep_loss.append(loss)

        # Validation complete (pas de dropout)
        B_v    = net.forward(X_val.astype(np.float64), training=False)
        v_loss, _, vp = compute_loss(
            B_v, Y_val.astype(np.float64), X_val.astype(np.float64),
            lambda_B=1.0, lambda_H=lambda_H, lambda_mono=lambda_mono)

        tl = float(np.mean(ep_loss))
        history['train_loss'].append(tl)
        history['val_loss'].append(float(v_loss))
        history['lr'].append(float(lr))
        history['val_B'].append(float(vp['B']))
        history['val_H'].append(float(vp['H']))
        history['val_mono'].append(float(vp['mono']))

        if verbose and (epoch + 1) % 10 == 0:
            elapsed = time.perf_counter() - t0
            print(f'  Ep {epoch+1:4d}/{n_epochs}  '
                  f'train={tl:.5f}  val={float(v_loss):.5f}  '
                  f'[B={vp["B"]:.4f} H={vp["H"]:.4f}]  '
                  f'lr={lr:.2e}  ({elapsed:.0f}s)')

        if float(v_loss) < best_val - 1e-5:
            best_val   = float(v_loss)
            best_w     = [p.copy() for p, _ in net.all_params()]
            no_improve = 0
            if save_path:
                net.save(save_path)
        else:
            no_improve += 1

        if no_improve >= patience:
            if verbose:
                print(f'  [Early stop] epoch {epoch+1},  best_val={best_val:.5f}')
            break

    # Restaurer les meilleurs poids
    for (p, _), w in zip(net.all_params(), best_w):
        p[:] = w

    history['best_val_loss']  = best_val
    history['n_epochs_done']  = epoch + 1
    history['total_time_s']   = round(time.perf_counter() - t0, 1)

    if verbose:
        print(f'\n  Termine : {epoch+1} epochs, {history["total_time_s"]}s, '
              f'best_val={best_val:.5f}')

    return history
