"""
ml_core.py — Reseau de neurones NumPy pur pour l'apprentissage d'operateurs.

PRINCIPE
════════
On apprend l'operateur inverse de Psi : B -> H = int_0^. B.

    Phi_theta : H^eps in R^m  ──►  B_hat in R^m

Le reseau est entraine sur des paires (H_noisy, B_true) generees
synthetiquement depuis des familles parametriques variees.

ARCHITECTURE : MLP Residuel Pre-Norm
══════════════════════════════════════

    H^eps (m,)
        │
    Linear(m -> d)  +  GELU             ← encodeur
        │
    ┌── ResidualBlock x n_layers ──┐
    │   x -> LayerNorm(d)          │    ← pre-normalisation
    │     -> Linear(d -> 4d)       │
    │     -> GELU                  │
    │     -> Dropout(p)            │    ← regularisation
    │     -> Linear(4d -> d)       │
    │     -> + x  (skip)           │    ← connexion residuelle
    └──────────────────────────────┘
        │
    LayerNorm(d)
        │
    Linear(d -> m)                       ← decodeur
        │
    Softplus(.)  >= 0                    ← contrainte B >= 0
        │
    B_hat (m,)

Justifications :
  - GELU : meilleure approximation des fonctions lisses que ReLU
  - Residuel pre-norm : convergence plus stable, gradient fluide
  - 4x expansion dans le FFN : capacite maximale (style Transformer)
  - Softplus en sortie : contrainte physique B >= 0
  - Dropout 10% : regularisation + MC-Dropout pour l'incertitude
  - LayerNorm : invariance a l'echelle, crucial car H varie de 0.1 a 50+

OPTIMISEUR : AdamW + Warmup lineaire + Decroissance cosinus
═══════════════════════════════════════════════════════════
  t <= T_w : lr = lr_max * t / T_w
  t >  T_w : lr = lr_min + 0.5*(lr_max-lr_min)*(1+cos(pi*p))
  Weight decay decouple (pas applique aux biais et gammas).
"""

from __future__ import annotations
import pickle
import numpy as np
from typing import Iterator, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# Activations
# ═══════════════════════════════════════════════════════════════════════════

def gelu(x: np.ndarray) -> np.ndarray:
    """GELU via approximation tanh (Hendrycks & Gimpel 2016)."""
    c = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))


def dgelu(x: np.ndarray) -> np.ndarray:
    """Derivee de GELU."""
    c    = np.sqrt(2.0 / np.pi)
    arg  = c * (x + 0.044715 * x ** 3)
    tanh = np.tanh(np.clip(arg, -15, 15))
    return (0.5 * (1.0 + tanh)
            + 0.5 * x * (1.0 - tanh**2) * c * (1.0 + 3.0 * 0.044715 * x**2))


def softplus(x: np.ndarray) -> np.ndarray:
    """log(1 + e^x) >= 0 — garantit B_hat >= 0."""
    return np.where(x > 30.0, x, np.log1p(np.exp(np.clip(x, -500, 30))))


def dsoftplus(x: np.ndarray) -> np.ndarray:
    """Derivee de softplus = sigmoide."""
    return np.where(x > 30.0, 1.0, 1.0 / (1.0 + np.exp(-np.clip(x, -500, 30))))


# ═══════════════════════════════════════════════════════════════════════════
# Couche lineaire
# ═══════════════════════════════════════════════════════════════════════════

class Linear:
    """
    y = x W^T + b,   W in R^{d_out x d_in},   b in R^{d_out}.

    Initialisation He adaptee a GELU : W ~ N(0, sqrt(2/d_in)).
    """

    def __init__(self, d_in: int, d_out: int, bias: bool = True):
        self.W  = np.random.randn(d_out, d_in) * np.sqrt(2.0 / d_in)
        self.b  = np.zeros(d_out) if bias else None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b) if bias else None
        self._x: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x : (batch, d_in) -> (batch, d_out)."""
        self._x = x
        out = x @ self.W.T
        if self.b is not None:
            out = out + self.b
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """dout : (batch, d_out) -> dx : (batch, d_in).  Accumule dW, db."""
        self.dW = dout.T @ self._x
        if self.b is not None:
            self.db = dout.sum(axis=0)
        return dout @ self.W

    def params(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield (valeur, gradient) pour l'optimiseur."""
        yield self.W, self.dW
        if self.b is not None:
            yield self.b, self.db


# ═══════════════════════════════════════════════════════════════════════════
# Layer Normalization
# ═══════════════════════════════════════════════════════════════════════════

class LayerNorm:
    """
    y = (x - mu) / sqrt(var + eps) * gamma + beta.

    Stabilise l'entrainement en maintenant les activations a echelle
    unitaire independamment de la magnitude de l'entree H^eps.
    """

    def __init__(self, d: int, eps: float = 1e-6):
        self.gamma  = np.ones(d)
        self.beta   = np.zeros(d)
        self.eps    = eps
        self.dgamma = np.zeros(d)
        self.dbeta  = np.zeros(d)
        self._cache: Optional[Tuple] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        mu  = x.mean(axis=-1, keepdims=True)
        var = x.var( axis=-1, keepdims=True)
        xn  = (x - mu) / np.sqrt(var + self.eps)
        self._cache = (xn, var)
        return self.gamma * xn + self.beta

    def backward(self, dout: np.ndarray) -> np.ndarray:
        xn, var = self._cache
        d = dout.shape[-1]
        self.dgamma = (dout * xn).sum(axis=0)
        self.dbeta  =  dout.sum(axis=0)
        dxn  = dout * self.gamma
        dvar = (-0.5 * dxn * xn / (var + self.eps)).sum(axis=-1, keepdims=True)
        dmu  = (-dxn / np.sqrt(var + self.eps)).sum(axis=-1, keepdims=True)
        return dxn / np.sqrt(var + self.eps) + 2.0 * dvar * xn / d + dmu / d

    def params(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        yield self.gamma, self.dgamma
        yield self.beta,  self.dbeta


# ═══════════════════════════════════════════════════════════════════════════
# Bloc residuel pre-norm
# ═══════════════════════════════════════════════════════════════════════════

class ResidualBlock:
    """
    out = x + FC(4d->d)( GELU( Dropout( FC(d->4d)( LN(x) ) ) ) )

    La connexion residuelle (skip) permet au reseau d'apprendre des
    corrections incrementales a une identite initiale, ce qui accelere
    la convergence et stabilise les gradients sur les couches profondes.
    L'expansion 4x dans le FFN (Feed-Forward Network) suit le design
    des Transformers pour maximiser la capacite d'approximation.
    """

    def __init__(self, d: int, dropout_rate: float = 0.10):
        self.norm = LayerNorm(d)
        self.fc1  = Linear(d, 4 * d)
        self.fc2  = Linear(4 * d, d)
        self.p    = dropout_rate
        self._pre_gelu: Optional[np.ndarray] = None
        self._mask    : Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        h = self.norm.forward(x)
        h = self.fc1.forward(h)
        self._pre_gelu = h.copy()          # cache avant GELU pour backward
        h = gelu(h)
        if training and self.p > 0:
            self._mask = (np.random.rand(*h.shape) > self.p) / (1.0 - self.p)
        else:
            self._mask = np.ones_like(h)
        h = h * self._mask
        h = self.fc2.forward(h)
        return x + h                       # connexion residuelle

    def backward(self, dout: np.ndarray) -> np.ndarray:
        # dout -> fc2 (la skip connection transmet dout directement)
        dh = self.fc2.backward(dout)
        dh = dh * self._mask
        dh = dh * dgelu(self._pre_gelu)
        dh = self.fc1.backward(dh)
        dx = self.norm.backward(dh)
        return dout + dx                   # skip : dout non modifie

    def params(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        yield from self.norm.params()
        yield from self.fc1.params()
        yield from self.fc2.params()


# ═══════════════════════════════════════════════════════════════════════════
# Neural Operator
# ═══════════════════════════════════════════════════════════════════════════

class NeuralOperator:
    """
    Phi_theta : H^eps in R^m  ->  B_hat in R^m.

    Apprend l'operateur inverse de Psi : B -> int_0^. B.
    A utiliser apres entrainement avec ml_train.train().

    Parameters
    ----------
    m        : taille de la grille (defaut 128)
    d        : dimension de l'espace latent (defaut 256)
    n_layers : nombre de blocs residuels (defaut 6)
    dropout  : taux de dropout — aussi utilise en inference MC (defaut 0.10)
    """

    def __init__(self, m: int = 128, d: int = 256,
                 n_layers: int = 6, dropout: float = 0.10):
        self.m = m
        self.d = d
        self.n_layers = n_layers

        self.encoder  = Linear(m, d)
        self.blocks   = [ResidualBlock(d, dropout) for _ in range(n_layers)]
        self.norm_out = LayerNorm(d)
        self.decoder  = Linear(d, m)
        # Init biais decoder : softplus(0.5) ~ 0.97,
        # sorte que le reseau parte d'une prediction ~1 (ordre de grandeur de B)
        self.decoder.b += 0.5

        # Caches forward pour backward
        self._z_enc: Optional[np.ndarray] = None
        self._logB : Optional[np.ndarray] = None

    def forward(self, H: np.ndarray, training: bool = True) -> np.ndarray:
        """
        H : (batch, m) -> B_hat : (batch, m).
        Stocke les caches internes pour backward().
        """
        z_enc = self.encoder.forward(H)
        self._z_enc = z_enc.copy()
        z = gelu(z_enc)

        for block in self.blocks:
            z = block.forward(z, training=training)

        z    = self.norm_out.forward(z)
        logB = self.decoder.forward(z)
        self._logB = logB
        return softplus(logB)               # garantit B_hat >= 0

    def backward(self, dB_hat: np.ndarray) -> None:
        """Retropropage dL/dB_hat. Accumule les gradients dans chaque couche."""
        d_logB = dB_hat * dsoftplus(self._logB)
        dz     = self.norm_out.backward(self.decoder.backward(d_logB))
        for block in reversed(self.blocks):
            dz = block.backward(dz)
        self.encoder.backward(dz * dgelu(self._z_enc))

    def predict_with_uncertainty(self, H: np.ndarray,
                                  n_mc: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """
        MC-Dropout : n_mc passages en mode training pour estimer l'incertitude.

        Retourne (B_mean, B_std) ou B_std est un proxy de l'incertitude
        epistemique du modele — zones de grande incertitude = faible confiance.

        References : Gal & Ghahramani (2016) "Dropout as a Bayesian Approximation"
        """
        samples = np.stack([
            self.forward(H, training=True) for _ in range(n_mc)
        ])                                  # (n_mc, batch, m)
        return samples.mean(axis=0), samples.std(axis=0)

    def all_params(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Itere sur tous les (valeur, gradient) du reseau."""
        yield from self.encoder.params()
        for block in self.blocks:
            yield from block.params()
        yield from self.norm_out.params()
        yield from self.decoder.params()

    def n_params(self) -> int:
        """Nombre total de parametres."""
        return sum(p.size for p, _ in self.all_params())

    def save(self, path: str) -> None:
        data = {
            'm': self.m, 'd': self.d, 'n_layers': self.n_layers,
            'weights': [p.copy() for p, _ in self.all_params()],
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f'  Modele sauvegarde -> {path}')

    @classmethod
    def load(cls, path: str) -> 'NeuralOperator':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        net = cls(m=data['m'], d=data['d'], n_layers=data['n_layers'])
        for (p, _), w in zip(net.all_params(), data['weights']):
            p[:] = w
        print(f'  Modele charge : m={data["m"]}, d={data["d"]}, '
              f'L={data["n_layers"]}, params={net.n_params():,}')
        return net


# ═══════════════════════════════════════════════════════════════════════════
# Optimiseur : AdamW + Warmup + Cosinus
# ═══════════════════════════════════════════════════════════════════════════

class AdamW:
    """
    AdamW (Loshchilov & Hutter, ICLR 2019) avec schedule lr.

    Le weight decay decouple regularise les poids independamment
    de la magnitude des gradients — superieur au L2 classique avec Adam.

    Schedule :
        t <= T_warmup : lr = lr_max * t / T_warmup  (montee lineaire)
        t >  T_warmup : lr = lr_min + 0.5*(lr_max-lr_min)*(1 + cos(pi*p))
                        p = (t - T_warmup) / (T_total - T_warmup)

    Parameters
    ----------
    net          : NeuralOperator dont les parametres sont mis a jour
    lr_max       : learning rate de crete
    lr_min       : learning rate plancher (fin du cosinus)
    beta1/beta2  : moments Adam (defauts classiques 0.9/0.999)
    weight_decay : regularisation L2 decouple
    T_warmup     : steps de montee lineaire
    T_total      : steps totaux (pour calculer le cosinus)
    """

    def __init__(self, net: NeuralOperator,
                 lr_max: float       = 3e-4,
                 lr_min: float       = 3e-6,
                 beta1: float        = 0.90,
                 beta2: float        = 0.999,
                 eps: float          = 1e-8,
                 weight_decay: float = 1e-4,
                 T_warmup: int       = 300,
                 T_total: int        = 5000):
        self.net          = net
        self.lr_max       = lr_max
        self.lr_min       = lr_min
        self.b1, self.b2  = beta1, beta2
        self.eps          = eps
        self.wd           = weight_decay
        self.T_w          = T_warmup
        self.T_tot        = T_total
        self.t            = 0

        params   = list(net.all_params())
        self.ms  = [np.zeros_like(p) for p, _ in params]
        self.vs  = [np.zeros_like(p) for p, _ in params]

    def current_lr(self) -> float:
        t = self.t
        if t <= self.T_w:
            return self.lr_max * t / max(self.T_w, 1)
        p = min((t - self.T_w) / max(self.T_tot - self.T_w, 1), 1.0)
        return self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1.0 + np.cos(np.pi * p))

    def step(self) -> float:
        """Effectue une mise a jour Adam. Retourne le lr courant."""
        self.t += 1
        lr = self.current_lr()
        b1, b2, eps = self.b1, self.b2, self.eps
        for i, (p, g) in enumerate(self.net.all_params()):
            self.ms[i] = b1 * self.ms[i] + (1 - b1) * g
            self.vs[i] = b2 * self.vs[i] + (1 - b2) * g ** 2
            m_hat = self.ms[i] / (1 - b1 ** self.t)
            v_hat = self.vs[i] / (1 - b2 ** self.t)
            # AdamW : weight decay decouple (s'applique sur p, pas sur le gradient)
            p -= lr * (m_hat / (np.sqrt(v_hat) + eps) + self.wd * p)
        return lr
