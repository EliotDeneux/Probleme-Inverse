# Probleme-Inverse — Division Cellulaire

Projet sur données réelles et simulées pour le cours de Marie Doumic - Master MSV

Données téléchargées via : https://team.inria.fr/merge/enseignement-cours-master-2/

Pour utiliser les scripts : cloner le repo, puis créer un dossier /data dans .../Probleme-Inverse ; puis mettre les données téléchargées dans .../Probleme-Inverse/data


---


Estimation du taux de division cellulaire $B$ à partir de données de population, par des méthodes de **problèmes inverses linéaires**

---

## Structure du projet

```
Probleme-Inverse/
├── run_all.py               ← script principal (voir ci-dessous)
├── simulate_division.py     ← simulation des données
├── evaluate.py              ← métriques d'erreur et études de convergence
├── plots.py                 ← toutes les fonctions de visualisation
│
├── src/                     ← bibliothèque cœur (modules réutilisables)
│   ├── direct_problem.py    ← opérateur Ψ et problème direct
│   ├── density_estimation.py← KDE, Nelson-Aalen, estimateur de hasard
│   ← linear_inverse.py     ← SVD analytique, Tikhonov, TSVD
│   ├── parameter_selection.py← sélection du paramètre α
│   ├── hazard_estimation.py ← pipeline complet d'estimation
│   └── __init__.py
│
└── data/
    ├── age/        {constant, weibull2, step}.npz
    ├── size/       {constant, linear, power}.npz
    ├── increment/  {constant, weibull2, step}.npz
    ├── cells.db    ← base SQLite unifiée
    └── metadata.json
```

---

## Lancement rapide

```bash
# Tout exécuter (sauf étude de convergence qui est longue)
python run_all.py

# Seulement le modèle âge, sous-échantillonné à 2000 cellules
python run_all.py --model age --n-max 2000

# Étape spécifique
python run_all.py --steps inverse alpha --model age increment

# Sans générer de figures (calculs uniquement)
python run_all.py --no-plots

# Étude de convergence (≈ 5 min)
python run_all.py --steps convergence
```

### Options de `run_all.py`

| Option | Valeurs possibles | Défaut |
|--------|-------------------|--------|
| `--steps` | `direct` `inverse` `alpha` `convergence` `summary` | tous sauf `convergence` |
| `--model` | `age` `size` `increment` | tous |
| `--n-max` | entier | 10 000 (complet) |
| `--no-plots` | flag | figures activées |

### Sorties produites

| Dossier | Contenu |
|---------|---------|
| `figures/` | Figures PNG (problème direct, estimations, courbes α, convergence, heatmaps) |
| `results/` | Métriques JSON (erreurs L², statistiques KS, paramètres α sélectionnés) |

---

## Modèles et taux de division

### Trois modèles de division

La croissance est exponentielle : $X_{ud} = X_{ub} \cdot e^{K A_{ud}}$ avec $K = \ln 2 / 70\ \text{min}^{-1}$.

| Modèle | Variable $T$ | Distribution | Opérateur |
|--------|-------------|--------------|-----------|
| **Âge** | $A_{ud}$ (âge à la division) | $f(a) = B(a)\,e^{-\int_0^a B}$ | $\Psi : B \mapsto \int_0^{(\cdot)} B$ |
| **Incrément** | $Z_{ud} = X_{ud} - X_{ub}$ | $f(z) = B(z)\,e^{-\int_0^z B}$ | idem |
| **Taille** | $X_{ud}$ (conditionné sur $X_{ub}$) | $f(x) = B(x)\,e^{-\int_{X_{ub}}^x B}$ | $\Psi$ avec troncature gauche |

### Taux simulés

**Modèle âge / incrément** (même structure mathématique) :

| Nom | Formule $B(t)$ | Distribution résultante | Paramètres |
|-----|---------------|------------------------|------------|
| `constant` | $\lambda$ | Exponentielle($\lambda$) | $\lambda = 0.02\ \text{min}^{-1}$ (âge), $2.0\ \mu\text{m}^{-1}$ (incr.) |
| `weibull2` | $2t/\sigma^2$ | Weibull($k=2$, $\sigma$) | $\sigma = 60\ \text{min}$ (âge), $0.7\ \mu\text{m}$ (incr.) |
| `step` | $\lambda \cdot \mathbf{1}(t \geq t_0)$ | $t_0 +$ Exponentielle($\lambda$) | $\lambda=0.05$, $t_0=20$ (âge) |

**Modèle taille** :

| Nom | Formule $B(x)$ | Interprétation biologique |
|-----|---------------|--------------------------|
| `constant` | $\beta$ | Incrément exponentiel (*adder* pur) |
| `linear` | $\alpha x$ | Seuil de taille absolu (modèle *sizer*) |
| `power` | $\beta \sqrt{x}$ | Interpolation *adder*/*sizer* |

---

## Documentation des modules

---

### `src/direct_problem.py`

Implémente l'**opérateur direct** $\Psi : B \mapsto H = \int_0^{(\cdot)} B$
et le calcul de toutes les quantités dérivées.

#### `DivisionRateSpec`
Dataclass décrivant un taux de division.

| Attribut | Type | Description |
|----------|------|-------------|
| `name` | `str` | Identifiant court (`'constant'`, `'weibull2'`, …) |
| `func` | `Callable` | $B(t)$ — accepte et retourne un `np.ndarray` |
| `params` | `dict` | Paramètres numériques |
| `description` | `str` | Formule lisible ($B(t) = 2t/\sigma^2$, …) |
| `domain` | `str` | `'age'` \| `'size'` \| `'increment'` |

#### Fonctions constructeurs

| Fonction | Signature | Description |
|----------|-----------|-------------|
| `make_constant(lam, domain)` | `float, str` | $B(t) = \lambda$ |
| `make_weibull2(sigma, domain)` | `float, str` | $B(t) = 2t/\sigma^2$ |
| `make_step(lam, t0, domain)` | `float, float, str` | $B(t) = \lambda \cdot \mathbf{1}(t \geq t_0)$ |
| `make_linear(alpha)` | `float` | $B(x) = \alpha x$ |
| `make_power(beta, gamma)` | `float, float` | $B(x) = \beta x^\gamma$ |

#### `KNOWN_RATES`
`Dict[tuple, DivisionRateSpec]` — catalogue des 9 taux simulés, indexé par `(model, rate_name)`.

#### `DirectProblemSolver`
Résout le problème direct : $B \mapsto \{H, S, f, F\}$.

| Méthode | Entrée | Sortie | Description |
|---------|--------|--------|-------------|
| `__init__(grid)` | `np.ndarray` | — | Initialise avec la grille de discrétisation |
| `compute_H(B_vals)` | array | array | $H(t) = \int_0^t B(s)\,ds$ (opérateur $\Psi$) |
| `compute_S(B_vals)` | array | array | $S(t) = e^{-H(t)}$ (survie) |
| `compute_f(B_vals)` | array | array | $f(t) = B(t) \cdot S(t)$ (densité) |
| `compute_all(B_vals)` | array | `dict` | Retourne `{grid, B, H, S, f, F}` |
| `compute_mean(B_vals)` | array | `float` | $\mathbb{E}[T] = \int_0^\infty S(t)\,dt$ |
| `verify_normalization(B_vals)` | array | `float` | $\int_0^\infty f(t)\,dt$ (doit être $\approx 1$) |
| `integration_matrix` | — | `np.ndarray` (m×m) | Matrice $A$ telle que $(AB)_i \approx \int_0^{t_i} B$ (trapèzes) |
| `apply_forward(B_vals)` | array | array | $H = A \cdot B$ (produit matriciel) |
| `apply_adjoint(H_vals)` | array | array | $\Psi^* H = A^T \cdot H$ |
| `grid_from_data(data, n_points, quantile_max)` | array | array | Construit une grille $[0, t_{\max}]$ adaptée |

---

### `src/density_estimation.py`

Estimation non-paramétrique de $f$ et $H$ à partir des données.

#### Sélection de bande passante

| Fonction | Signature | Description |
|----------|-----------|-------------|
| `bandwidth_silverman(data)` | array → `float` | $h = 0.9 \cdot \min(\sigma, \text{IQR}/1.34) \cdot n^{-1/5}$ |
| `bandwidth_scott(data)` | array → `float` | $h = 1.06 \cdot \sigma \cdot n^{-1/5}$ |
| `bandwidth_cross_validation(data, kernel, n_h)` | array → `float` | LOO-CV ($O(n^2)$, adapté $n \leq 5000$) |

#### `KernelDensityEstimator`
Estimateur à noyau $\hat{f}_h(t) = \frac{1}{nh} \sum_i K\!\left(\frac{t-T_i}{h}\right)$.

| Méthode | Description |
|---------|-------------|
| `__init__(kernel, bandwidth)` | `kernel` : `'gaussian'`\|`'epanechnikov'` ; `bandwidth` : `float`, `'silverman'`, `'scott'` ou `'cv'` |
| `fit(data)` | Calibre $h$ et stocke les données |
| `predict(grid)` | Évalue $\hat{f}_h$ sur la grille |
| `bandwidth_value` | Propriété : retourne $h$ calculé |

#### `NelsonAalanEstimator`
Estimateur non-paramétrique du hasard cumulé $\hat{H}(t) = \sum_{T_i \leq t} d_i/n_i$.
C'est la **mesure bruitée** $z^\varepsilon \approx H = \Psi B$ utilisée en entrée des méthodes de régularisation.

| Méthode | Description |
|---------|-------------|
| `fit(observations, entry_times=None)` | `entry_times` = $X_{ub}$ pour le modèle taille (troncature gauche) ; `None` pour âge/incrément |
| `predict(grid)` | Interpole $\hat{H}$ (en escalier) sur la grille |
| `smooth(grid, sigma_grid=5.0)` | Mollification gaussienne $\hat{H}_\sigma = \rho_\sigma \star \hat{H}$ |
| `noise_level_estimate(n)` | Retourne $\varepsilon \approx 1/\sqrt{n}$ |

#### `KDEHazardEstimator`
Estimation directe $\hat{B}(t) = \hat{f}_h(t) / \hat{S}(t)$.

| Méthode | Description |
|---------|-------------|
| `fit(observations, entry_times=None)` | Calibre le KDE |
| `predict_f(grid)` | Densité $\hat{f}_h$ |
| `predict_S(grid)` | Survie empirique $\hat{S}$ |
| `predict_B(grid)` | Taux $\hat{B} = \hat{f}/\hat{S}$ (tronqué en queue) |

---

### `src/linear_inverse.py`

Méthodes de régularisation pour l'opérateur d'intégration.

**Problème inverse** : estimer $B$ à partir de $H^\varepsilon \approx H = \Psi B$.

**SVD analytique de $\Psi$ sur $[0, T]$** (cours Section 3.4.1) :
$$\sigma_j = \frac{2T}{\pi(2j+1)}, \quad e_j(t) = \sqrt{\tfrac{2}{T}}\cos(\lambda_j t), \quad f_j(t) = \sqrt{\tfrac{2}{T}}\sin(\lambda_j t)$$

#### `AnalyticSVD`
SVD analytique de $\Psi$.

| Méthode | Description |
|---------|-------------|
| `__init__(T, J_max)` | $T$ = longueur du domaine, $J_{\max}$ = nombre de modes |
| `e_j(grid, j)` | Vecteur singulier droit $e_j(t) = \sqrt{2/T}\cos(\lambda_j t)$ |
| `f_j(grid, j)` | Vecteur singulier gauche $f_j(t) = \sqrt{2/T}\sin(\lambda_j t)$ |
| `inner_product_L2(u, v, grid)` | $\langle u, v \rangle_{L^2}$ par la règle des trapèzes |
| `picard_coefficients(H_eps, grid)` | $\hat{c}_j = \langle H^\varepsilon, f_j \rangle_{L^2}$ (critère de Picard) |
| `reconstruct(coeffs, filter_r, grid)` | $\hat{B}(t) = \sum_j r(\alpha, \sigma_j)/\sigma_j \cdot \hat{c}_j \cdot e_j(t)$ |

#### `TruncatedSVD`
SVD tronquée via la SVD analytique. Filtre : $r(\alpha, \sigma) = \mathbf{1}_{\sigma \geq \alpha}$.

| Méthode | Description |
|---------|-------------|
| `fit(H_eps, grid)` | Calcule les coefficients de Picard |
| `predict(alpha)` | $\hat{B}_\alpha$ avec le filtre de troncature (seuil $\alpha$ sur les $\sigma_j$) |
| `predict_n_modes(N)` | Reconstruction avec exactement $N$ modes |
| `sigma_values()` | Retourne les valeurs singulières $\sigma_j$ |
| `picard_plot_data()` | $(j, \sigma_j, |\hat{c}_j|)$ pour le diagramme de Picard |

#### `TikhonovRegularizer`
Régularisation de Tikhonov via discrétisation matricielle.
$$\hat{B}_\alpha = \arg\min_B \tfrac{1}{2}\|AB - H^\varepsilon\|^2 + \tfrac{\alpha^2}{2}\|LB\|^2$$

| Méthode | Description |
|---------|-------------|
| `__init__(A, p)` | `p=0` : $L=I$ (classique), `p=1` : $L=D_1$ (dérivée 1ère), `p=2` : $L=D_2$ |
| `fit(H_eps)` | Stocke $A^T H^\varepsilon$ |
| `predict(alpha)` | Résout $(A^T A + \alpha^2 L^T L)\hat{B} = A^T H^\varepsilon$ |
| `predict_via_svd(alpha)` | Formule explicite SVD (seulement `p=0`) |
| `residual(alpha)` | $\|A\hat{B}_\alpha - H^\varepsilon\|_2$ |
| `filter_function(alpha)` | Valeurs $r_p(\alpha, \sigma_j) = \sigma_j^2/(\sigma_j^2 + \alpha^2)$ |
| `effective_degrees_of_freedom(alpha)` | $\text{df}(\alpha) = \text{tr}(A(A^TA+\alpha^2I)^{-1}A^T)$ |

---

### `src/parameter_selection.py`

Méthodes de sélection du paramètre de régularisation $\alpha$.

#### `DiscrepancyPrinciple`
Principe de Morozov (a posteriori) : trouver $\alpha^*$ tel que $\|A\hat{B}_{\alpha^*} - H^\varepsilon\|_2 = \tau \varepsilon \sqrt{m}$.

| Méthode | Description |
|---------|-------------|
| `__init__(tau, alpha_min, alpha_max)` | $\tau > 1$ (défaut 1.1), bornes de recherche |
| `select(tikhonov, noise_level)` | Retourne $\alpha^*$ par dichotomie (Brentq) |
| `curve(tikhonov, alpha_grid)` | $({\alpha}, \text{résidu}(\alpha))$ pour visualisation |

#### `GeneralizedCrossValidation`
$$\text{GCV}(\alpha) = m \cdot \|A\hat{B}_\alpha - H^\varepsilon\|^2 / (m - \text{df}(\alpha))^2$$

| Méthode | Description |
|---------|-------------|
| `select(tikhonov, alpha_grid)` | Minimise GCV($\alpha$) |
| `values(tikhonov, alpha_grid)` | GCV($\alpha$) pour toute la grille |
| `curve(tikhonov, alpha_grid)` | Alias de `values` |

#### `LCurveMethod`
Heuristique : trouver le « coin » de la courbe $\log\|\hat{B}_\alpha\|$ vs $\log\|A\hat{B}_\alpha - H^\varepsilon\|$.

| Méthode | Description |
|---------|-------------|
| `compute_curve(tikhonov, alpha_grid)` | Retourne `(alpha_grid, log_residuals, log_norms)` |
| `select(tikhonov, alpha_grid)` | $\alpha^*$ au point de courbure maximale |

#### Fonctions autonomes

| Fonction | Description |
|----------|-------------|
| `alpha_apriori(epsilon, delta, s, p)` | $\alpha^* = (\varepsilon/\delta)^{1/(2s+1)}$ (cours Prop. 8) |
| `theoretical_convergence_rate(n_values, s, C)` | $C \cdot n^{-s/(2s+1)}$ (taux théorique optimal) |
| `alpha_grid_log(alpha_min, alpha_max, n_points)` | Grille logarithmique de $\alpha$ |

---

### `src/hazard_estimation.py`

Pipeline complet : données → $\hat{H}$ (Nelson-Aalen) → régularisation → $\hat{B}$.

#### `EstimationResult`
Dataclass résultat d'une estimation.

| Attribut | Type | Description |
|----------|------|-------------|
| `grid` | array | Grille de discrétisation |
| `B_hat` | array | Taux estimé $\hat{B}(t)$ |
| `B_true` | array\|None | Taux théorique $B(t)$ |
| `alpha` | float\|None | Paramètre de régularisation utilisé |
| `method` | str | Nom de la méthode |
| `model` | str | `'age'`, `'size'` ou `'increment'` |
| `rate_name` | str | Nom du taux (`'constant'`, `'weibull2'`, …) |
| `n_cells` | int | Taille de l'échantillon |
| `noise_level` | float | $\varepsilon \approx 1/\sqrt{n}$ |
| `extras` | dict | Données supplémentaires (résidus, courbe L, Picard, …) |
| `l2_error()` | → float | $\|\hat{B}-B\|_{L^2} / \|B\|_{L^2}$ (erreur relative) |
| `linf_error()` | → float | $\|\hat{B}-B\|_{L^\infty} / \max B$ |

#### `load_observations(model, rate_name, data_dir, n_max)`
Charge un fichier `.npz` et retourne les observations selon le modèle :
- `'age'` → `{'T': A_ud, 'raw': …}`
- `'increment'` → `{'T': Z_ud, 'raw': …}`
- `'size'` → `{'T': X_ud, 'X_ub': X_ub, 'raw': …}`

#### `HazardEstimationPipeline`
Pipeline principal.

| Méthode | Description |
|---------|-------------|
| `__init__(n_grid, quantile, J_tsvd)` | Paramètres globaux du pipeline |
| `run(model, rate_name, method, alpha_selection, data_dir, n_max, fixed_alpha)` | Lance l'estimation et retourne un `EstimationResult` |

**Méthodes disponibles** (`method=`) :

| Valeur | Algorithme | Qualification |
|--------|-----------|---------------|
| `'kde'` | $\hat{B} = \hat{f}_h / \hat{S}$ | — |
| `'tsvd'` | SVD tronquée analytique | $\infty$ |
| `'tikhonov_0'` | Tikhonov $p=0$, $L=I$ | 1 |
| `'tikhonov_1'` | Tikhonov $p=1$, $L=D_1$ | 3 |
| `'tikhonov_2'` | Tikhonov $p=2$, $L=D_2$ | 5 |

**Sélections de $\alpha$** (`alpha_selection=`) : `'discrepancy'`, `'gcv'`, `'lcurve'`, `'apriori_s1.0'` (remplacer `1.0` par la régularité souhaitée).

#### `run_all_methods(model, rate_name, data_dir, n_max)`
Lance les 5 méthodes et retourne `Dict[str, EstimationResult]`.

---

### `evaluate.py`

Métriques d'erreur, études de convergence et comparaisons.

| Fonction | Signature | Description |
|----------|-----------|-------------|
| `compute_errors(result)` | `EstimationResult → dict` | Calcule `l2_abs`, `l2_rel`, `linf_abs`, `linf_rel`, `alpha`, `n` |
| `verify_direct_problem(model, rate_name, data_dir)` | → `dict` | Compare histogramme empirique et densité théorique $f$, calcule la statistique KS. ⚠️ Pour `'size'`, le KS est élevé (attendu : $F$ est conditionnelle à $X_{ub}$). |
| `convergence_study(model, rate_name, method, alpha_selection, n_values, n_repeat, data_dir, seed)` | → `dict` | Étudie $\|\hat{B}-B\|_{L^2}$ en fonction de $n$ (sous-échantillonnage avec répétitions). Retourne `{n_values, l2_mean, l2_std, l2_all, alpha_mean}`. |
| `compare_all_methods(model, data_dir, n_max)` | → `dict` | Lance toutes les méthodes sur tous les taux d'un modèle. Retourne `{rate: {method: error_dict}}`. |
| `summary_table(comparison_dict)` | → `(array, rates, methods)` | Tableau (taux × méthodes) des erreurs L² relatives. |
| `analyze_alpha_selection(model, rate_name, data_dir)` | → `dict` | Compare discordance, GCV, courbe-L, et $\alpha$ a priori. Retourne les $\alpha$ sélectionnés, les erreurs associées, et les courbes pour visualisation. |

---

### `plots.py`

Toutes les fonctions de visualisation.

| Fonction | Figure produite | Description |
|----------|----------------|-------------|
| `plot_direct_problem(model, rate_name, data_dir, save_path)` | 2×2 : $B$, $H$, $f$+histogramme, $S$ | Vérifie la cohérence simulateur/théorie |
| `plot_estimation_results(results, title, save_path)` | 1×2 : $\hat{B}$ vs $B$ + erreur absolue | Compare toutes les méthodes sur un dataset |
| `plot_alpha_selection(analysis, title, save_path)` | 2×2 : résidu, GCV, courbe-L, barplot erreurs | Analyse la sélection de $\alpha$ |
| `plot_convergence(conv_results, theoretical_s, title, save_path)` | Log-log $L^2$ vs $n$ | Compare la convergence empirique avec les taux théoriques |
| `plot_error_heatmap(comparison, model, save_path)` | Heatmap taux×méthodes | Vue d'ensemble des erreurs $L^2$ relatives |
| `plot_picard_criterion(result, save_path)` | 1×2 : $|\hat{c}_j|$ et $|\hat{c}_j|/\sigma_j$ | Diagnostique le mal-positude |
| `plot_global_summary(all_results, save_path)` | Grille de panneaux | Vue d'ensemble tous modèles / tous taux |

---

## Rappel mathématique

### Opérateur direct

$$\Psi : B \mapsto H, \quad (\Psi B)(t) = \int_0^t B(s)\,ds$$

C'est l'opérateur paradigmatique du cours (compact, injectif, image dense → **problème mal posé**).

### SVD analytique de $\Psi$ sur $[0,T]$

$$\sigma_j = \frac{2T}{\pi(2j+1)}, \quad j = 0, 1, 2, \ldots \quad \Rightarrow \quad \sigma_j = O(j^{-1}) \text{ (degré de mal-positude = 1)}$$

### Estimateurs régularisés

$$\hat{B}_\alpha = \sum_j \frac{r(\alpha, \sigma_j)}{\sigma_j} \langle H^\varepsilon, f_j \rangle \, e_j$$

| Méthode | Filtre $r(\alpha, \sigma)$ | Qualification |
|---------|--------------------------|---------------|
| SVD tronquée | $\mathbf{1}_{\sigma \geq \alpha}$ | $\infty$ |
| Tikhonov $p=0$ | $\sigma^2/(\sigma^2 + \alpha^2)$ | 1 |
| Tikhonov $p=1$ | $1/(1 + \alpha^2 \sigma^{-6})$ | 3 |

### Taux de convergence (pour $B \in Y^s$, $\varepsilon = 1/\sqrt{n}$)

$$\|\hat{B}_{\alpha^*} - B\|_{L^2} = O\!\left(n^{-s/(2s+1)}\right) \quad \text{(optimal)}$$

La qualification limite l'ordre optimal atteignable : Tikhonov $p=0$ sature à $O(n^{-1/3})$ pour $s > 1$.

---

## Format des données

Les données simulées sont stockées en deux formats complémentaires :

**Fichiers `.npz`** (NumPy compressé, ~150 KB chacun) :
```python
import numpy as np
d = np.load('data/age/constant.npz')
# Colonnes : cell_id, birth_size (Xub), division_age (Aud),
#            division_size (Xud), increment (Zud = Xud - Xub)
```

**Base SQLite** (`data/cells.db`, requêtes cross-modèles) :
```python
import sqlite3, pandas as pd
df = pd.read_sql(
    "SELECT * FROM cells WHERE model='age' AND rate='constant'",
    sqlite3.connect('data/cells.db')
)
```