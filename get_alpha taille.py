import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize_scalar
from scipy.stats import gaussian_kde
import warnings

warnings.filterwarnings('ignore')

# =====================================================================
# 0. OUTILS DE SANITY CHECK ET DÉBOGAGE
# =====================================================================

def check_data_integrity(xb, xd, age):
    """Vérifie la cohérence biologique et numérique des données."""
    print("--- [SANITY CHECK] ---")
    
    # 1. Vérification des tailles
    issues = np.where(xd <= xb)[0]
    if len(issues) > 0:
        print(f"⚠️ ATTENTION : {len(issues)} cellules ont une taille xd <= xb !")
        # Optionnel : on pourrait filtrer ces données ici
    
    # 2. Vérification des valeurs nulles ou négatives
    if np.any(xb <= 0) or np.any(xd <= 0) or np.any(age <= 0):
        print("⚠️ ATTENTION : Valeurs négatives ou nulles détectées dans les données.")

    # 3. Statistiques descriptives pour le débogage d'échelle
    print(f"Moyenne xb: {np.mean(xb):.2f} | Moyenne xd: {np.mean(xd):.2f}")
    print(f"Étendue des tailles (xd): [{np.min(xd):.2f} - {np.max(xd):.2f}]")
    print(f"Âge moyen: {np.mean(age):.2f}")
    print("----------------------\n")

# =====================================================================
# 1. NOYAUX ET ESTIMATION DE CROISSANCE
# =====================================================================

def kernel_gaussian(z):
    return (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * z**2)

def estimate_growth_rate(xb, xd, a):
    """ Estime le taux de croissance exponentiel moyen mu. """
    # g(t) = mu * x(t) => xd = xb * exp(mu * a) => mu = ln(xd/xb)/a
    mu_i = (1.0 / a) * np.log(xd / xb)
    return np.mean(mu_i)

# =====================================================================
# 2. ESTIMATEURS DU TAUX DE DIVISION B(x)
# =====================================================================

def estimate_B_size(xb, xd, grid, alpha, method='naive'):
    n = len(xd)
    B_est = np.zeros_like(grid)
    
    if method == 'naive':
        # z_ij = (a_i - A_j) / alpha
        z = (grid[:, None] - xd[None, :]) / alpha
        f_est = (1.0 / (n * alpha)) * np.sum(kernel_gaussian(z), axis=1)
        
        dx = grid[1] - grid[0]
        F_est = cumulative_trapezoid(f_est, initial=0, dx=dx)
        F_est = np.clip(F_est, 0, 1)
        
        S_est = 1.0 - F_est + 1e-10
        B_est = f_est / S_est
        
    elif method == 'rigorous':
        # Ramlau-Hansen : Ensemble à risque R(x)
        # Une cellule est à risque à la taille x si xb <= x < xd
        R_xd = np.array([np.sum((xb <= x_val) & (xd >= x_val)) for x_val in xd])
        R_xd = np.maximum(R_xd, 1) 
        
        for k, x_val in enumerate(grid):
            k_vals = kernel_gaussian((x_val - xd) / alpha)
            B_est[k] = (1.0 / alpha) * np.sum(k_vals / R_xd)
            
    return B_est

# =====================================================================
# 3. OPTIMISATION DE ALPHA
# =====================================================================

def expected_l2_error_size(alpha, n_sample, xb_full, xd_full, grid, B_ref, num_trials, method):
    dx = grid[1] - grid[0]
    total_error = 0.0
    indices = np.arange(len(xd_full))
    
    # Zone de calcul de l'erreur (on évite les bords extrêmes instables)
    low_b, high_b = np.percentile(xd_full, 5), np.percentile(xd_full, 95)
    valid_idx = (grid >= low_b) & (grid <= high_b)

    if not np.any(valid_idx): # Sécurité si la grille ne couvre pas les données
        return 1e10

    for _ in range(num_trials):
        idx_sub = np.random.choice(indices, size=n_sample, replace=False)
        B_est = estimate_B_size(xb_full[idx_sub], xd_full[idx_sub], grid, alpha, method=method)
        l2_err = np.sum((B_est[valid_idx] - B_ref[valid_idx])**2) * dx
        total_error += l2_err
        
    return total_error / num_trials

def find_optimal_alpha_size(n_sample, xb_full, xd_full, grid, B_ref, bounds, method):
    objective = lambda alpha: expected_l2_error_size(alpha, n_sample, xb_full, xd_full, grid, B_ref, 10, method)
    res = minimize_scalar(objective, bounds=bounds, method='bounded')
    return res.x

# =====================================================================
# 4. ANALYSE SIZER / ADDER / TIMER (S/A/T)
# =====================================================================

def analyze_sat(xb, xd):
    delta_x = xd - xb
    slope, intercept, r_value, p_value, std_err = linregress(xb, delta_x)
    
    if slope < -0.5:
        model_name = "Tendance SIZER"
    elif -0.5 <= slope <= 0.5:
        model_name = "Tendance ADDER"
    else:
        model_name = "Tendance TIMER"
        
    return slope, intercept, r_value, model_name

# =====================================================================
# 5. SCRIPT PRINCIPAL
# =====================================================================

def main():
    # --- A. CHARGEMENT ---
    try:
        data = np.loadtxt('data/Eric1002_MDJ_sb_sd_ad.txt', delimiter=',')
        xb_data, xd_data, age_data = data[:, 0], data[:, 1], data[:, 2]
    except Exception as e:
        print(f"Erreur chargement fichier : {e}")
        return

    # Sanity Check immédiat
    check_data_integrity(xb_data, xd_data, age_data)
    
    method_choice = 'rigorous' 
    N_total = len(xb_data)

    # --- B. RÉGLAGES DYNAMIQUES DE LA GRILLE ET ALPHA ---
    # On cadre la grille sur les tailles de division observées
    x_min_grid = np.percentile(xd_data, 1) * 0.8
    x_max_grid = np.percentile(xd_data, 99) * 1.1
    grid_x = np.linspace(x_min_grid, x_max_grid, 400)
    
    # Bornes pour alpha : entre 0.5% et 30% de l'écart type des données
    std_xd = np.std(xd_data)
    alpha_bounds = (0.05 * std_xd, 1.5 * std_xd)
    
    print(f"Grille: [{grid_x[0]:.2f} - {grid_x[-1]:.2f}]")
    print(f"Recherche Alpha dans : [{alpha_bounds[0]:.2f} - {alpha_bounds[1]:.2f}]\n")

    # --- C. CALCULS ---
    mu_est = estimate_growth_rate(xb_data, xd_data, age_data)
    print(f"Taux mu moyen : {mu_est:.4f}")
    
    # Référence (Silverman amélioré pour le point de départ)
    alpha_ref = 1.06 * std_xd * (N_total ** (-1/5))
    B_ref = estimate_B_size(xb_data, xd_data, grid_x, alpha_ref, method=method_choice)
    
    # Étude de convergence
    sample_sizes = np.array([50, 100, 200, 400, 800, 1000, 1202])
    optimal_alphas = []
    
    print("Optimisation des alphas pour différentes tailles n...")
    for n in sample_sizes:
        a_opt = find_optimal_alpha_size(n, xb_data, xd_data, grid_x, B_ref, alpha_bounds, method_choice)
        optimal_alphas.append(a_opt)
        print(f"  n = {n:4d} | alpha_opt = {a_opt:.4f}")
        
    log_eps = np.log(1.0 / sample_sizes)
    log_alp = np.log(optimal_alphas)
    slope_alpha, intercept, _, _, _ = linregress(log_eps, log_alp)
    
    # Analyse SAT
    slope_sat, int_sat, r_sat, sat_model = analyze_sat(xb_data, xd_data)

    # --- D. CALCUL DE LA DENSITÉ f(x) via KDE (ksdensity) ---
    # On utilise le même lissage (alpha_ref) pour être cohérent
    kde_func = gaussian_kde(xd_data, bw_method=alpha_ref/np.std(xd_data))
    f_x = kde_func(grid_x)

    # --- E. VISUALISATION MISE À JOUR ---
    plt.figure(figsize=(18, 6))
    
    # Plot 1: B(x) ET f(x) SUPERPOSÉS
    ax1 = plt.subplot(1, 3, 1)
    
    # Axe gauche : Taux de division B(x)
    color_b = 'purple'
    ax1.set_xlabel('Taille (x)')
    ax1.set_ylabel('Taux de division B(x)', color=color_b)
    ax1.plot(grid_x, B_ref, color=color_b, linewidth=3, label='Taux B(x) (Hazard)')
    ax1.tick_params(axis='y', labelcolor=color_b)
    ax1.grid(True, alpha=0.3)

    # Création du deuxième axe (droite) pour f(x)
    ax2 = ax1.twinx()  
    color_f = 'orange'
    ax2.set_ylabel('Densité de division f(x)', color=color_f)
    ax2.fill_between(grid_x, f_x, alpha=0.2, color=color_f) # Ombrage pour la densité
    ax2.plot(grid_x, f_x, color=color_f, linestyle='--', linewidth=2, label='Densité f(x) (KDE)')
    ax2.tick_params(axis='y', labelcolor=color_f)

    plt.title("Superposition : Taux B(x) vs Densité f(x)")
    
    # Fusion des légendes des deux axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Plot 2: Convergence
    plt.subplot(1, 3, 2)
    plt.plot(log_eps, log_alp, 'ko-', label='Mesures')
    plt.plot(log_eps, slope_alpha * log_eps + intercept, 'r--', 
             label=f'Pente (Théorie 0.2) = {slope_alpha:.3f}')
    plt.title("Convergence de Alpha")
    plt.xlabel("log(1/n)")
    plt.ylabel("log(alpha_opt)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: SAT
    plt.subplot(1, 3, 3)
    plt.scatter(xb_data, xd_data - xb_data, alpha=0.15, color='teal', s=10)
    x_range = np.array([np.min(xb_data), np.max(xb_data)])
    plt.plot(x_range, slope_sat * x_range + int_sat, 'r-', linewidth=2, label=f'Pente = {slope_sat:.2f}')
    plt.title(f"{sat_model}\n(R² = {r_sat**2:.2f})")
    plt.xlabel("Taille naissance (xb)")
    plt.ylabel("Taille ajoutée (xd-xb)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()