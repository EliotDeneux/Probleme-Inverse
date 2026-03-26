import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from scipy.integrate import cumulative_trapezoid
import warnings

warnings.filterwarnings('ignore') # Pour éviter les avertissements de division par zéro en fin de grille

# =====================================================================
# 1. DÉFINITION DES NOYAUX (Modularité)
# =====================================================================
# Chaque fonction de noyau prend un tableau z = (a - A_i) / alpha
# et renvoie les valeurs du noyau.

def kernel_gaussian(z):
    """Noyau gaussien standard (ordre 2, lisse à l'infini)."""
    return (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * z**2)

def kernel_epanechnikov(z):
    """Noyau d'Epanechnikov (optimal en termes de variance, support compact)."""
    return np.where(np.abs(z) <= 1, 0.75 * (1 - z**2), 0.0)

# Dictionnaire pour appeler facilement les noyaux
KERNELS = {
    'gaussian': kernel_gaussian,
    'epanechnikov': kernel_epanechnikov
}

# =====================================================================
# 2. ESTIMATEUR PAR NOYAU (KDE) ET TAUX DE DIVISION B(a)
# =====================================================================

def estimate_kde(data, grid, alpha, kernel_name='gaussian'):
    """
    Estime la densité f(a) et la fonction de répartition F(a).
    
    Paramètres:
    - data : échantillon des âges à la division (A_i)
    - grid : points où l'on évalue la fonction (les âges 'a')
    - alpha : paramètre de régularisation (fenêtre / bandwidth)
    - kernel_name : choix du noyau
    """
    kernel_func = KERNELS[kernel_name]
    n = len(data)
    
    # Calcul vectorisé efficace : z_ij = (a_i - A_j) / alpha
    # grid[:, None] crée une colonne, data[None, :] crée une ligne
    z = (grid[:, None] - data[None, :]) / alpha
    
    # f_n,alpha(a) = (1 / n*alpha) * sum( K(z) )
    f_est = (1.0 / (n * alpha)) * np.sum(kernel_func(z), axis=1)
    
    # F_n,alpha(a) = intégrale cumulée de f_est
    # On utilise l'intégration numérique sur la grille
    da = grid[1] - grid[0]
    F_est = cumulative_trapezoid(f_est, initial=0, dx=da)
    
    # Sécurité pour s'assurer que F_est ne dépasse pas 1
    F_est = np.clip(F_est, 0, 1)
    
    return f_est, F_est

def compute_division_rate(f_est, F_est):
    """
    Calcule le taux de division B(a) = f(a) / (1 - F(a)).
    """
    # On ajoute une petite constante (1e-10) au dénominateur pour éviter 
    # la division par zéro aux âges avancés où F(a) tend vers 1.
    S_est = 1.0 - F_est + 1e-10 
    B_est = f_est / S_est
    return B_est

# =====================================================================
# 3. FONCTIONS D'OPTIMISATION (Trouver alpha optimal)
# =====================================================================

def expected_l2_error(alpha, n, full_data, grid, f_ref, num_trials, kernel_name, da):
    """
    Calcule l'erreur quadratique moyenne E[ || f_est - f_ref ||^2 ]
    en moyennant sur plusieurs sous-échantillons de taille n.
    """
    total_error = 0.0
    for _ in range(num_trials):
        # Tirage d'un sous-échantillon aléatoire de taille n
        subsample = np.random.choice(full_data, size=n, replace=False)
        
        # Estimation de la densité
        f_est, _ = estimate_kde(subsample, grid, alpha, kernel_name)
        
        # Erreur L2 numérique : intégrale de (f_est - f_ref)^2
        l2_err = np.sum((f_est - f_ref)**2) * da
        total_error += l2_err
        
    return total_error / num_trials

def find_optimal_alpha(n, full_data, grid, f_ref, num_trials=10, kernel_name='gaussian'):
    """Trouve le alpha qui minimise l'erreur L2 moyenne pour une taille n."""
    da = grid[1] - grid[0]
    
    # Fonction objectif à minimiser en fonction de alpha
    objective = lambda alpha: expected_l2_error(alpha, n, full_data, grid, f_ref, num_trials, kernel_name, da)
    
    # Optimisation scalaire (on cherche alpha entre 0.01 et 5.0 pour des âges)
    res = minimize_scalar(objective, bounds=(0.01, 5.0), method='bounded')
    return res.x

# =====================================================================
# 4. SCRIPT PRINCIPAL
# =====================================================================

def main():
    # --- A. CHARGEMENT DES DONNÉES ---
    # En pratique, utilise cette ligne :
    data_matrix = np.loadtxt('data/Eric1002_MDJ_sb_sd_ad.txt', delimiter=',')
    ages = data_matrix[:, 2] # 3ème colonne
    N_total = len(ages)
    print(f"Nombre total de cellules (N) : {N_total}")

    # # Pour la démonstration, je génère un jeu de données simulant des âges à la division
    # # (Mélange de deux gaussiennes pour simuler une distribution biologique)
    # np.random.seed(42)
    # N_total = 10000
    # ages = np.concatenate([
    #     np.random.normal(22, 2, int(N_total*0.7)),
    #     np.random.normal(26, 3, int(N_total*0.3))
    # ])
    
    # Paramètres de l'étude
    kernel_choice = 'gaussian' # Tu peux changer en 'epanechnikov'
    grid = np.linspace(10, 40, 500) # Grille d'âges de 10 à 40
    da = grid[1] - grid[0]
    
    # --- B. CRÉATION DE LA RÉFÉRENCE (Silverman's Rule of Thumb) ---
    print("Calcul de la densité de référence f_ref...")
    std_data = np.std(ages)
    alpha_ref = 1.06 * std_data * (N_total ** (-1/5)) # Règle de Silverman
    f_ref, F_ref = estimate_kde(ages, grid, alpha_ref, kernel_choice)
    B_ref = compute_division_rate(f_ref, F_ref)
    
    # --- C. BOUCLE SUR LES TAILLES D'ÉCHANTILLONS (n) ---
    sample_sizes = np.array([30, 50, 75, 100, 200, 300, 500, 750, 1000])
    epsilons = 1.0 / sample_sizes
    optimal_alphas = []
    
    print("Recherche des alpha optimaux...")
    for n in sample_sizes:
        # On calcule le alpha optimal pour ce n
        a_opt = find_optimal_alpha(n, ages, grid, f_ref, num_trials=15, kernel_name=kernel_choice)
        optimal_alphas.append(a_opt)
        print(f"n = {n:4d} (eps = {1/n:.5f}) -> alpha_opt = {a_opt:.4f}")
        
    optimal_alphas = np.array(optimal_alphas)
    
    # --- D. CALCUL DE LA PENTE (Ordre de convergence) ---
    # On fait une régression linéaire sur y = a*x + b
    log_eps = np.log(epsilons)
    log_alpha = np.log(optimal_alphas)
    slope, intercept = np.polyfit(log_eps, log_alpha, 1)
    
    # --- E. TRACÉS ET VISUALISATION ---
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Densité f(a) et Taux de division B(a) de référence
    plt.subplot(1, 2, 1)
    plt.plot(grid, f_ref, label='Densité f(a)', color='blue', linewidth=2)
    
    # On coupe l'affichage de B(a) à la fin de la grille car l'estimation 
    # explose naturellement quand F(a) -> 1 (plus de cellules survivantes)
    valid_idx = grid < 35 
    plt.plot(grid[valid_idx], B_ref[valid_idx], label='Taux de div. B(a)', color='red', linestyle='--')
    
    plt.title("Référence théorique (Grand N)")
    plt.xlabel("Âge (a)")
    plt.ylabel("Densité / Taux")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: log(alpha) en fonction de log(epsilon)
    plt.subplot(1, 2, 2)
    plt.plot(log_eps, log_alpha, 'ko-', label='Mesures expérimentales')
    plt.plot(log_eps, slope * log_eps + intercept, 'r--', 
             label=f'Régression linéaire\nPente (ordre) = {slope:.3f}')
    
    plt.title(r"Vitesse de convergence du paramètre de lissage : $\log(\alpha_{opt})$ vs $\log(\epsilon)$")
    plt.xlabel(r"$\log(\epsilon)$ avec $\epsilon = 1/n$")
    plt.ylabel(r"$\log(\alpha_{opt})$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()