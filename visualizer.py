#!/usr/bin/env python3
"""
Visualisation des densités (KDE) pour les simulations de division cellulaire.
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Résolution dynamique du chemin (pour éviter de chercher à la racine du disque C:)
DATA_DIR = Path(__file__).parent / "data"
DB_PATH = DATA_DIR / "cells.db"

verbose = False  # Print des informations pour visualiser les premières lignes de la base de données

def main():
    if not DB_PATH.exists():
        print(f"Erreur : La base de données est introuvable au chemin {DB_PATH}.")
        print("As-tu bien corrigé le chemin dans le script de simulation et relancé la génération ?")
        return

    print("Chargement des données depuis SQLite...")
    # On charge uniquement les colonnes nécessaires pour économiser de la RAM
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT model, rate, division_age, division_size FROM cells", con)
    con.close()

    if verbose :

        # ── Exploration de la structure ──────────────────────────────────────────
        print(f"\n{'─'*40}")
        print(f" Structure de la base de données ")
        print(f"{'─'*40}")
        
        # Nombre d'entrées (lignes, colonnes)
        n_rows, n_cols = df.shape
        print(f"Nombre total de cellules simulées : {n_rows:,}")
        print(f"Nombre de colonnes par cellule    : {n_cols}")
        print(f"Colonnes disponibles              : {list(df.columns)}")
        
        print("\nAperçu des 5 premières lignes :")
        print(df.head()) # Affiche les 5 premières lignes
        
        # Petit comptage par modèle pour vérifier l'équilibre
        print("\nRépartition des simulations :")
        print(df.groupby(['model', 'rate']).size().unstack(fill_value=0))
        print(f"{'─'*40}\n")

    # Configuration de l'esthétique des graphiques
    sns.set_theme(style="whitegrid", palette="muted")
    
    # Récupération de la liste des modèles (age, size, increment)
    models = df['model'].unique()

    # Création d'une grille de graphiques : 1 ligne par modèle, 2 colonnes (Âge et Taille)
    fig, axes = plt.subplots(nrows=len(models), ncols=2, figsize=(14, 4 * len(models)))
    fig.suptitle("Densités de l'âge et de la taille à la division par modèle (KDE)", fontsize=16, y=0.98)

    for i, model in enumerate(models):
        subset = df[df['model'] == model]

        # ── Colonne de Gauche : Âge à la division ─────────────────────────────
        ax_age = axes[i, 0]
        sns.kdeplot(
            data=subset, x="division_age", hue="rate", fill=True, 
            common_norm=False, alpha=0.4, linewidth=2, ax=ax_age
        )
        ax_age.set_title(f"Modèle {model.upper()} : Densité de l'Âge à la division", fontweight="bold")
        ax_age.set_xlabel("Âge à la division [min]")
        ax_age.set_ylabel("Densité")

        # ── Colonne de Droite : Taille à la division ──────────────────────────
        ax_size = axes[i, 1]
        sns.kdeplot(
            data=subset, x="division_size", hue="rate", fill=True, 
            common_norm=False, alpha=0.4, linewidth=2, ax=ax_size
        )
        ax_size.set_title(f"Modèle {model.upper()} : Densité de la Taille à la division", fontweight="bold")
        ax_size.set_xlabel("Taille à la division [µm]")
        ax_size.set_ylabel("Densité")

    # Ajustement automatique des marges pour éviter que les textes ne se chevauchent
    plt.tight_layout()
    plt.subplots_adjust(top=0.92) # Laisse un peu de place pour le suptitle
    
    print("Affichage des graphiques...")
    plt.show()

if __name__ == "__main__":
    main()