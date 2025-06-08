# trouver les états stationnaires

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# Paramètres
N = 1000              # Nombre de points
L = 10.0              # Taille de la boîte (doit contenir le puits + zones "tunnel")
x = np.linspace(-L/2, L/2, N)
dx = x[1] - x[0]

a = 1.0               # Demi-largeur du puits
V0 = 50.0             # Hauteur à l'extérieur

# Potentiel fini
V = np.where(np.abs(x) < a, 0, V0)

# Matrice cinétique (opérateur Laplacien)
diag = np.ones(N) * -2
off_diag = np.ones(N - 1)
T = (np.diag(diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)) / (dx**2)

# Hamiltonien H = -1/2 * T + diag(V)
H = -0.5 * T + np.diag(V)

# Diagonalisation (on prend les 5 premiers états propres)
energies, states = eigh(H, subset_by_index=(0, 4))

# Normalisation des fonctions d’onde
states_normalized = states / np.sqrt(dx)

# Affichage
plt.figure(figsize=(10, 6))
for n in range(5):
    plt.plot(x, states_normalized[:, n] + energies[n], label=f'n={n}')
plt.plot(x, V / np.max(V) * np.max(energies), 'k--', label='V(x)')
plt.title("États stationnaires dans un puits de potentiel fini")
plt.xlabel("x")
plt.ylabel("Énergie / Fonction d'onde")
plt.legend()
plt.grid()
plt.savefig("etats_stationnaires.png")
