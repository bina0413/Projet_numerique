# calculer le coefficient de transmission
import numpy as np
import matplotlib.pyplot as plt
import math
import time

# Parametres globaux
dt = 1E-7
dx = 0.001
nx = int(1 / dx) * 2
nt = 90000
s = dt / dx ** 2
xc = 2
sigma = 0.05
A = 1 / (math.sqrt(sigma * math.sqrt(math.pi)))
v0 = -15

# Grille spatiale
o = np.linspace(-10, 10, nx)
V = np.zeros(nx)
V[(o >= 3) & (o <= 5)] = v0

def transmission_coeff(e):
    E = e * abs(v0)
    k = math.sqrt(2 * E)
    cpt = A * np.exp(1j * k * o - ((o - xc) ** 2) / (2 * sigma ** 2))

    # Partie reelle et imaginaire
    re = np.real(cpt)
    im = np.imag(cpt)

    densite = np.zeros((nt, nx))
    densite[0, :] = np.abs(cpt) ** 2

    b = np.zeros(nx)

    for i in range(1, nt):
        if i % 2 != 0:
            b[1:-1] = im[1:-1]
            im[1:-1] += s * (re[2:] + re[:-2] - 2 * re[1:-1]) - V[1:-1] * re[1:-1] * dt * 2
            densite[i, 1:-1] = re[1:-1] ** 2 + im[1:-1] * b[1:-1]
        else:
            re[1:-1] -= s * (im[2:] + im[:-2] - 2 * im[1:-1]) - V[1:-1] * im[1:-1] * dt * 2

    # Probabilites
    x0 = 2.5  # Avant barriere
    x1 = 5.5  # Apres barriere
    P_initial = np.sum(np.abs(cpt[o < x0]) ** 2) * dx
    P_transmise = np.sum(densite[-1, o > x1]) * dx
    return P_transmise / P_initial

# Boucle sur plusieurs valeurs de E/|V0|
e_values = np.linspace(0.5, 3.5, 30)
T_values = []

start_time = time.time()
for e in e_values:
    T = transmission_coeff(e)
    print(f"E/|V₀| = {e:.2f}, T = {T:.3f}")
    T_values.append(T)

# Tracer T(E)
plt.figure(figsize=(8, 5))
plt.plot(e_values, T_values, 'o-', label="Transmission numérique")
plt.xlabel("E / |V₀|")
plt.ylabel("Coefficient de transmission T")
plt.title("Effet Ramsauer–Townsend : Transmission vs Énergie")
plt.grid()
plt.legend()
plt.savefig("transmission_vs_energy.png")


#end_time = time.time()
#print(f"Temps total: {end_time - start_time:.2f} s")
