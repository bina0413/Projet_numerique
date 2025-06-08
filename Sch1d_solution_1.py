import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

start_time = time.time()

def init():
    line.set_data([], [])
    return line,

def animate(j):
    line.set_data(o, final_densite[j,:]) #Crée un graphique pour chaque densite sauvegarde
    return line,

dt=1E-7
dx=0.001
nx=int(1/dx)*2
nt=90000 # En fonction du potentiel il faut modifier ce parametre car sur certaines animations la particule atteins les bords
nd=int(nt/1000)+1#nombre d image dans notre animation
n_frame = nd
s=dt/(dx**2)
xc=2
sigma=0.05
A=1/(math.sqrt(sigma*math.sqrt(math.pi)))
v0=-5
e=0.5 #Valeur du rapport E/V0
E=e*abs(v0)
k=math.sqrt(2*E)



o=np.zeros(nx)
V=np.zeros(nx)

# Initialisation des tableaux
o = np.linspace(-10, 3, nx)
#o = np.linspace(0, (nx - 1) * dx, nx)
V = np.zeros(nx)
V[(o >= 3) & (o<=5)] = v0  # Potentiel

cpt = A * np.exp(1j * k * o - ((o - xc) ** 2) / (2 * (sigma ** 2)))
densite=np.zeros((nt,nx))
densite[0,:] = np.absolute(cpt[:]) ** 2
final_densite=np.zeros((n_frame,nx))
re=np.zeros(nx)
re[:]=np.real(cpt[:])

b=np.zeros(nx)

im=np.zeros(nx)
im[:]=np.imag(cpt[:])

it=0
for i in range(1, nt):
    if i % 2 != 0:
        b[1:-1]=im[1:-1]
        im[1:-1] = im[1:-1] + s * (re[2:] + re[:-2]) - 2 * re[1:-1] * (s + V[1:-1] * dt)
        densite[i,1:-1] = re[1:-1]*re[1:-1] + im[1:-1]*b[1:-1]
    else:
        re[1:-1] = re[1:-1] - s * (im[2:] + im[:-2]) + 2 * im[1:-1] * (s + V[1:-1] * dt)

for i in range(1,nt):
    if((i-1)%1000==0):
        it+=1
        final_densite[it][:]=densite[i][:]



fig = plt.figure() # initialise la figure principale
line, = plt.plot([], [])
plt.ylim(-15,3.5)
plt.xlim(-10,10)
plt.plot(o,V,label="Potentiel")
# Fais apparaitre sur tous les graphiques le potentiel
#En changeant V par ((V*6)/v0) ca permet de faire rentrer le potentiel dans le graphique
plt.title("Marche Ascendante avec E/Vo=1")
plt.xlabel("x")
plt.ylabel("Densité de probabilité de présence")
#plt.legend() #Permet de faire apparaitre la legende

frames = 200
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=nd, blit=True, interval=50, repeat=False)
# # #ani = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=True)

ani.save('min2.mp4', writer = animation.FFMpegWriter(fps=30, bitrate=5000))


#file_name = 'animation_paquet_onde2.mp4'
#ani.save(file_name, writer=animation.FFMpegWriter(fps=60, bitrate=5000))
#print(f"Animation sauvegardée dans {file_name}")


# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Elapsed Time: {elapsed_time} seconds")

