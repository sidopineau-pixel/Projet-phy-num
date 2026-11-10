"""
IMPORT DES BIBLIOTHEQUES
"""

import numpy as np
import matplotlib.pyplot as plt


"""
PARAMETRES
"""

N=100 # nombre de particules
R=1 # diamètre particule (unité naturelle)
epsilon = 1 # énergie (unité naturelle)
kb = 1 # constante de Boltzmann (unité naturelle)
m=1 # masse des particules (unité naturelle)
dt=0.00001 # intervalle de temps (unités naturelles)
Nb_iterations=1000

r_lim=10000 # portée maximale de l'interraction (à ajuster)

g=0.02 # accélération de la pesanteur


"""
BOITE 2D : bords fixes (2 possibilités de conditions aux limites : périodique ou réflexion sur la surface)
"""

# 2D : boite de 10 par 10

x_min=0
x_max=44
y_min=0
y_max=44
Lx=x_max - x_min
Ly=y_max - y_min
dx=0.01
dy=0.01

d=2 # dimension de la simulation (ici : 2D)

A=Lx*Ly
rho=N/A # densité de particules

"""
EXTRACTION DES DONNES (POSITIONS VITESSES ACCELERATIONS)
"""

# lecture des données et mise en forme 
data=np.loadtxt("calcul_r_v_a_analytique_atmosphere.txt",skiprows=1, delimiter=",", dtype=float)
data_array = data.reshape(Nb_iterations + 1, N, 7) # même forme de tableau que celui issu de la fonction calcul_positions_vitesses_accelerations : pour chaque temps et pour chaque particule : [temps, X, Y, Vx, Vy, Ax, Ay]

#print(data_array)

# extraction de temps / positions / vitesses / accélérations

time_array=data_array[:,:,0]
time=time_array[:,0]
r=data_array[:,:,1:3]
V=data_array[:,:,3:5]
a=data_array[:,:,5:7]

"""
ANIMATION TRAJECTOIRE DES PARTICULES (code de cette partie fait par Chat GPT)
"""

from matplotlib.animation import FuncAnimation

# Création de la figure
fig, ax = plt.subplots(figsize=(8, 8))

# Définition des limites de la boîte
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Mouvement des particules au cours du temps")

ax.set_aspect('equal')
ax.grid()

# Initialisation des particules
scatter = ax.scatter(r[0, :, 0], r[0, :, 1],
                     s=50, color='blue')

# Affichage du temps
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

# Fonction d'initialisation
def init():
    scatter.set_offsets(r[0])
    time_text.set_text('')
    return scatter, time_text

# Fonction de mise à jour à chaque image
def update(frame):
    scatter.set_offsets(r[frame])
    time_text.set_text(f"Temps : {time[frame]:.5f}")
    return scatter, time_text

# Création de l'animation

ani = FuncAnimation(
    fig,
    update,
    frames=range(0, Nb_iterations + 1, 20),
    init_func=init,
    interval=1,
    blit=False,
    repeat=False
)

plt.show()


# Sauvegarde

ani.save(
    "animation_particules_atmosphere.gif",
    writer="pillow",
    fps=100
)


"""
DENSITE DE PARTICULES EN FONCTION DE L'ALTITUDE
"""

nb_tranches=10


def calcul_densite_altitude(r):
    y=r[:,1] # altitude
    
    epaisseur_tranche_y=Ly/nb_tranches
    Aire_tranche=epaisseur_tranche_y*Lx
    
    effectif_par_tranche, bords_tranches = np.histogram(y, bins=nb_tranches, range=(0, Ly))

    densite=effectif_par_tranche/Aire_tranche

    return bords_tranches, densite


"""
MODELE THEORIQUE DE L'ATMOSPHERE ISOTHERME
"""
def atmosphere_isotherme(y,T,rho_0):
    H_0=kb*T/(m*g)
    return rho_0*np.exp(-y/H_0)


"""
PLOT a différents instants t
"""

nb_de_plot=10
pas_plot=int(Nb_iterations/nb_de_plot)
temps_plot=[time[i] for i in range(0,Nb_iterations,pas_plot)]

plt.figure()
for t in temps_plot :
    indice_temps=np.where(time==t)[0][0]
    bords_tranches, densite = calcul_densite_altitude(r[indice_temps])
    plt.stairs(densite,bords_tranches,orientation="vertical", label=f't={t}')
plt.xlabel("altitude y")
plt.ylabel("densité de particules")
plt.legend()
plt.title("densité de particules à plusieurs instants")
plt.show()