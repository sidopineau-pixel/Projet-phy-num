"""
IMPORT DES BIBLIOTHEQUES
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

"""
PARAMETRES
"""

N=150 # nombre de particules
R=1 # diamètre particule (unité naturelle)
epsilon = 1 # énergie (unité naturelle)
kb = 1 # constante de Boltzmann (unité naturelle)
m=1 # masse des particules (unité naturelle)
dt=0.001 # intervalle de temps (unités naturelles)
Nb_iterations=10000
nb_cascades=16
Nb_iterations_total= nb_cascades*Nb_iterations +1 # nombre d'iterations mises bout à bout
r_lim=10000 # portée maximale de l'interraction (à ajuster)


"""
BOITE 2D : bords fixes (2 possibilités de conditions aux limites : périodique ou réflexion sur la surface)
"""

# 2D : boite de 10 par 10

x_min=0
x_max=200
y_min=0
y_max=200
Lx=x_max - x_min
Ly=y_max - y_min
dx=5
dy=5

d=2 # dimension de la simulation (ici : 2D)

A=Lx*Ly
rho=N/A # densité de particules

"""
EXTRACTION DES DONNES (POSITIONS VITESSES ACCELERATIONS) pour une simulation en cascade
"""

dossier_base = Path(r"C:\Documents\magistère\M1\physique numérique\projet\distributions de Maxwell-Bolztmann\150 particules E conservée")

liste_data = []

for i in range(1, nb_cascades+1):
    chemin = dossier_base / f"simulation cascade {i}" / "calcul_r_v_a_analytique.txt"
    data = np.loadtxt(chemin, skiprows=1, delimiter=",", dtype=float)
    data_array = data.reshape(Nb_iterations + 1, N, 7)

    if i > 1:
        data_array = data_array[1:] # l'état initial de la cascade suivante = l'état final de la cascade précedente
        
    liste_data.append(data_array)

data_total=np.concatenate(liste_data, axis=0) # on met les cascades bout à bout sur l'axe du temps

time=np.arange(Nb_iterations_total)*dt
r=data_total[:,:,1:3]
V=data_total[:,:,3:5]
a=data_total[:,:,5:7]

# """
# EXTRACTION DES DONNES (POSITIONS VITESSES ACCELERATIONS) sans cascade
# """

# # lecture des données et mise en forme 
# data=np.loadtxt("calcul_r_v_a_analytique.txt",skiprows=1, delimiter=",", dtype=float)
# data_array = data.reshape(Nb_iterations + 1, N, 7) # même forme de tableau que celui issu de la fonction calcul_positions_vitesses_accelerations : pour chaque temps et pour chaque particule : [temps, X, Y, Vx, Vy, Ax, Ay]

# # extraction de temps / positions / vitesses / accélérations

# time_array=data_array[:,:,0]
# time=time_array[:,0]
# r=data_array[:,:,1:3]
# V=data_array[:,:,3:5]
# a=data_array[:,:,5:7]

"""
ERGODICITE DANS L'ESPACE DES POSITIONS
"""
particule_i=random.randrange(0,N) # tirage d'une particule  
t_eq=Nb_iterations_total -1
t_debut_eq=int(Nb_iterations_total/2)

# # une seule particule au cours du temps 

# r_i=r[:,particule_i]
# x_i=r_i[:,0]
# y_i=r_i[:,1]

# x_bins=np.arange(x_min,x_max+dx,dx)
# y_bins=np.arange(y_min,y_max+dy,dy)
# histogramme_positions_1_particule = np.histogram2d(x_i, y_i, bins=[x_bins, y_bins]) [0]

# plt.figure()
# plt.imshow(histogramme_positions_1_particule.T, origin="lower",extent=[x_min, x_max, y_min, y_max],cmap="hot")
# plt.colorbar(label="temps passé dans le carré")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.savefig("histogramme_positions_1_particule.png")
# plt.show()

# # N particules à un instant t

# r_t=r[t_eq,:,:]
# x_t=r_t[:,0]
# y_t=r_t[:,1]

# histogramme_positions_N_particules = np.histogram2d(x_t, y_t, bins=[x_bins, y_bins]) [0]

# plt.figure()
# plt.imshow(histogramme_positions_N_particules.T, origin="lower",extent=[x_min, x_max, y_min, y_max],cmap="hot")
# plt.colorbar(label="nombre de particules dans le carré")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.savefig("histogramme_positions_N_particules.png")
# plt.show()

"""
ERGODICITE DANS l'ESPACE DES VITESSES
"""

# suivi d'une seule particule au cours du temps

V_i=V[t_debut_eq:,particule_i]
Vx_i=V_i[:,0]
Vy_i=V_i[:,1]
norm_V_i=np.sqrt(Vx_i**2 + Vy_i**2)

# N particules à un temps t

# V_t=V[t_eq,:,:]
# Vx_t=V_t[:,0]
# Vy_t=V_t[:,1]
# norm_V_t=np.sqrt(Vx_t**2 + Vy_t**2)

# plt.figure()
# plt.hist(norm_V_i, bins=20,density="True",label="1 particule dans le temps")
# plt.hist(norm_V_t, bins=20,density="True",label="N particules à tf")
# plt.legend()
# plt.xlabel("norme de la vitesse")
# plt.savefig("ergodicite_vitesses.png")
# plt.show()

# vitesse de la particule en fonction du temps

# plt.figure()
# plt.plot(time[t_debut_eq:],norm_V_i)
# plt.xlabel("t")
# plt.ylabel("Vitesse")
# plt.title("Vitesse particule i au cours du temps")
# plt.show()

"""
LE SYSTEME PERD T-IL LA MEMOIRE ?
"""

V=data_total[:,:,3:5]

def fonction_autocorrelation(V):
    Cv=np.zeros(Nb_iterations_total)
    V0=V[0,:,:] # vitesse initiale
    V0_carre=V0[:,0]**2 +V0[:,1]**2
    moy_V0_carre=np.mean(V0_carre)
    for t in range(Nb_iterations_total):
        V_t=V[t,:,:]
        produit_scalaire=np.sum(V_t*V0,axis=1)
        Cv[t]=np.mean(produit_scalaire)/moy_V0_carre
    return Cv

Cv=fonction_autocorrelation(V)
tc=np.argmin(np.abs(Cv-1/np.e)) # temps de corrélation
print(tc)

plt.figure()
plt.plot(time,Cv)
plt.xlabel("temps")
plt.ylabel("autocorrélation")
plt.savefig("autocorrelation_ergodicite.png")
plt.show()
