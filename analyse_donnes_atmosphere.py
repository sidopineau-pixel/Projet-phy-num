"""
IMPORT DES BIBLIOTHEQUES
"""

import numpy as np
import matplotlib.pyplot as plt


"""
PARAMETRES
"""

N=500 # nombre de particules
R=1 # diamètre particule (unité naturelle)
epsilon = 1 # énergie (unité naturelle)
kb = 1 # constante de Boltzmann (unité naturelle)
m=1 # masse des particules (unité naturelle)
dt=0.00001 # intervalle de temps (unités naturelles)
Nb_iterations=1000

r_lim=10000 # portée maximale de l'interraction (à ajuster)


"""
BOITE 2D : bords fixes (2 possibilités de conditions aux limites : périodique ou réflexion sur la surface)
"""

# 2D : boite de 10 par 10

x_min=0
x_max=111
y_min=0
y_max=111
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
CALCUL DE LA DENSITE DE PARTICULES EN FONCTION DE L'ALTITUDE
"""

