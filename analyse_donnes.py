"""
IMPORT DES BIBLIOTHEQUES
"""

import numpy as np
import matplotlib.pyplot as plt

"""
PARAMETRES
"""

N=10 # nombre de particules
R=1 # diamètre particule (unité naturelle)
epsilon = 1 # énergie (unité naturelle)
kb = 1 # constante de Boltzmann (unité naturelle)
m=1 # masse des particules (unité naturelle)

dt=0.01 # intervalle de temps (unités naturelles)
Nb_iterations=200


"""
BOITE 2D : bords fixes (2 possibilités de conditions aux limites : périodique ou réflexion sur la surface)
"""

# 2D : boite de 10 par 10

x_min=0
x_max=10
y_min=0
y_max=10
Lx=x_max - x_min
Ly=y_max - y_min
dx=0.01
dy=0.01

"""
EXTRACTION DES DONNES (POSITIONS VITESSES ACCELERATIONS)
"""

# lecture des données et mise en forme 
data=np.loadtxt("calcul_r_v_a_analytique.txt",skiprows=1, delimiter=",", dtype=float)
data_array = data.reshape(Nb_iterations + 1, N, 7) # même forme de tableau que celui issu de la fonction calcul_positions_vitesses_accelerations : pour chaque temps et pour chaque particule : [temps, X, Y, Vx, Vy, Ax, Ay]

#print(data_array)

# extraction de temps / positions / vitesses / accélérations

time_array=data_array[:,:,0]
time=time_array[:,0]
r=data_array[:,:,1:3]
V=data_array[:,:,3:5]
a=data_array[:,:,5:7]

"""
PLOT DE LA TRAJECTOIRE DES PARTICULES AU COURS DU TEMPS
"""

plt.figure()
for j in range(N): # tracé de la trajectoire pour chaque particule (boucle sur les particules)
    plt.plot(r[:, j, 0], r[:, j, 1],'o' ,linestyle=':',label=f'Particule {j}')
    plt.scatter(r[0,j,0],r[0,j,1],color='black',s=150,label="positions initiales" if j==N-1 else None) # ajout d'un point noir pour indiquer la position initiale
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Trajectoires des particules')
plt.legend()
plt.grid()
plt.xlim(x_min,x_max)
plt.ylim(y_min,y_max)
plt.savefig("trajectoires particules.png")
plt.show()

"""
CALCUL DE L'ENERGIE DU SYSTEME
"""

V_x=data_array[:,:,3] # vitesses selon x
V_y=data_array[:,:,4] # vitesses selon y


def energie_cinetique_totale():
    Ec=[] 
    for t in range(Nb_iterations) : # boucle sur le temps
        Ec_t=0 
        for i in range(N): # boucle sur le nombre de particules
            V_carre=V_x[t][i]**2 + V_y[t][i]**2 # vitesse au carré de la particule i au temps t
            Ec_t+=0.5*m*V_carre
        Ec.append(Ec_t)
    return np.array(Ec)

def Lennard_Jones(r):
    u=4*epsilon*((R/r)**12 - (R/r)**6)
    return u

r_lim=100 # portée maximale de l'interraction (à ajuster)

def energie_potentielle_par_particule(r): # r correspond au n-uplet contenant toutes les positions des particules / on tient compte des CL périodiques
    U_tot=np.zeros((N,N)) # initialisation de la matrice d'interraction
    for i in range(N): # boucle sur le nombre de particules
        for j in range(N): 
            if i!=j : # on ne calcule pas le potentiel d'interaction entre une particule et elle même, U_tot aura tous ses coeffs diagonaux nuls
                # distance réelle entre la particule i et la particule j
                delta_x=r[i,0]-r[j,0]
                delta_y=r[i,1]-r[j,1]
                # distance avec les conditions périodiques entre la particule i et la particule j
                distances_possibles_x=np.array(np.abs([delta_x -Lx, delta_x, delta_x +Lx])) # on ne considère que les boites directement adjacentes à notre boite
                distances_possibles_y=np.array(np.abs([delta_y -Ly, delta_y, delta_y +Ly]))
                d_x_périodique=np.min(distances_possibles_x, axis=0) # distance selon x périodique (voir brouillon pour le calcul)
                d_y_periodique=np.min(distances_possibles_y,axis=0) # distance selon y périodique
                d_i_j=np.sqrt(d_x_périodique**2+d_y_periodique**2) # distance entre la particule i et la particules j en tenant compte des CL périodiques
                if d_i_j < r_lim : # on limite la portée de l'interaction
                    u_i_j=Lennard_Jones(d_i_j)
                    U_tot[i][j]=u_i_j
    return U_tot

def energie_potentielle_totale(r):
    Ep=[]
    for t in range(Nb_iterations):
        r_t=r[t]
        matrice_energies_potentielles=energie_potentielle_par_particule(r_t)
        Ep.append(np.sum(matrice_energies_potentielles)/2) # l'énergie potentielle totale est la somme des énergies potentielles d'interaction entre chaque particule / 2 (on divise par 2 pour ne pas compter 2 fois la même interaction entre i et j et j et i)
    return np.array(Ep)

def energie_totale(r):
    return energie_cinetique_totale() + energie_potentielle_totale(r) # return un tableau de longueur Nb_iterations correspondant à l'énergie totale au temps t

energie=energie_totale(r)

#print("energie=",energie)

plt.figure()
plt.plot(time[:-1], np.log(energie))
plt.title("log de l'énergie totale du système en fonction du temps")
plt.xlabel("temps")
plt.ylabel("log(E)")
plt.grid()
plt.show()