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

# plt.figure()
# for j in range(N): # tracé de la trajectoire pour chaque particule (boucle sur les particules)
#     plt.plot(r[:, j, 0], r[:, j, 1],'o' ,linestyle=':',label=f'Particule {j}')
#     #plt.plot(r[:, j, 0], r[:, j, 1],'o',label=f'Particule {j}')
#     plt.scatter(r[0,j,0],r[0,j,1],color='black',s=150,label="positions initiales" if j==N-1 else None) # ajout d'un point noir pour indiquer la position initiale
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Trajectoires des particules')
# plt.legend()
# plt.grid()
# plt.xlim(x_min,x_max)
# plt.ylim(y_min,y_max)
# plt.savefig("trajectoires particules.png")
# plt.show()


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

# calcul des énergies
energie=energie_totale(r)
Ec = energie_cinetique_totale()
Ep = energie_potentielle_totale(r)

# # plot log de l'énergie totale
# plt.figure()
# plt.plot(time[:-1], np.log(energie))
# plt.title("log de l'énergie totale du système en fonction du temps")
# plt.xlabel("temps")
# plt.ylabel("log(E)")
# plt.grid()
# plt.savefig("log_energie_au_cours_du_temps.png")
# plt.show()


# plot des énergies
plt.figure()
plt.plot(time[:len(Ec)],Ec,label="énergie cinétique")
plt.plot(time[:len(Ep)],Ep,label="énergie potentielle")
plt.plot(time[:len(energie)],energie, label="energie totale")
plt.xlabel("temps")
plt.ylabel("energie")
plt.legend()
plt.grid()
plt.savefig("energie_au_cours_du_temps.png")
plt.show()

# sauvegarde des énergies
data_Ec=np.column_stack((time[:len(Ec)],Ec))
data_Ep=np.column_stack((time[:len(Ep)],Ep))
data_E=np.column_stack((time[:len(energie)],energie))

np.savetxt("Energie_cinetique.txt",data_Ec,delimiter=",",header="Temps,Ec")
np.savetxt("Energie_potentielle.txt",data_Ep,delimiter=",",header="Temps,Ep")
np.savetxt("Energie_totale.txt",data_E,delimiter=",",header="Temps,E")

"""
DETERMINATION DE L'EQUILIBRE DU SYSTEME
"""

def equilibre(Ec,critere_Ec,critere_fenetre): 
    # un système est à l'équilibre lorsque ses grandeurs macroscopiques ne dépendent plus du temps, or V,N sont fixés, on veut voir quand T devient constante
    # T est définie à l'équilibre via l'énergie cinétique moyenne via le théorème d'équipartition de l'énergie, on veut voir quand ets-ce que Ec et constante
    fenetre=int(Nb_iterations/critere_fenetre)
    moy_Ec_fenetre=[]
    for i in range(fenetre//2,len(Ec)-fenetre//2):
        moy=np.mean(Ec[i-fenetre//2:i+fenetre//2])
        moy_Ec_fenetre.append(moy)
    for j in range(len(moy_Ec_fenetre)-1):
        if np.abs(moy_Ec_fenetre[j]-moy_Ec_fenetre[j+1])<critere_Ec:
            debut_equilibre=j+fenetre//2
            temperature_equilibre=2*moy_Ec_fenetre[j]/(d*N*kb)
            return debut_equilibre, temperature_equilibre 
    return None, None # au cas où aucun équilibre est atteint

# paramètres pour l'équilibre

critere_Ec=0.1*(10**4)
critere_fenetre=Nb_iterations/100

eq=equilibre(Ec, critere_Ec, critere_fenetre)
if eq[0]!= None and eq[1]!= None : # si aucun équilibre est détecté alors début_equilibre et T_eq ne seront pas définis et python enverra un message d'erreur
    debut_equilibre=eq[0]
    T_eq=eq[1]

print("debut_equilibre=",debut_equilibre)
print("T_eq=",T_eq)

"""
DISTRIBUTION DES VITESSES A L'EQUILIBRE ET COMPARAISON AVEC LA DISTRIBUTION DE MAXWELL BOLTZMANN
"""

def distribution_Maxwell_Boltzmann_direction_i(Vi): # Vi : vitesse selon la direction i
    return np.sqrt(m/(2*np.pi*kb*T_eq))*np.exp(-m*(Vi**2)/(2*kb*T_eq))

def distribution_Malwell_Boltzmann_norme_vitesse_2d(V):
    return (m*V/(kb*T_eq))*np.exp(-m*(V**2)/(2*kb*T_eq))

# distributions théoriques
Vx_min_eq=np.min(V_x[debut_equilibre])
Vx_max_eq=np.max(V_x[debut_equilibre])
Vx_th_eq=np.arange(Vx_min_eq,Vx_max_eq,(Vx_max_eq-Vx_min_eq)/100)
distr_Vx_th_eq=distribution_Maxwell_Boltzmann_direction_i(Vx_th_eq)

Vy_min_eq=np.min(V_y[debut_equilibre])
Vy_max_eq=np.max(V_y[debut_equilibre])
Vy_th_eq=np.arange(Vy_min_eq,Vy_max_eq,(Vy_max_eq-Vy_min_eq)/100)
distr_Vy_th_eq=distribution_Maxwell_Boltzmann_direction_i(Vy_th_eq)

V_eq=np.sqrt(V_x[debut_equilibre]**2 + V_y[debut_equilibre]**2)
V_th=np.arange(0, np.max(V_eq), np.max(V_eq)/100)
distr_V_th_eq=distribution_Malwell_Boltzmann_norme_vitesse_2d(V_th)

# plot
plt.figure()
plt.hist(V_x[debut_equilibre],bins=20,density=True, label="simulation") # normalisation avec density=True afin de pouvoir comparer la densité de proba théorique avec l'histogramme issu de la simulation
plt.plot(Vx_th_eq,distr_Vx_th_eq,label="théorie")
plt.title("Distribution des vitesses à l'équilibre selon x")
plt.xlabel("vitesse")
plt.ylabel("nombre de particules")
plt.legend()
plt.savefig("distribution_Vx_equilibre.png")
plt.show()

plt.figure()
plt.hist(V_y[debut_equilibre],bins=20,density=True,label="simulation")
plt.plot(Vy_th_eq,distr_Vy_th_eq,label="théorie")
plt.title("Distribution des vitesses à l'équilibre selon y")
plt.xlabel("vitesse")
plt.ylabel("nombre de particules")
plt.legend()
plt.savefig("distribution_Vy_equilibre.png")
plt.show()

plt.figure()
plt.hist(V[debut_equilibre],bins=20,density=True,label="simulation")
plt.plot(V_th,distr_V_th_eq,label="théorie")
plt.title("Distribution des normes de vitesse à l'équilibre")
plt.xlabel("vitesse")
plt.ylabel("nombre de particules")
plt.legend()
plt.savefig("distribution_V_equilibre.png")
plt.show()