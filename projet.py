"""
IMPORT DES BIBLIOTHEQUES
"""


import numpy as np
import matplotlib.pyplot as plt

"""
PARAMETRES
"""

N=2 # nombre de particules
R=1 # diamètre particule (unité naturelle)
epsilon = 1 # énergie (unité naturelle)
kb = 1 # constante de Boltzmann (unité naturelle)
m=1 # masse des particules (unité naturelle)


"""
CONDITIONS AUX LIMITES : bords fixes (2 possibilités : périodique ou réflexion sur la surface)
"""

# 2D : boite de 100 par 100

x_min=0
x_max=100
y_min=0
y_max=100
dx=0.01
dy=0.01

"""
CONDITIONS INITIALES
"""

def distribution_uniforme_positions(N): # on veut tirer la position initiale des N particules selon une probabilité uniforme
    x=np.random.uniform(x_min,x_max,size=N)
    y=np.random.uniform(y_min,y_max,size=N)
    r=np.dstack((x,y))
    return r

T_init=300 # température initiale (en K)

def distribution_gaussienne_vitesses(N):
    mu_init=np.sqrt(3*kb*T_init/m) # correspond à la vitesse quadratique moyenne (d'après la thermo, permet d'avoir un ordre d'idée)
    sigma_init=1 # à réfléchir
    Vx=np.random.normal(mu_init,sigma_init,size=N)
    Vy=np.random.normal(mu_init,sigma_init,size=N)
    V=np.dstack((Vx,Vy))
    return V

r_init = distribution_uniforme_positions(N)
V_init = distribution_gaussienne_vitesses(N)
a_init = np.zeros_like(r_init) # je ne sais pas

print("r_init",r_init)
print("V_init",V_init)

"""
POTENTIEL DE LENNARD-JONES
"""

def Lennard_Jones(r):
    u=4*epsilon*((R/r)**12 - (R/r)**6)
    return u

r_Lennard_Jones=np.arange(0,50,0.01)

plt.figure()
plt.plot(r_Lennard_Jones,Lennard_Jones(r_Lennard_Jones))
plt.xlabel("r")
plt.ylabel("u(r)")
plt.xlim(0,10)
plt.ylim(-1.2,1)
plt.title("Potentiel de Lennard-Jones")
plt.show()

"""
ALGORITHME DE VERLET POUR LE CALCUL DES VITESSES
"""

# temps :
dt=0.01 # intervalle de temps (unités naturelles)
Nb_iterations=100

def calcul_positions_vitesses(Nb_iterations,r_init,V_init,a_init):
    
    # conditions initiales de position et vitesse
    r=r_init
    V=V_init
    a=a_init
    
    # iteration :
    for i in range(Nb_iterations):

        # initialisation des nouvelles positions, vitesses et accélérations
        r_new=np.zeros_like(r)
        V_new=np.zeros_like(V)
        a_new=np.zeros_like(a)

        # calcul des nouvelles positions et vitesses
        r_new[i]=r[i]+V[i]*dt+0.5*a[i]*(dt**2)
        V_new[i]=V[i]+0.5*a[i]*dt

        # calcul des nouvelles accélérations
        U= # résultat de la matrice d'interaction 
        a_new[i]=-np.grad(U))/m

        # calcul des nouvelles vitesses
        V_new[i]=V[i]+0.5*a_new[i]*dt


"""
CALCUL DU POTENTIEL D'INTERRACTION
"""

def potentiel_total(u,r): # u correspond au potentiel généré par une particule, r correspond au n-uplet contenant toutes les positions des particules
    U_tot=np.zeros((N,N)) # initialisation de la matrice d'interraction
# il faut calculer la distance entre chaque particule, si elle est plus grande que r_lim on dit que le potentiel d'interaction entre les 2 particules est nul