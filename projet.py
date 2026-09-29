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
    r=np.column_stack((x, y))
    return r

T_init=300 # température initiale (en K)

def distribution_gaussienne_vitesses(N):
    mu_init=np.sqrt(3*kb*T_init/m) # correspond à la vitesse quadratique moyenne (d'après la thermo, permet d'avoir un ordre d'idée)
    sigma_init=1 # à réfléchir
    Vx=np.random.normal(mu_init,sigma_init,size=N)
    Vy=np.random.normal(mu_init,sigma_init,size=N)
    V=np.column_stack((Vx,Vy))
    return V

r_init = distribution_uniforme_positions(N) # positions initiales
V_init = distribution_gaussienne_vitesses(N) # vitesses initiales
a_init = np.zeros_like(r_init) # accélérations initiales (je ne sais pas)

# plot des positions initiales dans la boite
x_init=r_init[:,0]
y_init=r_init[:,1]

plt.figure()
plt.scatter(x_init,y_init,color='red',s=50)
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(x_min,x_max)
plt.ylim(y_min,y_max)
#plt.axis('scaled')
plt.title("positions initiales")
plt.show()

# histogramme des vitesses initiales
Vx_init=V_init[:,0]
Vy_init=V_init[:,1]
V_init_norm=np.sqrt(Vx_init**2+Vy_init**2)

plt.figure()
plt.hist(V_init_norm,bins=15)
plt.title("Distribution de la norme des vitesses initiales")
plt.show()

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

# def calcul_positions_vitesses(Nb_iterations,r_init,V_init,a_init):
    
#     # conditions initiales de position et vitesse
#     r=r_init
#     V=V_init
#     a=a_init
    
#     # iteration :
#     for i in range(Nb_iterations):

#         # initialisation des nouvelles positions, vitesses et accélérations
#         r_new=np.zeros_like(r)
#         V_new=np.zeros_like(V)
#         a_new=np.zeros_like(a)

#         # calcul des nouvelles positions et vitesses
#         r_new[i]=r[i]+V[i]*dt+0.5*a[i]*(dt**2)
#         V_new[i]=V[i]+0.5*a[i]*dt

#         # calcul des nouvelles accélérations
#         U= potentiel_total(r)
#         U_i=0
#         a_new[i]=-np.grad(U_i))/m

#         # calcul des nouvelles vitesses
#         V_new[i]=V[i]+0.5*a_new[i]*dt


"""
CALCUL DU POTENTIEL D'INTERRACTION
"""

r_lim=100 # portée maximale de l'interraction

def potentiel_total_par_particule(r): # r correspond au n-uplet contenant toutes les positions des particules
    U_tot=np.zeros((N,N)) # initialisation de la matrice d'interraction
    for i in range(N):
        for j in range(N):
            if i!=j : # on ne calcule pas le potentiel d'interaction entre une particule et elle même, U_tot aura tous ses coeffs diagonaux nuls
                d_i_j=np.sqrt((r[i,0]-r[j,0])**2+(r[i,1]-r[j,1])**2) # distance entre la particule i et la particules j
                if d_i_j < r_lim : # on limite la portée de l'interaction
                    u_i_j=Lennard_Jones(d_i_j)
                    U_tot[i][j]=u_i_j
    return U_tot

# test en partant des conditions initiales (on vérifie que la matrice d'interactions est bien symétrique)
potentiel_initial=potentiel_total_par_particule(r_init)

plt.figure()
plt.imshow(potentiel_initial)
plt.colorbar()
plt.xlabel("particule i")
plt.ylabel("particule j")
plt.title("potentiel d'interaction entre particule i et particule j")
plt.show()

# def champ(r): # but : calculer le champs en tout point
#     x=np.arange(x_min,x_max,dx)
#     y=np.arange(y_min,y_max,dy)
#     U_tot=np.zeros((len(x),len(y))) #initialisation du champs en chaque point de l'espace
#     for i in range(len(x)): # boucles sur les points de l'espace
#         for j in range(len(y)):
#             u_M_k=0 # initialisation de la valeur du champ au point M
#             for k in range(len(r)) : # boucle sur le nombre de particules
#                 d_M=np.sqrt((x[i]-r[k,0])**2+(y[j]-r[k,1])) # distance entre le point M de l'espace et la particule k 
#                 if d_M>0.0001 : # si la particule se situe au point M, on ne tient pas compte du champ qu'elle créé au point M (le champ serait infini)
#                     u_M_k+=Lennard_Jones(d_M)
#             U_tot[i][j]=u_M_k # le champs au point M est la somme des champs créée par toutes les particules au point M
#     return U_tot


def champ(r): # but : calculer le champs en tout point pour pouvoir en calculer son gradient
    x=np.arange(x_min,x_max,dx)
    y=np.arange(y_min,y_max,dy)
    X,Y=np.meshgrid((x,y)) # discrétisation de l'espace
    U_tot=np.zeros((len(x),len(y))) # initialisation du champs en chaque point de l'espace
    u_M_k=0 # initialisation de la valeur du champ au point M
    for k in range(len(r)) : # boucle sur le nombre de particules
        d_M=np.sqrt((X-r[k,0])**2+(Y-r[k,1])**2) # distance entre le point M de l'espace et la particule k 
        if d_M>0.0001 : # si la particule se situe au point M, on ne tient pas compte du champ qu'elle créé au point M (le champ serait infini)
            u_M_k+=Lennard_Jones(d_M)
        U_tot[][]=u_M_k # le champs au point M est la somme des champs créée par toutes les particules au point M
    return U_tot

# teste en partant des conditions initiales
champ_initial=champ(r_init)

plt.figure()
plt.imshow(potentiel_initial)
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.title("U(x,y)")
plt.show()

def gradient_champ(U,dx,dy):
    grad=np.zeros_like(U) # initialisation
    