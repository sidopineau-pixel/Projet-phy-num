"""
CODE POUR LE CALCUL DES POSITIONS VITESSES ET ACCELERATIONS
"""


"""
IMPORT DES BIBLIOTHEQUES
"""

import numpy as np
import matplotlib.pyplot as plt

"""
PARAMETRES
"""

N=3 # nombre de particules
R=1 # diamètre particule (unité naturelle)
epsilon = 1 # énergie (unité naturelle)
kb = 1 # constante de Boltzmann (unité naturelle)
m=1 # masse des particules (unité naturelle)

# temps :
dt=0.01 # intervalle de temps (unités naturelles)
Nb_iterations=10 # nombre d'itérations dans le calcul de positions, vitesses, accélérations

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

# # plot des positions initiales dans la boite
# x_init=r_init[:,0]
# y_init=r_init[:,1]

# plt.figure()
# plt.scatter(x_init,y_init,color='red',s=50)
# plt.xlabel("x")
# plt.ylabel("y")
# plt.xlim(x_min,x_max)
# plt.ylim(y_min,y_max)
# plt.axis('scaled')
# plt.title("positions initiales")
# plt.savefig("positions_initiales_100_particules.png")
# plt.show()

# # histogramme des vitesses initiales
# Vx_init=V_init[:,0]
# Vy_init=V_init[:,1]
# V_init_norm=np.sqrt(Vx_init**2+Vy_init**2)

# plt.figure()
# plt.hist(V_init_norm,bins=15)
# plt.title("Distribution de la norme des vitesses initiales")
# plt.show()

"""
POTENTIEL DE LENNARD-JONES
"""

def Lennard_Jones(r):
    u=4*epsilon*((R/r)**12 - (R/r)**6)
    return u

# r_Lennard_Jones=np.arange(0,50,0.01)

# plt.figure()
# plt.plot(r_Lennard_Jones,Lennard_Jones(r_Lennard_Jones))
# plt.xlabel("r")
# plt.ylabel("u(r)")
# plt.xlim(0,10)
# plt.ylim(-1.2,1)
# plt.title("Potentiel de Lennard-Jones")
# plt.show()

"""
ALGORITHME DE VERLET POUR LE CALCUL DES VITESSES
"""


def calcul_positions_vitesses_accelerations(Nb_iterations,r_init,V_init,a_init):

    # initialisation du tableau de donné (méthode vectorisée bien plus rapide, l'autre était beaucoup trop lente)
    data=np.zeros((Nb_iterations + 1,N,7),dtype=float) # à chaque particule on associe 7 données : [temps, X, Y, Vx, Vy, Ax, Ay]

    # conditions initiales
    data[0,:,0]=0
    data[0,:,1:3]=r_init
    data[0,:,3:5]=V_init
    data[0,:,5:7]=a_init

    r=np.array(r_init,dtype=float)
    V=np.array(V_init,dtype=float)
    a=np.array(a_init,dtype=float)

    # itération :
    for i in range(Nb_iterations) : # boucle sur le temps

        # calcul des nouvelles positions avec conditions aux limites périodiques
        r_new=r+V*dt+0.5*a*(dt**2)
        r_new[:,0]=r_new[:,0]%x_max # CL périodiques
        r_new[:,1]=r_new[:,1]%y_max

        # calcul des nouvelles vitesses
        V_new=V+0.5*a*dt

        # calcul du champ d'accélérations
        U=champ(r_new) 
        grad_x, grad_y=gradient_2d(U,dx,dy) 
        accel_x=-grad_x/m
        accel_y=-grad_y/m
        champ_accel=np.column_stack((accel_x, accel_y)) # champs d'accélération calculé en tout point 

        # calcul de l'accélération des particules : si une particule est présente en un point (x,y) de l'espace alors on ajoute l'accélération calculée au point(x,y) au vecteur a_new / problème : on ne peut pas faire de nouveau une boucle sur chaque point de l'espace car cela augmente encore la complexité
        j_indice = np.round(r_new[:, 0] / dx).astype(int)% accel_x.shape[1] # x=j*dx donc j=x/dx avec j indice de colonne de la matrice du champ des accélérations / on arrondit au plus proche et convertit en int / si l'indice le plus proche correspond au dernier et sort du tableau : CL périodique : revient de l'autre côté
        i_indice = np.round(r_new[:, 1] / dy).astype(int)% accel_x.shape[0] # y=i*dy donc i=y/dy avec i indice de ligne de la matrice du champ des accélérations
        ax_new=accel_x[i_indice,j_indice]
        ay_new=accel_y[i_indice,j_indice]
        a_new=np.column_stack((ax_new,ay_new))
        
        # calcul de la nouvelle vitesse
        V_new=V_new+0.5*a_new*dt

        # stockage des données
        data[i+1,:,0]=dt*(i+1)
        data[i+1,:,1:3]=r_new
        data[i+1,:,3:5]=V_new
        data[i+1,:,5:7]=a_new

        # réinitialisation des positions, vitesses et accélérations pour le tour suivant
        r=r_new
        V=V_new
        a=a_new

    return data


"""
CALCUL DU POTENTIEL D'INTERRACTION
"""

r_lim=100 # portée maximale de l'interraction (à ajuster)

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

# # test en partant des conditions initiales (on vérifie que la matrice d'interactions est bien symétrique)
# potentiel_initial=potentiel_total_par_particule(r_init)

# plt.figure()
# plt.imshow(np.log(potentiel_initial))
# plt.colorbar()
# plt.xlabel("particule i")
# plt.ylabel("particule j")
# plt.title("log du potentiel d'interaction entre particule i et particule j")
# plt.show()


# cette fonction a une trop grande complexité son temps d'exécussion pour rien que 3 particules est déjà trop long

# def champ(r): # but : calculer le champs en tout point pour pouvoir en calculer son gradient
#     x = np.arange(x_min, x_max, dx)
#     y = np.arange(y_min, y_max, dy)
#     X, Y = np.meshgrid(x, y) # discrétisation de l'espace
#     U_tot = np.zeros((len(y), len(x))) # initialisation du champs en chaque point de l'espace
#     for i in range(len(y)): # boucle sur la position
#         for j in range(len(x)):
#             u_M_k = 0 # initialisation de la valeur du champ au point M
#             for k in range(len(r)):  # boucle sur le nombre de particules
#                 d_M = np.sqrt((X[i,j] - r[k,0])**2 + (Y[i,j] - r[k,1])**2) # distance entre le point M de l'espace et la particule k 
#                 if d_M > 0.0001: # si la particule se situe au point M, on ne tient pas compte du champ qu'elle créé au point M (le champ serait infini) / de plus la particule n'interragit pas avec elle-même
#                     u_M_k += Lennard_Jones(d_M)
#             U_tot[i,j] = u_M_k # le champs au point M est la somme des champs créée par toutes les particules au point M
#     return U_tot


# def champ(r): # but : calculer le champs en tout point pour pouvoir en calculer son gradient (cette fonction ne tient pas compte des CL périodiques)
#     x = np.arange(x_min, x_max, dx)
#     y = np.arange(y_min, y_max, dy)
#     X, Y = np.meshgrid(x, y) # Discrétisation de l'espace
#     U_tot = np.zeros_like(X, dtype=float) # Initialisation du champ en chaque point de l'espace
#     for k in range(len(r)): # Boucle sur les particules
#         d_M = np.sqrt((X - r[k, 0])**2 + (Y - r[k, 1])**2) # Distance entre chaque point M de l'espace et la particule k
#         masque = d_M > 0.0001 #si la particule se situe au point M, on ne tient pas compte du champ qu'elle créé au point M (le champ serait infini) / de plus la particule n'interragit pas avec elle-même
#         U_tot[masque] += Lennard_Jones(d_M[masque]) # le champs au point M est la somme des champs créée par toutes les particules au point M
#     return U_tot 



def champ(r): # but : calculer le champs en tout point pour pouvoir en calculer son gradient (cette fonction tient compte des CL périodiques)
    x = np.arange(x_min, x_max, dx)
    y = np.arange(y_min, y_max, dy)
    X, Y = np.meshgrid(x, y) # Discrétisation de l'espace
    U_tot = np.zeros_like(X, dtype=float) # Initialisation du champ en chaque point de l'espace
    for k in range(len(r)): # Boucle sur les particules
        delta_x=X - r[k, 0]
        delta_y=Y - r[k, 1]
        distances_possibles_x=np.array(np.abs([delta_x -Lx, delta_x, delta_x +Lx])) # on ne considère que les boites directement adjacentes à notre boite
        distances_possibles_y=np.array(np.abs([delta_y -Ly, delta_y, delta_y +Ly]))
        d_x_périodique=np.min(distances_possibles_x, axis=0) # distance selon x périodique (voir brouillon pour le calcul)
        d_y_periodique=np.min(distances_possibles_y,axis=0) # distance selon y périodique
        d_M = np.sqrt(d_x_périodique**2 + d_y_periodique**2) # Distance entre chaque point M de l'espace et la particule k
        masque = d_M > 0.0001 #si la particule se situe au point M, on ne tient pas compte du champ qu'elle créé au point M (le champ serait infini) / de plus la particule n'interragit pas avec elle-même
        U_tot[masque] += Lennard_Jones(d_M[masque]) # le champs au point M est la somme des champs créée par toutes les particules au point M
    return U_tot 


# teste en partant des conditions initiales
champ_initial=champ(r_init)

# # calcul des max et min pour ajuster les échelles de couleurs dans imshow
# print("U_init max=",np.max(champ_initial))
# print("U_init min=", np.min(champ_initial))

# plt.figure()
# #plt.imshow(champ_initial)
# plt.imshow(np.log(champ_initial), extent=[x_min, x_max, y_min, y_max], origin="lower")
# plt.colorbar()
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("log du champ initial U(x,y)")
# plt.savefig("champ_initial_100_particules.png")
# plt.show()

"""
CALCUL DU GRADIENT DU POTENTIEL
"""

def gradient_2d(f,dx,dy): # cf TP1

    # initialisation
    n,m=np.shape(f)
    grad_x=np.zeros((n,m)) 
    grad_y=np.zeros((n,m))

    # calcul des gradients à l'intérieur
    grad_x[:,1:-1]=(f[:,2:]-f[:,:-2])/(2*dx)
    grad_y[1:-1,:]=(f[2:,:]-f[:-2,:])/(2*dy)

    # # calcul des gradients sur les bords : mais ne tient pas des conditions aux limites périodiques... 
    # grad_x[:,0]=(f[:,1]-f[:,0])/(2*dx)
    # grad_x[:,-1]=(f[:,-1]-f[:,-2])/(2*dx)
    # grad_y[0,:]=(f[1,:]-f[0,:])/(2*dy)
    # grad_y[-1,:]=(f[-1,:]-f[-2,:])/(2*dy)

    # calcul des gradients sur les bords : en tenant compte des conditions aux limites périodiques
    grad_x[:,0]=(f[:,1]-f[:,-1])/(2*dx)
    grad_y[0,:]=(f[1,:]-f[-1,:])/(2*dy)
    grad_x[:,m-1]=(f[:,0]-f[:,m-2])/(2*dx)
    grad_y[n-1,:]=(f[0,:]-f[n-2,:])/(2*dy)

    return grad_x, grad_y
    
# test en partant des conditions initiales

# grad_champ_initial_selon_x=gradient_2d(champ_initial,dx,dy)[0]
# grad_champ_initial_selon_y=gradient_2d(champ_initial,dx,dy)[1]

# # calcul des max et min pour ajuster les échelles de couleurs dans imshow
# print("x max:", np.max(grad_champ_initial_selon_x))
# print("x min :", np.min(grad_champ_initial_selon_x))
# print("y max", np.max(grad_champ_initial_selon_y))
# print("y min", np.min(grad_champ_initial_selon_y))

# plt.figure()
# #plt.imshow(champ_initial)
# plt.imshow(np.log(grad_champ_initial_selon_x), extent=[x_min, x_max, y_min, y_max], origin="lower")
# plt.colorbar()
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("log(gradient) du champ initial selon x")
# plt.show()

# plt.figure()
# #plt.imshow(champ_initial)
# plt.imshow(np.log(grad_champ_initial_selon_y), extent=[x_min, x_max, y_min, y_max], origin="lower")
# plt.colorbar()
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("log(gradient) du champ initial selon y")
# plt.show()

"""
SAUVEGARDE DES DONNES DE POSITION, VITESSE, ACCELERATION
"""


test_dynamique_syst_2D=calcul_positions_vitesses_accelerations(Nb_iterations,r_init,V_init,a_init)
#print(test_dynamique_syst_2D)
test_dynamique_syst_2D = test_dynamique_syst_2D.reshape(-1, 7) # afin de pouvoir convertir le fichier de données au format txt je change le format du tableau : on aura sept colonnes comprennant : temps, x, y, V_x, V_y, a_x, a_y (le -1 indique que python calcule automatiquement le nombre de lignes necessaires)

np.savetxt("test_dynamique_syst_2D.txt",test_dynamique_syst_2D, delimiter=",",header="Temps,X,Y,Vx,Vy,Ax,Ay")
