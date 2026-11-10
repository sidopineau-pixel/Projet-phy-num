"""
CODE POUR L'ETUDE DE LA DENSITE DE GAZ DANS UN CHAMP DE GRAVITATION
"""

"""
ce qui change par rapport au code calcul_r_v_a_anaytique.py : 
- conditions aux limites de réflexion : modifie le calcul du gradient et le calcul des positions
- ajout d'une force selon l'axe y 
"""


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

# temps :
dt=0.00001 # intervalle de temps (unités naturelles)
Nb_iterations=1000 # nombre d'itérations dans le calcul de positions, vitesses, accélérations

r_lim=10000 # portée maximale de l'interraction (à ajuster)
g=0.02 # accélération de la pesanteur

"""
BOITE 2D : bords fixes (2 possibilités de conditions aux limites : réflexion sur les parois (on ne peut pas mettre de CL périodique pour étudier la gravité...)
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

A=Lx*Ly
rho=N/A # densité de particules


"""
POTENTIEL DE LENNARD-JONES
"""

def Lennard_Jones(r1,r2): # r1,r2 = vecteurs positions des particules 1 et 2
    x1=r1[0]
    y1=r1[1]
    x2=r2[0]
    y2=r2[1]
    r_norm=np.sqrt((x1-x2)**2+(y1-y2)**2)
    return 4*epsilon*((R/r_norm)**12 - (R/r_norm)**6)

"""
GRADIENT DE L'ENERGIE POTENTIELLE (f=-grad(u)) calcul analytique : conditions périodiques que selon x / réflexion selon y
"""

def gradient_u(r1,r2): # gradient de l'énergie potentielle / r1, r2= vecteurs position / ne tient pas compte des conditions périodiques
    x1=r1[0]
    y1=r1[1]
    x2=r2[0]
    y2=r2[1]
    norm_r=np.sqrt((x2-x1)**2+(y2-y1)**2)
    norm_grad = 4*epsilon*(-12*(R**12)/(norm_r**13)+6*(R**6)/(norm_r**7)) # dérivée par rapport à r du potentiel de Lennard-Jones
    grad_x = norm_grad*(x2-x1)/norm_r
    grad_y= norm_grad*(y2-y1)/norm_r
    return np.array([grad_x, grad_y])

def gradient_u_CL_périodiques_x(r1, r2):

    # positions réelles dans la boîte
    x1=r1[0]
    y1=r1[1]
    x2=r2[0]
    y2=r2[1]

    delta_x=x1-x2
    delta_y=y1-y2

    # distances possibles avec périodicité uniquement selon x
    distances_possibles_x = np.abs([delta_x - Lx, delta_x,delta_x + Lx])

    # indice de la distance selon x min
    indice_x = np.argmin(distances_possibles_x)

    # distance selon x en tenant compte du signe
    distance_x_signe = np.array([delta_x - Lx, delta_x, delta_x + Lx])
    d_x_periodique = distance_x_signe[indice_x]

    # pas de périodicité selon y
    d_y=delta_y

    # calcul de la distance entre les particules
    norm_r = np.sqrt(d_x_periodique**2 + d_y**2)

    # calcul du gradient en tenant compte des CL périodiques que selon x
    norm_grad = 4*epsilon*(-12*(R**12)/(norm_r**13)+ 6*(R**6)/(norm_r**7))
    grad_x = norm_grad*d_x_periodique/norm_r
    grad_y = norm_grad*d_y/norm_r

    return np.array([grad_x, grad_y])


def somme_des_gradients(r): # r est le n-uplet contenant les positions de toutes les particules
    somme_grad=np.zeros((N,2))
    for i in range(N): # boucle sur toutes les particules i
        grad_ressenti_par_particule_i=np.zeros(2)
        for j in range(N): # boucle sur toutes les particules j!= i 
            if i!=j :
                grad_ressenti_par_particule_i += gradient_u_CL_périodiques_x(r[i],r[j])
                somme_grad[i][0]=grad_ressenti_par_particule_i[0]
                somme_grad[i][1]=grad_ressenti_par_particule_i[1]
    return somme_grad 

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
        
        # calcul des nouvelles vitesses avec condition de réflexion :
        V_new=V+0.5*a*dt

        # calcul de l'accélération en tenant compte de la gravité
        a_new[:,0]=(-1/m)*somme_des_gradients(r_new)[:,0]
        a_new[:,1]=(-1/m)*somme_des_gradients(r_new)[:,1]-g
        
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


def calcul_positions_vitesses_accelerations(Nb_iterations, r_init, V_init, a_init): # conditions périodiques que selon x / réflexion selon y

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

    # itération
    for i in range(Nb_iterations):

        # calcul des nouvelles positions :
        r_new=r+V*dt+0.5*a*(dt**2)
        
        # calcul des nouvelles vitesses :
        V_new=V+0.5*a*dt

        # conditions périodiques selon x :
        r_new[:,0]=r_new[:,0]%Lx # CL périodiques

        # conditions de réflexion selon y
        for j in range(N): # pour chaque particule
            if r_new[j,1] < y_min: # si la particule dépasse le bord inférieur
                r_new[j,1] = 2*y_min - r_new[j,1] # nouvelle position à l'intérieur de la boite
                V_new[j,1] = -V_new[j,1] # Vy change de signe

            elif r_new[j,1] > y_max: # si la particule dépasse le bord supérieur
                r_new[j,1] = 2*y_max - r_new[j,1] # nouvelle position à l'intérieur de la boite
                V_new[j,1] = -V_new[j,1] # Vy change de signe

        # calcul de la nouvelle accélération
        gradients = somme_des_gradients(r_new)

        a_new = np.zeros((N, 2))
        a_new[:,0]=-gradients[:,0]/m
        a_new[:,1]=-gradients[:,1]/m - g

        # calcul de la nouvelle vitesse
        V_new=V_new+0.5*a_new*dt

        # stockage des données
        data[i+1,:,0]=dt*(i+1)
        data[i+1,:,1:3]=r_new
        data[i+1,:,3:5]=V_new
        data[i+1,:,5:7]=a_new

        # 8. Réinitialisation
        r=r_new
        V=V_new
        a=a_new

    return data


"""
CONDITIONS INITIALES
"""


def distribution_uniforme_positions(N): # on veut tirer la position initiale des N particules selon une probabilité uniforme
    x=np.random.uniform(x_min,x_max,size=N)
    y=np.random.uniform(y_min,y_max,size=N)
    r=np.column_stack((x, y))
    return r


def distribution_gaussienne_vitesses(N):
    # mu_init=np.sqrt(3*kb*T_init/m) # correspond à la vitesse quadratique moyenne (d'après la thermo, permet d'avoir un ordre d'idée)
    mu_init=0 # vitesse moyenne nulle
    sigma_init=1 # à réfléchir
    Vx=np.random.normal(mu_init,sigma_init,size=N)
    Vy=np.random.normal(mu_init,sigma_init,size=N)
    V=np.column_stack((Vx,Vy))
    return V

#def temperature_initiale(r_init): # en phase gaz 


r_init = distribution_uniforme_positions(N) # positions initiales
V_init = distribution_gaussienne_vitesses(N) # vitesses initiales 
a_init=(-1/m)*somme_des_gradients(r_init) # accélération initiale 
a_init[:,1]+= -g  # accélération initiale selon y contient aussi l'accélération de la pesanteur


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
# plt.savefig("distribution_norme_vitesses_initiales.png")
# plt.show()

# plt.figure()
# plt.hist(Vx_init,bins=15)
# plt.title("Distribution des vitesses initiales selon x")
# plt.savefig("distribution_vitesses_initiales_selon_x.png")
# plt.show()

# plt.figure()
# plt.hist(Vy_init,bins=15)
# plt.title("Distribution des vitesses initiales selon y")
# plt.savefig("distribution_vitesses_initiales_selon_y.png")
# plt.show()


"""
SAUVEGARDE DES DONNES DE POSITION, VITESSE, ACCELERATION
"""


calcul_r_v_a_analytique_atmosphere=calcul_positions_vitesses_accelerations(Nb_iterations,r_init,V_init,a_init)
calcul_r_v_a_analytique_atmosphere = calcul_r_v_a_analytique_atmosphere.reshape(-1, 7) # afin de pouvoir convertir le fichier de données au format txt je change le format du tableau : on aura sept colonnes comprennant : temps, x, y, V_x, V_y, a_x, a_y (le -1 indique que python calcule automatiquement le nombre de lignes necessaires)

np.savetxt("calcul_r_v_a_analytique_atmosphere.txt",calcul_r_v_a_analytique_atmosphere, delimiter=",",header="Temps,X,Y,Vx,Vy,Ax,Ay")


