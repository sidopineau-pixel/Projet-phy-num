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

"""
BUT : avoir une joli figure du potentiel de Lennard-Jones
"""


def Lennard_Jones_trace(r): 
    u=4*epsilon*((R/r)**12 - (R/r)**6)
    return u

r_Lennard_Jones_trace=np.arange(0,50,0.01)

plt.figure()
plt.plot(r_Lennard_Jones_trace,Lennard_Jones_trace(r_Lennard_Jones_trace))
plt.xlabel("r")
plt.ylabel("u(r)")
plt.xlim(0,10)
plt.ylim(-1.2,1)
plt.title("Potentiel de Lennard-Jones")
plt.savefig("potentiel_Lennard_Jones.png")
plt.show()