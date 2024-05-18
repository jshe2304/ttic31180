'''
Lattice Implementations and Functions for Ising Models
'''

import numpy as np

create_lattice = lambda h, w : np.random.choice(a=(-1, 1), size=(h, w))

def magnetization(lattice):
    '''
    Compute Normalized Magnetization
    '''
    n = lattice.shape[0] * lattice.shape[1]
    return np.sum(lattice)/n

def energy(lattice, J, B):
    '''
    Compute Ising Model Hamiltonian
    '''
    S = 0
    for i in range(lattice.shape[0]):
        for j in range(lattice.shape[1]):
            s = lattice[i, j+1] if j < lattice.shape[1] - 1 else 0
            s += lattice[i+1, j] if i < lattice.shape[0] - 1 else 0
            
            S += lattice[i, j] * s
            
    return -J * S - B * np.sum(lattice)