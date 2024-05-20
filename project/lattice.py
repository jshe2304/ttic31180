'''
Lattice Implementations and Functions for Ising Models
'''

import numpy as np

def create_spin_lattice(h, w):
    '''
    Create a binary fermion spin lattice
    '''
    return np.random.choice(a=(-1, 1), size=(h, w))

def create_belief_lattice(h, w):
    '''
    Create a lattice distribution over binary fermion spin up probabilities. 
    '''
    return np.tile([-1, 1], (h, w, 1))

def magnetization(lattice):
    '''
    Compute bormalized magnetization
    '''
    n = lattice.shape[0] * lattice.shape[1]
    return np.sum(lattice)/n

def energy(lattice, J, B):
    '''
    Compute Ising model Hamiltonian
    '''
    S = 0
    for i in range(lattice.shape[0]):
        for j in range(lattice.shape[1]):
            s = lattice[i, j+1] if j < lattice.shape[1] - 1 else 0
            s += lattice[i+1, j] if i < lattice.shape[0] - 1 else 0
            
            S += lattice[i, j] * s
            
    return -J * S - B * np.sum(lattice)