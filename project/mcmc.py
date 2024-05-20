'''
Implementation of MCMC algorithm for inference on Ising models
'''

import numpy as np

def compute_dE(i, j, lattice, J, B):
    '''
    Compute change in energy resulting from spin flip. 
    '''
    s = lattice[i, j-1] if j > 0 else 0
    s += lattice[i, j+1] if j < lattice.shape[1] - 1 else 0
    s += lattice[i-1, j] if i > 0 else 0
    s += lattice[i+1, j] if i < lattice.shape[0] - 1 else 0
    
    return 2 * lattice[i, j] * (J * s + B)

def step_one(lattice, J, B, beta):
    '''
    Randomly choose a spin and decide whether to flip. 
    '''
    i, j = np.random.randint(low=(0, 0), high=lattice.shape)

    dE = compute_dE(i, j, lattice, J, B)

    if dE < 0 or np.random.rand() < np.exp(-beta * dE): 
        lattice[i, j] *= -1
        
def step_all(lattice, J, B, beta):
    '''
    Randomly iterate through all spins and decide whether to flip. 
    '''
    spins = [(i, j) for i in range(lattice.shape[0]) for j in range(lattice.shape[1])]
    np.random.shuffle(spins)
    
    for i, j in spins:
        dE = compute_dE(i, j, lattice, J, B)

        if dE < 0 or np.random.rand() < np.exp(-beta * dE): 
            lattice[i, j] *= -1