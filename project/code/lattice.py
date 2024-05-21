'''
Lattice Implementations and Functions for Ising Models
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Patch

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

def gaussian_B(h, w):
    '''
    A nice magnetic field for testing inference algorithms. 
    '''
    B = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            B[i, j] = np.exp(-(i-h/2)**2 / (4*h) - (j-w/2)**2 / (4*w))

    return B


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
    n = lattice.shape[0] * lattice.shape[1]
    
    S = 0
    for i in range(lattice.shape[0]):
        for j in range(lattice.shape[1]):
            s = lattice[i, j+1] if j < lattice.shape[1] - 1 else 0
            s += lattice[i+1, j] if i < lattice.shape[0] - 1 else 0
            
            S += lattice[i, j] * s
            
    return -J * S - np.sum(B * lattice) / n


def render_timesteps(timesteps, fps, interval, fname):
    '''
    Render an array of timesteps. 
    '''
    
    fig, ax = plt.subplots()
    plt.xticks([])
    plt.yticks([])
    
    fig.set_tight_layout(True)
    
    labels = [
        Patch(facecolor='white', edgecolor='black', label='Spin down'),
        Patch(facecolor='black', edgecolor='black', label='Spin up')
    ]

    plt.legend(handles=labels, fancybox=True, loc='center left', bbox_to_anchor=(1, 0.5))


    img = ax.imshow([[0]], 'binary', vmin=0, vmax=1)

    def init():
        img.set_data([[0]])
        return img

    def render(i):
        img.set_data(timesteps[i])
        return img

    anim = FuncAnimation(fig, render, init_func=init, frames=len(timesteps), interval=interval)
    
    anim.save(fname, writer='ffmpeg', fps=fps)