'''
Implementation of belief propagation algorithm for inference on Ising models
'''

import numpy as np

def compute_unary_messages(i, j, unary_potentials, unary_messages):
    '''
    Compute messages for U, D, L, R neighbors in log space
    '''
    
    potential = unary_potentials[i, j]
    messages = unary_messages[i, j]
    
    # Log product
    u = potential * messages[1] * messages[2] * messages[3]
    d = potential * messages[0] * messages[2] * messages[3]
    l = potential * messages[0] * messages[1] * messages[3]
    r = potential * messages[0] * messages[1] * messages[2]
    
    # Normalize
    u /= np.sum(u)
    d /= np.sum(d)
    l /= np.sum(l)
    r /= np.sum(r)

    return u, d, l, r

def compute_pairwise_messages(i, j, pairwise, pairwise_messages):
    '''
    Compute messages for U, D or L, R neighbors in log space
    '''
    
    messages = pairwise_messages[i, j]
    
    # Log product
    ul = pairwise * messages[1]
    dr = pairwise * messages[0]

    # Sum
    ul = np.sum(ul, axis=0)
    dr = np.sum(dr, axis=0)
    
    # Normalize
    ul /= np.sum(ul)
    dr /= np.sum(dr)

    return ul, dr

def step(h, w, unary_potentials, pairwise_potential, unary_messages, col_pairwise_messages, row_pairwise_messages):
    '''
    Compute and send all messages in log space
    '''
    
    # Send messages from unary nodes
    for i in range(h):
        for j in range(w):
            u, d, l, r = compute_unary_messages(i, j, unary_potentials, unary_messages)

            if i > 0: col_pairwise_messages[i-1, j, 1] = u # Send message up
            if i < h-1: col_pairwise_messages[i, j, 0] = d # Send message down
            if j > 0: row_pairwise_messages[i, j-1, 1] = l # Send message left
            if j < w-1: row_pairwise_messages[i, j, 0] = r # Send Message right

    # Send messages from column pairwise nodes
    for i in range(h-1):
        for j in range(w):
            u, d = compute_pairwise_messages(i, j, pairwise_potential, col_pairwise_messages)

            unary_messages[i, j, 1] = u # Send up
            unary_messages[i+1, j, 0] = d # Send down

    # Send messages from row pairwise nodes
    for i in range(h):
        for j in range(w-1):
            l, r = compute_pairwise_messages(i, j, pairwise_potential, row_pairwise_messages)

            unary_messages[i, j, -1] = l # Send left
            unary_messages[i, j+1, -2] = r # Send right

def compute_unary_beliefs(h, w, unary_potentials, unary_messages):
    beliefs = np.empty((h, w, 2))

    for i in range(h):
        for j in range(w):
            prod = unary_potentials[i, j] * unary_messages[i, j, 0] * unary_messages[i, j, 1]
            beliefs[i, j] = prod

    return beliefs

def normalize_beliefs(beliefs):
    return (beliefs/np.expand_dims(np.sum(beliefs, axis=-1), axis=-1))[:, :, 1]