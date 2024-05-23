'''
Implementation of belief propagation algorithm for inference on Ising models
'''

import numpy as np

def compute_unary_messages(i, j, unary_potentials, unary_messages):
    '''
    Compute messages for U, D, L, R neighbors
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
    Compute messages for U, D or L, R neighbors
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
    Compute and send all messages
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
            
###########################################################################
###########################################################################
###########################################################################

def compute_unary_beliefs(unary_potentials, unary_messages):
    '''
    Compute normalized unary beliefs. 
    Belief is equal to product of node potential and messages
    '''
    h, w, *_ = unary_messages.shape
    beliefs = np.empty(
        (h, w, 2)
    )
    
    # Product
    for i in range(h):
        for j in range(w):
            prod = unary_potentials[i, j] * np.prod(unary_messages[i, j], axis=0)
            beliefs[i, j] = prod

    # Normalize
    beliefs /= np.expand_dims(np.sum(beliefs, axis=-1), axis=-1)
    
    return beliefs

def compute_pairwise_beliefs(pairwise_potential, pairwise_messages):
    '''
    Compute normalize pairwise beliefs. 
    Belief is equal to product of node potential and messages
    '''
    h, w, *_ = pairwise_messages.shape
    beliefs = np.empty(
        (h, w, 2, 2)
    )
    
    # Product
    for i in range(pairwise_messages.shape[0]):
        for j in range(w):
            prod = pairwise_potential * pairwise_messages[i, j, 0] * pairwise_messages[i, j, 1].T
            beliefs[i, j] = prod
            
    # Normalize
    beliefs /= np.expand_dims(np.sum(beliefs, axis=(2, 3)), axis=(-1, -2))
    
    return beliefs

def compute_beliefs(unary_potentials, pairwise_potential, unary_messages, col_pairwise_messages, row_pairwise_messages):
    '''
    Nice wrapper for computing all beliefs
    '''
    return (
        compute_unary_beliefs(unary_potentials, unary_messages), 
        compute_pairwise_beliefs(pairwise_potential, col_pairwise_messages), 
        compute_pairwise_beliefs(pairwise_potential, row_pairwise_messages)
    )