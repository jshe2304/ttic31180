'''
Implementation of belief propagation algorithm for inference on Ising models
'''

import numpy as np

def compute_unary_log_messages(i, j, unary, unary_log_messages):
    '''
    Compute messages for U, D, L, R neighbors in log space
    '''
    messages = unary_log_messages[i, j]

    # Log product
    u = unary + np.sum(messages[(1, 2, 3), :], axis=0)
    d = unary + np.sum(messages[(0, 2, 3), :], axis=0)
    l = unary + np.sum(messages[(0, 1, 3), :], axis=0)
    r = unary + np.sum(messages[(0, 1, 2), :], axis=0)

    return u, d, l, r

def compute_pairwise_log_messages(i, j, pairwise, pairwise_log_messages):
    '''
    Compute messages for U, D or L, R neighbors in log space
    '''
    # Log product
    ul = pairwise + pairwise_log_messages[i, j, 0]
    dr = pairwise + pairwise_log_messages[i, j, 1]
    
    # Sum
    ul = np.sum(np.exp(ul), axis=0)
    dr = np.sum(np.exp(dr), axis=0)
    
    return np.log(ul), np.log(dr)

def step(h, w, unary_potential, pairwise_potential, unary_log_messages, col_pairwise_log_messages, row_pairwise_log_messages):
    '''
    Compute and send all messages in log space
    '''
    
    # Send messages from unary nodes
    for i in range(h):
        for j in range(w):
            u, d, l, r = compute_unary_log_messages(i, j, unary_potential, unary_log_messages)

            if i > 0: col_pairwise_log_messages[i-1, j, 1] = u # Send message up
            if i < h-1: col_pairwise_log_messages[i, j, 0] = d # Send message down
            if j > 0: row_pairwise_log_messages[i, j-1, 1] = l # Send message left
            if j < w-1: row_pairwise_log_messages[i, j, 0] = r # Send Message right

    # Send messages from column pairwise nodes
    for i in range(h-1):
        for j in range(w):
            u, d = compute_pairwise_log_messages(i, j, pairwise_potential, col_pairwise_log_messages)

            unary_log_messages[i, j, 1] = u # Send up
            unary_log_messages[i+1, j, 0] = d # Send down

    # Send messages from row pairwise nodes
    for i in range(h):
        for j in range(w-1):
            l, r = compute_pairwise_log_messages(i, j, pairwise_potential, row_pairwise_log_messages)

            unary_log_messages[i, j-1, 3] = u # Send left
            unary_log_messages[i, j, 2] = d # Send right























def compute_unary_log_messages(i, j, unary, unary_log_messages):
    '''
    Compute messages for U, D, L, R neighbors in log space
    '''
    u = unary + np.sum(unary_log_messages[i, j, (1, 2, 3)], axis=0)
    d = unary + np.sum(unary_log_messages[i, j, (0, 2, 3)], axis=0)
    l = unary + np.sum(unary_log_messages[i, j, (0, 1, 3)], axis=0)
    r = unary + np.sum(unary_log_messages[i, j, (0, 1, 2)], axis=0)

    return u, d, l, r

def compute_pairwise_log_messages(i, j, pairwise, pairwise_log_messages):
    '''
    Compute messages for U, D or L, R neighbors in log space
    '''
    # Log product
    ul = pairwise + pairwise_log_messages[i, j, 0]
    dr = pairwise + pairwise_log_messages[i, j, 1]
    
    # Sum
    ul = np.sum(np.exp(ul), axis=0)
    dr = np.sum(np.exp(dr), axis=0)
    
    return np.log(ul), np.log(dr)

def step(h, w, unary_potential, pairwise_potential, unary_log_messages, col_pairwise_log_messages, row_pairwise_log_messages):
    '''
    Compute and send all messages in log space
    '''
    
    # Send messages from unary nodes
    for i in range(h):
        for j in range(w):
            u, d, l, r = compute_unary_log_messages(i, j, unary_potential, unary_log_messages)

            if i > 0: col_pairwise_log_messages[i-1, j, 1] = u # Send message up
            if i < h-1: col_pairwise_log_messages[i, j, 0] = d # Send message down
            if j > 0: row_pairwise_log_messages[i, j-1, 1] = l # Send message left
            if j < w-1: row_pairwise_log_messages[i, j, 0] = r # Send Message right

    # Send messages from column pairwise nodes
    for i in range(h-1):
        for j in range(w):
            u, d = compute_pairwise_log_messages(i, j, pairwise_potential, col_pairwise_log_messages)

            unary_log_messages[i, j, 1] = u # Send up
            unary_log_messages[i+1, j, 0] = d # Send down

    # Send messages from row pairwise nodes
    for i in range(h):
        for j in range(w-1):
            l, r = compute_pairwise_log_messages(i, j, pairwise_potential, row_pairwise_log_messages)

            unary_log_messages[i, j-1, 3] = u # Send left
            unary_log_messages[i, j, 0] = d # Send right