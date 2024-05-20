'''
Implementations of Belief Popogation for Inference on Ising Models
'''
import numpy as np 

# Initialize messages
def initialize_messages(h, w):
    messages = {}
    for i in range(h):
        for j in range(w):
            if i > 0:
                messages[((i-1, j), (i, j))] = np.array([0.5, 0.5])
            if i < h - 1:
                messages[((i+1, j), (i, j))] = np.array([0.5, 0.5])
            if j > 0:
                messages[((i, j-1), (i, j))] = np.array([0.5, 0.5])
            if j < w - 1:
                messages[((i, j+1), (i, j))] = np.array([0.5, 0.5])
    return messages

# Update messages
def update_messages(lattice, messages, J, B, beta, num_iterations):
    h, w = lattice.shape
    for _ in range(num_iterations):
        new_messages = {}
        for (i, j), _ in messages.keys():
            new_messages[((i, j), (i, (j+1) % w))] = update_message(i, j, (i, (j+1) % w), lattice, messages, J, B, beta)
            new_messages[((i, j), ((i+1) % h, j))] = update_message(i, j, ((i+1) % h, j), lattice, messages, J, B, beta)
            new_messages[(((i-1) % h, j), (i, j))] = update_message((i-1) % h, j, i, j, lattice, messages, J, B, beta)
            new_messages[((i, (j-1) % w), (i, j))] = update_message(i, (j-1) % w, i, j, lattice, messages, J, B, beta)
        messages = new_messages
    return messages

# Update a single message
def update_message(i, j, ni, nj, lattice, messages, J, B, beta):
    h, w = lattice.shape
    incoming_messages = []
    for direction in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        if (ni+direction[0], nj+direction[1]) != (i, j) and (0 <= ni+direction[0] < h) and (0 <= nj+direction[1] < w):
            incoming_messages.append(messages[((ni+direction[0]) % h, (nj+direction[1]) % w), (ni, nj)])
    
    m_plus = np.exp(beta * (J + B))
    m_minus = np.exp(beta * (-J - B))
    for m in incoming_messages:
        m_plus *= m[1]
        m_minus *= m[0]

    total = m_plus + m_minus
    return np.array([m_minus / total, m_plus / total])

# Compute beliefs from messages
def compute_beliefs(lattice, messages, beta, B):
    h, w = lattice.shape
    beliefs = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            m_plus = np.exp(beta * B)
            m_minus = np.exp(-beta * B)
            for direction in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if 0 <= (i+direction[0]) < h and 0 <= (j+direction[1]) < w:
                    m_plus *= messages[((i+direction[0]) % h, (j+direction[1]) % w), (i, j)][1]
                    m_minus *= messages[((i+direction[0]) % h, (j+direction[1]) % w), (i, j)][0]
            beliefs[i, j] = m_plus / (m_plus + m_minus)
    return beliefs

# Belief Propagation Algorithm
# def belief_propagation(h, w, J, B, beta, num_iterations):
#     lattice = create_lattice(h, w)
#     messages = initialize_messages(h, w)
#     messages = update_messages(lattice, messages, J, B, beta, num_iterations)
#     beliefs = compute_beliefs(lattice, messages, beta, B)
#     return beliefs