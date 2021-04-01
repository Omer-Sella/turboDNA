# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 09:12:34 2021

@author: Omer Sella
"""
import numpy as np

# Constants
G_2 = np.array([1, 0], [1, 1])


"""
1. Generate the N'th  kronecker product of G_2
2. 
"""

def polarEncode(c, numberOfParityBits, Q_I_N_COMPLEMENT, Q_PC_N, powerOfG2):
    
    G = G_2
    # Maybe there is a -1 missing here
    for i in range(powerOfG2):
        # Check the order of a,b in np.kron(a,b)
        G = np.kron(G_2, G)
    N = c.shape[0]
    y = np.zeros(4, dtype = int)
    
    u = np.zeros(N, dtype = int)
    k = 0
    if numberOfParityBits > 0:
        for n in range(N):
            y = np.roll(y,1)
            if n in Q_I_N_COMPLEMENT:
                if n in Q_PC_N:
                    u[n] = y[0]
                else:
                    u[n] = c[k]
                    k = k + 1
                    y[0] = (y[0] + u[n]) % 2
            else:
                u[n] = 0
    else:
        for n in range(N):
            if n in Q_I_N_COMPLEMENT:
                u[n] = c[k]
                k = k + 1
            else:
                u[n] = 0
    encodedSequence =(u.dot(G) % 2)
    return encodedSequence