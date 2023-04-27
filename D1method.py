# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 10:52:45 2023

@author: oscar
"""

import numpy as np
from tau_matrices import D1, Q_mat, P_mat

def setup_matrices(a, R, M, flowtype):
    '''
        Sets up the matrices for the first order method

    Parameters
    ----------
    a : FLOAT
        Wavenumber
    R : FLOAT
        Reynolds number
    M : INT
        Number of polynomials being used
    flowtype : CHAR
        'c' for Couette flow and 'p' for Poiseuille

    Returns
    -------
    G : MATRIX
        Left hand side matrix
    H : MATRIX
        Right hand side matrix

    '''
    
    N = 4*M

    I = np.identity(M)

    D_1 = D1(M)

    P = P_mat(M, flowtype)

    Q = Q_mat(M, flowtype)

    G = np.zeros([N, N]) + 1j*np.zeros([N, N])
    H = np.zeros([N, N]) + 1j*np.zeros([N, N])

    G[:M, :M] = G[M:2*M, M:2*M]= G[2*M:3*M, 2*M:3*M] = D_1
    G[3*M:, 3*M:] = D_1 / (1j * a * R)
    G[:M, M:2*M] = G[M:2*M, 2*M:3*M] = G[2*M:3*M, 3*M:] = -I
    G[3*M:, :M] = (P + a**2 * Q) + a**3 * I /(1j*R)
    G[3*M:, 2*M:3*M] = -Q - 2 *a * I / (1j*R)
    
    H[3*M:, :M] = a**2 * I
    H[3*M:, 2*M:3*M] = -I

    # boundary conditions
    # z_1=0 on x=-1
    G[M-1] = 0
    H[M-1] = 0
    for k in range(0, M):
        if k % 2 == 0:
            G[M-1, k] = 1

    # z_1=0 on x=1
    G[2*M-1] = 0
    H[2*M-1] = 0
    for k in range(0, M):
        if k % 2 == 1:
            G[2*M-1, k] = 1
        
    # z_2 = 0 on x=-1
    G[3*M - 1] = 0
    H[3*M - 1] = 0
    for k in range(0, M):
        if k % 2 == 0:
            G[3*M - 1, M + k] = 1
        
    # z_2 = 0 on x=1
    G[N-1] = 0
    H[N-1] = 0
    for k in range(0, M):
        if k % 2 == 1:
            G[N-1, M + k] = 1
        
    return G, H