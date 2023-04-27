# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 10:52:45 2023

@author: oscar
"""

import numpy as np
from tau_matrices import D2, Q_mat, P_mat

def setup_matrices(a, R, M, flowtype):
    '''
        Sets up the matrices for the second order method

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
    E : MATRIX
        Left hand side matrix
    F : MATRIX
        Right hand side matrix

    '''
    
    N = 2*M

    I = np.identity(M)

    D_2 = D2(M)

    P = P_mat(M, flowtype)

    Q = Q_mat(M, flowtype)

    E = np.zeros([N, N]) + 1j*np.zeros([N, N])
    F = np.zeros([N, N]) + 1j*np.zeros([N, N])

    E[:M, :M] = D_2
    E[:M, M:] = -I
    E[M:, :M] = (P + a**2 * Q) + a**3 * I / (1j * R)
    E[M:, M:] = (D_2 - 2*a**2 * I) / (1j * a * R) - Q 

    F[M:, :M] = a**2 * I
    F[M:, M:] = -I

    # boundary conditions
    E[M-2] = 0
    F[M-2] = 0
    for k in range(0, M):
        if k % 2 == 0:
            E[M-2, k] = 1

    E[M-1] = 0
    F[M-1] = 0
    for k in range(0, M):
        if k % 2 == 1:
            E[M-1, k] = 1
        
    E[N-2] = 0
    F[N-2] = 0
    for k in range(0, M):
        if k % 2 == 0:
            E[N-2, k] = k**2
        
    E[N-1] = 0
    F[N-1] = 0
    for k in range(0, M):
        if k % 2 == 1:
            E[N-1, k] = k**2 
        
    return E, F

