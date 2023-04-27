# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 15:58:51 2023

@author: oscar
"""

import numpy as np
from tau_matrices import D2, D4, Q_mat, P_mat

def setup_matrices(a, R, M, flowtype):
    '''
        Sets up the matrices for the fourth order method

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
    A : MATRIX
        Left hand side matrix
    B : MATRIX
        Right hand side matrix

    '''
    
    D_2 = D2(M)
    D_4 = D4(M)
    
    P = P_mat(M, flowtype)
    Q = Q_mat(M, flowtype)
    
    I = np.identity(M)

    A = (D_4 - 2 * a**2 * D_2 + a**4 * I) / (1j * a * R) - np.matmul(Q, D_2 - a**2 * I) + P
    B = -(D_2 - a**2 * I)

    # apply boundary conditions
    for k in range(0, M):
        A[M - 4, k] = 0
        A[M - 3, k] = 0
        A[M - 2, k] = 0
        A[M - 1, k] = 0
        
        B[M - 4, k] = 0
        B[M - 3, k] = 0
        B[M - 2, k] = 0
        B[M - 1, k] = 0
        
        if k % 2 == 0:
            A[M - 3, k] = 1
            A[M - 1, k] = k**2
        else:
            A[M - 4, k] = 1
            A[M - 2, k] = k**2

    return A, B