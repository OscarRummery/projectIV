# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 11:20:18 2023

@author: oscar
"""

import numpy as np

# matrices needed for the tau method

def c_k(k):
    if k < 0:
        return 0
    elif k == 0:
        return 2
    else:
        return 1
    
# first derivative matrix    
def D1(M):
    D1 = np.zeros([M, M])
    
    for n in range(0, M):
        for p in range(n + 1, M):
            if (p + n) % 2 == 1:
                D1[n, p] = 2/c_k(n) * p
            
    return D1

# second derivative matrix
def D2(M):
    D2 = np.zeros([M, M])
    
    for n in range(0, M):
        for p in range(n + 2, M):
            if (p + n) % 2 == 0:
                D2[n, p] = 1/c_k(n) * p * (p**2 - n**2)
            
    return D2

# fourth derivative matrix
def D4(M):
    D4 = np.zeros([M, M])
    
    for n in range(0, M):
        for p in range(n + 4, M):
            if (p + n) % 2 == 0:
                q = p - n
                D4[n, p] = 1/(24 * c_k(n)) * p * (q**4 * (q**2 - 8) + n * q**3 * (6 * q**2 - 32) + (12 * n**2 * q**2 + 8 * n**3 * q) * (q**2 - 4) + 16 * q**2 + 32 * n * q)
            
    return D4

# Orszag matrix for plane Poiseuille flow (NOT USED)
def y2_mat(N):
    y2_mat = np.zeros([N, N])
    
    for n in range(0, N):
        if n - 2 >= 0:
            y2_mat[n, n - 2] = c_k(n - 2)
            
        y2_mat[n, n] = c_k(n) + c_k(n - 1)
            
        if n + 2 < N:
            y2_mat[n, n + 2] = 1
    
    return 1/4 * y2_mat

# Q as defined in Bukarri thesis 
def Q_mat(M, flowtype="p"):
    if flowtype == "p":
        return Q_p(M)
    elif flowtype == "c":
        return Q_c(M)
    else:
        print("Invalid flow type entered, defaulting to Poiseuille flow")
        return Q_p(M)

# for Poiseuille flow
def Q_p(M):
    Q = np.zeros([M, M])
    
    for i in range(0, M):
        if i < M-2:
            Q[i, i+2] = -1/4
            
        if i > 1:
            Q[i, i-2]  = -1/4
            
        Q[i, i] = 1/2
        
    Q[1, 1] = 1/4
    Q[2, 0] = -1/2

    return Q

# for Couette flow
def Q_c(M):
    Q = np.zeros([M, M])
    
    for i in range(0, M):
        if i < M-1:
            Q[i, i+1] = 1/2
            
        if i > 0:
            Q[i, i-1]  = 1/2
        
    Q[1, 0] = 1

    return Q

# P as defined in Bukhari thesis
def P_mat(M, flowtype="p"):
    if flowtype == "p":
        return -2 * np.identity(M)
    elif flowtype == "c":
        return np.zeros([M, M])+ np.zeros([M, M]) * 1j
    else:
        print("Invalid flow type entered, defaulting to Poiseuille flow")
        return -2 * np.identity(M)