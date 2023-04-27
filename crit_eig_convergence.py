# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 14:10:35 2023

@author: oscar
"""

from chebyshev_tau import ChebyshevTau
import numpy as np
import matplotlib.pyplot as plt

def get_crit_eig(order, M):
    tau = ChebyshevTau(order, M)

    tau.get_eigvals_eigvecs(1, 10000)
    
    curr_crit_eig = tau.critical_eig()
    
    return curr_crit_eig

def main(maxM, step):
    startM = 30
    Mvals = np.linspace(startM, maxM, int((maxM - startM)/step) + 1)
    print(Mvals)
    
    N = len(Mvals)
    
    orszag_imag = np.ones(N) * 0.00373967
    orszag_real = np.ones(N) * 0.23752649
    
    error_D1 = np.zeros(N) + np.zeros(N) * 1j
    error_D2 = np.zeros(N) + np.zeros(N) * 1j
    error_D4 = np.zeros(N) + np.zeros(N) * 1j
    
    for i in range(0, N):
        print("calculating error for M = " + str(Mvals[i]))
        error_D1[i] = get_crit_eig(1, int(Mvals[i]))
        error_D2[i] = get_crit_eig(2, int(Mvals[i]))
        error_D4[i] = get_crit_eig(4, int(Mvals[i]))
        
    print(orszag_imag[1])
    print(error_D1.imag)
    print(error_D2.imag)
    print(error_D4.imag)
        
    plt.plot(Mvals, orszag_imag, '-')
    plt.plot(Mvals, error_D1.imag,'-')
    plt.plot(Mvals, error_D2.imag,'-')    
    plt.plot(Mvals, error_D4.imag,'-')
    plt.show()
    
main(205, 10)