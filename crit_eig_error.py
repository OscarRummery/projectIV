# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 14:10:35 2023

@author: oscar
"""

from chebyshev_tau import ChebyshevTau
import numpy as np
import matplotlib.pyplot as plt
import math

def calculate_error(last_eig, cur_eig):
    # same_count = 0
    # length = len(str(last_eig.real))
    # for i in range(0, length):
    #     if str(last_eig.real)[i] == str(cur_eig.real)[i] and str(last_eig.imag)[i] == str(cur_eig.imag)[i]:
    #         same_count += 1
    #     else:
    #         return 10 ** (-same_count)
    
    # return 10 ** (-same_count)
    return abs(cur_eig.imag - last_eig.imag)    

def error_check(order, M, last_eig):
    #prev_tau = ChebyshevTau(order, M - 1)
    curr_tau = ChebyshevTau(order, M)

    #prev_tau.get_eigvals_eigvecs(1, 10000)
    curr_tau.get_eigvals_eigvecs(1, 10000)
    
    prev_crit_eig = last_eig # prev_tau.critical_eig()
    curr_crit_eig = curr_tau.critical_eig()
    
    error = calculate_error(prev_crit_eig, curr_crit_eig)
    
    return error, curr_crit_eig

def main(maxM, step):
    startM = 5
    Mvals = np.linspace(startM, maxM, int((maxM - startM)/step) + 1)
    print(Mvals)
    
    N = len(Mvals)
    
    error_D1 = np.zeros(N)
    error_D2 = np.zeros(N)
    error_D4 = np.zeros(N)
    
    last_D1 = 0 + 0j
    last_D2 = 0 + 0j
    last_D4 = 0 + 0j
    
    for i in range(0, N):
        print("calculating error for M = " + str(Mvals[i]))
        error_D1[i], last_D1 = error_check(1, int(Mvals[i]), last_D1)
        error_D2[i], last_D2 = error_check(2, int(Mvals[i]), last_D2)
        error_D4[i], last_D4 = error_check(4, int(Mvals[i]), last_D4)
        
    print(error_D1)
        
    plt.plot(np.log(Mvals), -np.log10(error_D1),'-')
    plt.plot(np.log(Mvals), -np.log10(error_D2),'-')    
    plt.plot(np.log(Mvals), -np.log10(error_D4),'-')
    plt.show()
    
    print(-np.log10(error_D2))
    
main(101, 2)