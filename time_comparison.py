# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 11:45:26 2023

@author: oscar
"""

from chebyshev_tau import ChebyshevTau
import time as time
import numpy as np
import epssaver

# this program compares the times taken for the different order methods to run
# as such takes a long time to run

def get_time(order, M):
    # get the time taken to run for specific M
    tau = ChebyshevTau(order, M)
    
    time_start = time.time()
    
    tau.get_eigvals_eigvecs(1, 10000)
    
    time_end = time.time()
    
    return time_end - time_start

def compare_time(M):
    time_D1 = get_time(1, M)
    time_D2 = get_time(2, M)
    time_D4 = get_time(4, M)
    
    return [time_D1, time_D2, time_D4]

def main(maxM, step):
    Mvals = np.linspace(0, maxM, int(maxM/step) + 1)
    
    Mvals[0] = 10
    print(Mvals)
    
    N = len(Mvals)
    
    times_D1 = np.zeros(N)
    times_D2 = np.zeros(N)
    times_D4 = np.zeros(N)
    
    for i in range(0, N):
        print("calculating times for M = " + str(Mvals[i]))
        times_D1[i], times_D2[i], times_D4[i] = compare_time(int(Mvals[i]))
    
    plt = epssaver.saver(6, 6)
    
    plt.plot(Mvals, times_D1,linestyle='-', label='First order')
    plt.plot(Mvals, times_D2,linestyle='-', label='Second order')
    plt.plot(Mvals, times_D4,linestyle='-', label='Fourth order')
    plt.legend()
    plt.saveeps('time_comparison')
    
main(500, 50)
        
    