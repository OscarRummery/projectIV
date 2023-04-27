# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 17:00:42 2023

@author: oscar
"""

import numpy as np
import chebyshev as cheby
import epssaver
from chebyshev_tau import ChebyshevTau

def plot_eigenfunction(crit_vec):
    '''
        Plots the eigenfunction from a given eigenvector

    Parameters
    ----------
    crit_vec : LIST OF FLOATS
        Critical eigenvector

    Returns
    -------
    None.

    '''
    
    accuracy = 201
    
    vals = np.zeros(accuracy)
    points = np.linspace(-1, 1, accuracy)
    zero_val = 0
    for i in range(0, len(crit_vec)):
        Tk = cheby.ChebyshevK(i, accuracy)
        
        zero_val += crit_vec[i] * Tk.evaluate(0)
        tk_vals = Tk.get_vals()
        
        vals = vals + crit_vec[i] * tk_vals   
    
    vals = vals / zero_val
    
    real_vals = vals.real
    imag_vals = vals.imag
    
    plt = epssaver.saver(6, 3)
    plt.set_args(xlabel=r'$z$')
    plt.plot(points, real_vals, label=r'$ \phi_r$')
    plt.plot(points, imag_vals, label=r'$ \phi_i$')
    
    plt.legend()
    
    plt.saveeps('eigenfunction')
    
def main():
    tau = ChebyshevTau(1, 200)
    
    tau.get_eigvals_eigvecs(1, 10000)   
    
    crit_vec = tau.critical_vec()
    plot_eigenfunction(crit_vec)

main()