# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 17:00:42 2023

@author: oscar
"""

import numpy as np
import chebyshev as cheby
from chebyshev_tau import ChebyshevTau
import epssaver

def plot_streamfunction(crit_vec, c, t):
    '''
        Plots the streamfunction from a given eigenvector

    Parameters
    ----------
    crit_vec : LIST OF FLOATS
        Critical eigenvector
    
    c : FLOAT
        Critical eigenvalue
        
    t : FLOAT
        Time to display the streamfunction at

    Returns
    -------
    None.

    '''
    
    z_accuracy = 200
    x_accuracy = 400
    
    z_vals = np.zeros(z_accuracy)
    z = np.linspace(-1, 1, z_accuracy)
    x = np.linspace(0, 2*np.pi, x_accuracy)
    xz = np.zeros([z_accuracy, x_accuracy]) + 1j * np.zeros([z_accuracy, x_accuracy])
    zero_val = 0
    for i in range(0, len(crit_vec)):
        Tk = cheby.ChebyshevK(i, z_accuracy)
        
        zero_val += crit_vec[i] * Tk.evaluate(0)
        tk_vals = Tk.get_vals()
        
        z_vals = z_vals + crit_vec[i] * tk_vals   
    
    z_vals = z_vals / zero_val
    
    for j in range(0, x_accuracy):
        xz[:, j] = z_vals * np.exp(1j * x[j])       

    t_xz = xz * np.exp(-1j * c * t)
    
    levels = np.linspace(np.min(t_xz.real), np.max(t_xz.real), 21)    
    
    plt = epssaver.saver(8, 4)
    
    plt.set_args(xlabel=r'$x$', ylabel=r'$z$')
    
    plt.contourf(x, z, t_xz.real, levels=levels)
    plt.colorbar(levels)
    
    plt.saveeps('streamfunction')
    
def main(t):
    tau = ChebyshevTau(1, 200)
    
    tau.get_eigvals_eigvecs(1, 10000)   
    
    crit_vec = tau.critical_vec()
    crit_eig = tau.critical_eig()
    
    plot_streamfunction(crit_vec, crit_eig, t)

main(0)