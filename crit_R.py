# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 15:24:28 2023

@author: oscar
"""

import numpy as np
from chebyshev_tau import ChebyshevTau
from epssaver import saver

# the following bijection code is modified from 
# https://stackoverflow.com/questions/14392208/how-to-do-the-bisection-method-in-python
def samesign(a, b):
    return a * b > 0

def bisect(a, lowR, highR):
    '''
        Find root of continuous function where f(low) and f(high) have opposite signs

    Parameters
    ----------
    a : FLOAT
        Wavenumber to calculate at
    lowR : FLOAT
        Lowest R value
    highR : FLOAT
        Highest R value

    Returns
    -------
    margR : FLOAT
        Marginal R value calculated

    '''
    
    tau = ChebyshevTau(4, 50)
    
    tau.get_eigvals_eigvecs(a, lowR)
    lowImag = tau.critical_eig().imag
    
    tau.get_eigvals_eigvecs(a, highR)
    
    for i in range(54):
        midR = (lowR + highR) / 2.0        
        
        tau.get_eigvals_eigvecs(a, lowR)
        lowImag = tau.critical_eig().imag
        
        tau.get_eigvals_eigvecs(a, midR)
        midImag = tau.critical_eig().imag
        
        if samesign(lowImag, midImag):
            lowR = midR
        else:
            highR = midR
            
    margR = midR

    return margR

def compute_marginal_r(low_a, high_a, points):
    '''
        Computes the marginal R value for a specified number of points between specified values of a 

    Parameters
    ----------
    low_a : FLOAT
        Lowest a value
    high_a : FLOAT
        Highest a value
    points : INT
        Number of points to calculate at

    Returns
    -------
    a_vals : LIST OF FLOATS
        a values the marginal R values have been calculated at
    r_vals : LIST OF FLOATS
        the marginal R values calculated

    '''
    a_vals = np.linspace(low_a, high_a, points)
    r_vals = np.zeros(points)
    
    for i in range(0, points):
        r_vals[i] = bisect(a_vals[i], 4000, 10000)
    
    return a_vals, r_vals

def find_crit_r(a_min, a_max, recursion_count):  
    '''
        Recursively searches further and further into minimum point to find critical Reynolds number

    Parameters
    ----------
    a_min : FLOAT
        Minimum wavenumber - the start of the search
    a_max : FLOAT
        Maximum wavenumber - the end of the search
    recursion_count : INT
        Maximum number of recursions to do

    Returns
    -------
    Critical R value : FLOAT
        Value of the critical Reynolds number found
    Critical wavenumber : FLOAT
        Value of the critical wavenumber found

    '''
    a_vals, r_vals = compute_marginal_r(a_min, a_max, 5)
    min_index = np.argmin(r_vals)
    
    # if our value is accurate enough, exit and return the values found
    if  abs(np.max(r_vals) - r_vals[min_index]) < 0.0001 or recursion_count >= 20:
        return r_vals[min_index], a_vals[min_index]

    if min_index == 0:
        # the minimum value is at the mimimum of our range of a
        # search between that a value and the next
        
        return find_crit_r(a_vals[min_index], a_vals[min_index + 1],  recursion_count + 1)
    elif min_index == len(a_vals) - 1:
        # the minimum value is at the maximum of our range of a 
        # search between that a value and the previous a value
        
        return find_crit_r(a_vals[min_index - 1], a_vals[min_index], recursion_count + 1)
    else:
        # the minimum value is not at either endpoint
        # check between the a values on both sides of the minimum value found
        
        left_r, left_a = find_crit_r(a_vals[min_index - 1], a_vals[min_index], recursion_count + 1)
        right_r, right_a = find_crit_r(a_vals[min_index], a_vals[min_index + 1], recursion_count + 1)
        
        if left_r < right_r:
            return left_r, left_a
        else:
            return right_r, right_a

def main():
    # THIS PROGRAM TAKES AGES TO RUN
    # Comment out the section you don't want to calculate 
    # or reduce the M values used
    
    # find the critical R and a values
    crit_r, crit_a = find_crit_r(1.02, 1.03, 0)
    
    print('Critical R=' + str(crit_r))
    print('Critical alpha=' + str(crit_a))
    
    # plot a curve of marginal stability
    computed_a, computed_r = compute_marginal_r(0.843, 1.0972, 150) 
    
    plt = saver(4, 4)
    plt.plot(computed_a, computed_r, linestyle='-')
    plt.plot(crit_a, crit_r, color='#1f77b4', marker='.')
    plt.set_args(xlabel=r'$\alpha$', ylabel=r'$R$')
    
    plt.saveeps('marginal_stability_curve')
    
main()    
        
            
    

