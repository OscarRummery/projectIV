# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 19:59:44 2022

@author: oscar
"""

import chebyshev as cheby
import numpy as np
from tau_matrices import D1, D2, D4
import epssaver as ep

def c_k(k):
    if k < 0:
        return 0
    elif k == 0:
        return 2
    else:
        return 1

def f(z):
    # function to approximate
    
    return np.cos(z)

def f_diff(z):
    # first derivative of function
    return -np.sin(z)

def f_diff2(z):
    # second derivative of function
    return -np.cos(z)

def f_diff4(z):
    # fourth dericative of function
    return np.cos(z)

def calculate_coeff(k, N, zeros):
    # calculates coefficient for Chebyshev polynomial of specific order
    f_zeros = np.array([f(zi) for zi in np.cos(zeros)])
        
    elems = f_zeros * np.cos(k * zeros)
    
    return (2 / (c_k(k) * (N + 1))) * np.sum(elems)

def get_coeffs(N):
    # get the coefficients of the functions Chebyshev expansion
    coeffs = np.zeros(N)
    
    Tmax = cheby.ChebyshevK(N + 1, 32)
    zeros = Tmax.get_zeros_cosless()
    
    for i in range(0, N):
        coeffs[i] = calculate_coeff(i, N, zeros)
        
    return coeffs    

def plot_comparison(points, func_vals, approx_vals, file_name):
    # plot a comparison between actual function and the approximation
    plt = ep.saver(3.75, 3.75)
        
    plt.set_args(xlim=[-1, 1])
    plt.set_args(ylim=[-1.02, 1.02])
    plt.set_args(xlabel=r'$z$')
    
    plt.plot(points, func_vals , linestyle='-', color='#ff7f0e')
    plt.plot(points, approx_vals, linestyle='-', color='#1f77b4')
       
    plt.saveeps(str(file_name))

def main():
    order = 8
    accuracy = 1000
    
    D_1 = D1(order + 1)
    D_2 = D2(order + 1)
    D_4 = D4(order + 1)
    
    # get the approximation coefficients
    coeffs = get_coeffs(order + 1)
    
    # work out the coefficients of the derivatives using D matrices
    firstDiffCoeffs = np.matmul(D_1, coeffs)
    secondDiffCoeffs = np.matmul(D_2, coeffs)
    fourthDiffCoeffs = np.matmul(D_4, coeffs)
    
    vals = np.zeros(accuracy)
    firstDiffvals = np.zeros(accuracy)
    secondDiffvals = np.zeros(accuracy)
    fourthDiffvals = np.zeros(accuracy)
    
    points = np.linspace(-1, 1, accuracy)
    for i in range(0, order + 1):
        Tk = cheby.ChebyshevK(i, accuracy)
        
        tk_vals = Tk.get_vals()
        
        vals = vals + coeffs[i] * tk_vals
        firstDiffvals = firstDiffvals + firstDiffCoeffs[i] * tk_vals
        secondDiffvals = secondDiffvals + secondDiffCoeffs[i] * tk_vals
        fourthDiffvals = fourthDiffvals + fourthDiffCoeffs[i] * tk_vals
        
    # plot approximation to main function
    func_vals = np.array([f(point) for point in points])
    
    plot_comparison(points, func_vals, vals, 'chebyapprox_order=' + str(order))
    
    # plot approximation to first deriv
    diff1_func_vals = np.array([f_diff(point) for point in points])
    
    plot_comparison(points, diff1_func_vals, firstDiffvals, 'diff1approx_order=' + str(order))
    
    # plot approximation to second deriv
    diff2_func_vals = np.array([f_diff2(point) for point in points])
 
    plot_comparison(points, diff2_func_vals, secondDiffvals, 'diff2approx_order=' + str(order))
    
    # plot approximation to fourth deriv
    diff4_func_vals = np.array([f_diff4(point) for point in points])
    
    plot_comparison(points, diff4_func_vals, fourthDiffvals, 'diff4approx_order=' + str(order))
    
main()