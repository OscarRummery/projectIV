# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 20:16:17 2022

@author: oscar
"""
import numpy as np
import matplotlib.pyplot as plt
'''
    A class that implements a Chebyshev polynomial
    with functions to obtain its important properties
'''
class ChebyshevK():
    def __init__(self, k, N):
        '''
        Initialise a Chebyshev polynomial of order k, with N points

        Parameters
        ----------
        k : INT
            The order of the Chebyshev polynomial
        N : INT
            The number of points to initialise with

        Returns
        -------
        None.

        '''
        
        self.k = k
        self.N = N
        
        self.set_points()
        self.calculate_vals()
        
    def evaluate(self, z):
        '''
        Evaluate the Chebyshev polynomial at a specific z value

        Parameters
        ----------
        z : FLOAT
            The z value to evaluate the polynomial at

        Returns
        -------
        val : FLOAT
            The value of the polynomial at this point

        '''
        
        theta = np.arccos(z)
        
        val = np.cos(self.k * theta)
        
        return val
    
    def calculate_vals(self):
        '''
        Calculate the values the polynomial takes at its points

        Returns
        -------
        None.

        '''
        vals = np.zeros(self.N)
        for i in range(0, self.N):
            z = self.points[i]
            vals [i] = self.evaluate(z)
                    
        self.vals = vals
        
    def get_vals(self):
        '''
        Get a list of all values this polynomial takes at its points

        Returns
        -------
        List of floats
            The pre calculated values

        '''
        return np.array(self.vals)
    
    def set_points(self):
        '''
        Set the points to pre evaluate this polynomial at

        Returns
        -------
        None.

        '''
        points = np.linspace(-1, 1, self.N)
            
        self.points = points 
        
    def get_zeros_cosless(self):
        '''
        Get the zeros of this polynomial without applying the np.cos function to them

        Returns
        -------
        zeros : LIST OF FLOATS
            A list of the zeros of this polynomial

        '''
        zeros = np.zeros(self.k)
        
        for i in range(0, self.k):
            zeros[i] = (1 + 2*i) * np.pi / (2 * self.k)
            
        return zeros
    
    def gauss_lobatto(self):
        '''
        The Gauss-Lobatto points for this polynomial. Needed for the Chebyshev collocation method.

        Returns
        -------
        LIST OF FLOATS
            A list of the Gauss-Lobatto points of this polynomial

        '''
        zeros = np.pi * np.linspace(0, self.k, self.k, True) / self.k        
        
        return np.cos(zeros)
    
    def plot(self):       
        '''
        Plot the polynomial

        Returns
        -------
        None.

        '''
        plt.plot(self.points, self.vals, '-', label=r'$T_' + str(self.k) + '(x)$')
        plt.draw()
        
    