# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 11:16:46 2023

@author: oscar
"""

import numpy as np
from scipy.linalg import eig as sci_eig
from D1method import setup_matrices as D1_setup
from D2method import setup_matrices as D2_setup
from D4method import setup_matrices as D4_setup

'''
    Implements the Chebyshev Tau method
'''
class ChebyshevTau():
    def __init__(self, order, M, flowtype='p'):
        '''
        Initialise an instance of the Chebyshev Tau method with a specified order for a specific flow

        Parameters
        ----------
        order : INT
            The order of the method. 4 is fourth order, 2 is second order and 1 is first order
        M : INT
            The number of polynomials to use
        flowtype : CHAR, optional
            The type of flow, enter 'c' for Couette and 'p' for Poiseuille. The default is 'p'.

        Raises
        ------
        Exception
            If an invalid order or flowtype is entered

        Returns
        -------
        None.

        '''
        
        if not order in [1, 2, 4]:
            raise Exception("Invalid order entered. Please enter either 1, 2 or 4")
            
        if not flowtype in ['p', 'c']:
            raise Exception("Invalid flowtype entered. Please enter either p or c")
            
        self.order = order
        self.M = M
        self.flowtype = flowtype
        
        self.eigs = np.zeros(0)
        self.eig_vecs = np.zeros([M, 0]) + 1j * np.zeros([M, 0])
        
    def get_eigvals_eigvecs(self, a, R, ci_lower=-1):
        '''
            Get the eigenvalues and eigenvectors of a flow with a specific a and R 

        Parameters
        ----------
        a : FLOAT
            The wavenumber to solve for. Must be greater than 0
        R : FLOAT
            The Reynolds number of the flow.
        ci_lower : FLOAT, optional
            How far 'down' to truncate the eigvalues at. The default is -1.

        Returns
        -------
        LIST OF FLOAT
            Calculated eigenvalues
        LIST OF LIST OF FLOATS
            Calculated eigenvectors

        '''
        
        # setup matrices using entered parameters
        lhs_matrix, rhs_matrix = self.setup_system(a, R)
        
        # find the eigenvalues and eigenvectors using QZ algorithm
        eigs_split, vecs = sci_eig(lhs_matrix, rhs_matrix, homogeneous_eigvals=True)

        # filter out relevant eigenvalues and their corresponding eigenvectors
        self.eigs, self.eig_vecs = self.valid_eigs(eigs_split, vecs, a, R, ci_lower)
        
        return self.eigs, self.eig_vecs
    
    
    def valid_eigs(self, eigs_split, eig_vecs, a, R, ci_lower):
        '''
            Uses the bounds provided by Joseph (1969) to get the valid eigenvalues for the flow.
            Also removes spurious eigenvalues returned by the QZ algorithm

        Parameters
        ----------
        eigs_split : LISTS OF FLOATS
            Alpha and Beta returned by the QZ algorithm
        eig_vecs : LIST OF LIST OF FLOATS
            Eigenvectors returned by the QZ algorithm
        a : FLOAT
            Wavenumber of the flow
        R : FLOAT
            The Reynolds number of the flow.
        ci_lower FLOAT: 
            Lower bound for c_i

        Returns
        -------
        bound_eigs : LIST OF FLOATS
            The eigenvalues within the bounds we are considering
        bound_vecs : LIST OF LIST OF FLOATS
            Corresponding eigenvectors to the bounded eigenvalues

        '''
        alpha = eigs_split[0]
        beta = eigs_split[1]
        
        cr_lower = 0
        q = 0
        
        # apply Joseph's bounds
        if self.flowtype == 'p':
            cr_lower = -4 / (np.pi**2 + a**2) 
            q = 2
        elif self.flowtype == 'c':
            cr_lower = -1
            q = 1
        
        ci_upper = q / 2*a - (a * R)**(-1) * ((np.pi**2 * (np.pi**2 + a**2))/ (np.pi**2 + 4 * a**2) + a**2)
        
        count = len(alpha)
        
        bound_eigs = np.zeros(0)
        bound_vecs = np.zeros([self.M, count]) + 1j * np.zeros([self.M, count])
     
        vec_count = 0
        for i in range(0, count):            
            if beta[i] != 0: # exclude spurious eigenvalues caused by QZ algorithm
                valid_eig = np.longcomplex(alpha[i]) / np.longcomplex(beta[i])
                
                if valid_eig.real > cr_lower and valid_eig.real < 1:
                    if valid_eig.imag < ci_upper and valid_eig.imag > ci_lower:
                        bound_eigs = np.append(bound_eigs, valid_eig)
                        bound_vecs[vec_count] = eig_vecs[:, i]
                        vec_count += 1
        
        return bound_eigs, bound_vecs[:vec_count]
    
    def critical_eig(self):
        '''
            Get the critical eigenvalue for the system (the eigenvalue with the greatest imaginary part)

        Returns
        -------
        crit_eig : FLOAT
            The critical eigenvalue of the system

        '''
        
        max_imag = 0 
        crit_eig = 0 
        for i in range(0, len(self.eigs)):
            if self.eigs[i].imag > max_imag or i == 0:
                max_imag = self.eigs[i].imag
                crit_eig = self.eigs[i]
                        
        return crit_eig
        
    def critical_vec(self):
        '''
            Get the eigenvector for the critical mode of the system

        Returns
        -------
        crit_vec : LIST OF FLOATS
            The critical eigenvector of the system

        '''
        max_imag = 0  
        crit_index = 0
        for i in range(0, len(self.eigs)):
            if self.eigs[i].imag > max_imag or i == 0:
                max_imag = self.eigs[i].imag
                crit_index = i
       
        crit_vec = self.eig_vecs[crit_index, :self.M]
        
        return crit_vec
        
    def setup_system(self, a, R):
        '''
            Setup the matrix system using the correct order method

        Parameters
        ----------
        a : FLOAT
            Wavenumber of the system
        R : FLOAT
            Reynolds number of the system

        Returns
        -------
        lhs_matrix, rhs_matrix : MATRICES
            Matrices making up the left and right hand sides of the generalised eigenvalue problem for this system

        '''
        if self.order == 1:
            return D1_setup(a, R, self.M, self.flowtype)
        elif self.order == 2:
            return D2_setup(a, R, self.M, self.flowtype)
        elif self.order == 4:
            return D4_setup(a, R, self.M, self.flowtype)

