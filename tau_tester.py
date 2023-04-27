# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 16:15:04 2023

@author: oscar
"""

from chebyshev_tau import ChebyshevTau
import epssaver
import numpy as np

# use this method to test the tau method for specific flows
# plots the spectrum and returns the critical eigenvalue 
# along with whether the flow is stable or not

order = 1 # order of method: 4, 2 or 1 
M = 500 # number of polynomials
a = 1 # wavenumber
R = 30000 # Reynolds number
flowtype = 'p' # 'c' or 'p'

tau = ChebyshevTau(order, M, flowtype)

eigs, vecs = tau.get_eigvals_eigvecs(a, R)

print('Stable: ' + str(tau.critical_eig().imag > 0))
print('Critical eigenvalue= ' + str(tau.critical_eig()))

plt = epssaver.saver(4, 4)

plt.plot(np.real(eigs), np.imag(eigs), marker='.', linestyle=' ')
plt.set_args(xlabel=r'$c_r$', ylabel=r'$c_i$')
plt.set_args(xlim=[0, 1], ylim=[-1, 0.05])

plt.saveeps(flowtype + 'R=' + str(R) + 'a=' + str(a) + 'M=' + str(M) + 'order=' + str(order)) 
