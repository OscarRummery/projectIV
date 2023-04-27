# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 20:09:00 2022

@author: oscar
"""
import chebyshev as cheby
from matplotlib import pyplot as plt

'''
    Plots a bunch of Chebyshev polynomials
'''
def main():
   
    for i in range(0, 5):
        Tk = cheby.ChebyshevK(i, 50)
        Tk.plot()
    
    plt.legend()
    
    # the label may not work if you don't have latex installed
    plt.set_args(xlim=[-1,1], ylim=[-1.05,1.05], xlabel=r'$x$')
    plt.show()
    
main()