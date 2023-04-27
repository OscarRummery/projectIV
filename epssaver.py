# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 20:00:02 2023

@author: oscar
"""

import matplotlib
import matplotlib.pyplot as plt

## code from https://stackoverflow.com/questions/67512225/saving-matplotlib-graphs-with-latex-fonts-as-eps
## turned into a class for general use
class saver():
    def __init__(self, fig_x, fig_y, font_size=11):
        '''
        
        Parameters
        ----------
        fig_x : float
            Size of the figure - x dimension.
        fig_y : float
            Size of the figure - y dimension.
        font_size : float, optional
            Font size of the figure. The default is 11.

        Returns
        -------
        None.
        
        '''
        
        ## may not be necessary - uncomment if issues with outputted plot
        ## matplotlib.use('QtAgg')
        
        ## reset defaults
        plt.rcdefaults()
        
        ## Set up LaTeX fonts
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": font_size,
            })
        
        self.fig, self.ax = plt.subplots(figsize=(fig_x, fig_y))
        
    def plot(self, x_data, y_data, **kwargs):
        '''
        
        plot data on the figure
        
        Parameters
        ----------
        x_data : array
            A 1D array of the x-coordinates.
        y_data : 2D array
            A 1D array of the y-data. Must be same size as x_data.
        **kwargs : args 
            args to be applied to the plotted data e.g. linestyle, marker, linewidth.
            Must be in the form arg=argvalue.

        Returns
        -------
        None.

        '''
            
        self.ax.plot(x_data, y_data, **kwargs)
        
    def contourf(self, x, z, vals, **kwargs):
        self.contour = self.ax.contourf(x, z, vals, **kwargs)
        
        
    def colorbar(self, levels):
        self.fig.colorbar(self.contour, ticks=levels)
            
    def set_args(self, **kwargs):
        '''

        set the general args for the plot

        Parameters
        ----------
        **kwargs : args
            args to be applied to the whole plot e.g. xlim, ylim. 
            Must be in the form arg=argvalue.

        Returns
        -------
        None.

        '''
        self.ax.set(**kwargs)
        
    def legend(self):
        '''
        Enables legend

        '''
        
        self.ax.legend(loc='upper right')
        

    def saveeps(self, file_name): 
        '''
        
        Saves the figure as a .eps file

        Parameters
        ----------
        file_name : string
            name of saved file.

        Returns
        -------
        None.

        '''
        
        self.fig.savefig(file_name + ".eps", format="eps", dpi=1200, bbox_inches="tight", transparent=True)