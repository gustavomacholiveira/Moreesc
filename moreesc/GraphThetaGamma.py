#!/usr/bin/python
# -*- coding: utf-8 -*-

#Copyright (C) 2006 Fabricio Silva

"""
Functions to draw Theta/Gamma graphs (as in Wilson & Beavers)
(frequency and pressure at oscillation threshold)
"""

__all__ = ['GraphThetaGamma','addtoTheta','addtoGamma','savefig','fill_between']

import matplotlib.pyplot as plt
import numpy as np


def fill_between(Graph, x, yinf, ysup):
    Sup = np.maximum(yinf, ysup)
    Inf = np.minimum(yinf, ysup)
    Vec_x = np.concatenate((x, x[::-1]))
    Vec_y = np.concatenate((Sup, Inf[::-1]))
    Graph.fill(Vec_x, Vec_y, facecolor='0.9', alpha=0.5)

def addtoTheta(Graph,*args, **kwargs):
    "Add a curve to Theta subplot without autoscaling."
    kwargs.update({'scalex':False,'scaley':False})
    return Graph.SubTheta.plot(*args, **kwargs)

def addtoGamma(Graph,*args, **kwargs):
    "Add a curve to Gamma subplot without autoscaling."
    kwargs.update({'scalex':False,'scaley':False})
    return Graph.SubGamma.plot(*args, **kwargs)

def savefig(Graph, basename, formats, dpi=300, size=(15,10)):
    "Save figure into specified formats."
    w = size[0]/2.54
    h = size[1]/2.54
    Graph.Fig.set_size_inches(w, h) #(w,h) : cm to inches
    for ext in formats:
        if ext in ['eps', 'png', 'ps', 'svg']:
            print(basename + '.' + ext)
            Graph.Fig.savefig(basename + '.' + ext, dpi=dpi)
        else:
            print("%s non valide." % ext)

class GraphThetaGamma():
    def __init__(self,Xlim = (0.0,9.0),Tlim = (0.0,1.1),Glim = (0, 1), \
        size=(15/2.54, 10/2.54)):
        """
        A GraphThetaGamma instance contains a figure with two subplots,
        the first is used to plot Theta data (frequency of emerging oscillation),
        the second one to Gamma data (mouth pressure at start of the oscillation).
        The X-axis is shared, and input arguments controls axis limits
        (no autoscaling).
        """
        self.Fig = plt.figure(None,size)
        self.Font = {}
#        self.Font = {'fontname' : 'Helvetica', 'fontsize' : 10}
        self.SubGamma = self.Fig.add_subplot(212)
        self.SubGamma.clear()
        self.SubGamma.grid('on')
        self.SubGamma.set_xlabel(r'$k_aL$',self.Font)
        self.SubGamma.set_ylabel(r'$\gamma$',self.Font)

        self.SubTheta = self.Fig.add_subplot(211, sharex=self.SubGamma)
        self.SubTheta.clear()
        self.SubTheta.grid('on')
        self.SubTheta.set_ylabel(r'$\theta$',self.Font)
        self.SubTheta.label_outer()
        
        self.setaxislim(Xlim, Tlim, Glim)

    def setaxislim(self, Xlim=None, Tlim=None, Glim=None):
        if Xlim!=None:
            self.Xlim = Xlim
        if Tlim!=None:
            self.Tlim = Tlim
        if Glim!=None:
            self.Glim = Glim
        self.SubTheta.set_xlim(self.Xlim)
        self.SubTheta.set_ylim(self.Tlim)
        self.SubGamma.set_xlim(self.Xlim)
        self.SubGamma.set_ylim(self.Glim)
        
    def addtoTheta(self,*args, **kwargs):
        "Add a curve to Theta subplot of this instance."
        return addtoTheta(self,*args, **kwargs)

    def addtoGamma(self,*args, **kwargs):
        "Add a curve to Gamma subplot of this instance."
        return addtoGamma(self,*args, **kwargs)

    def savefig(self, *args, **kwargs):
        "Save figure into specified formats."
        self.setaxislim()
        return savefig(self, *args, **kwargs)
    
    def show(self):
        self.setaxislim()
        plt.show()
