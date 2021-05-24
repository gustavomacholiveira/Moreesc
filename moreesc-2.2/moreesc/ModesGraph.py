#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Functions to draw complex modes evolution graphs
(frequencies and dampings)
"""

from numpy import array_equal
#from matplotlib.figure import Figure
import matplotlib.pyplot as plt

class ModesGraph():
    def __init__(self, Glim=(0,1.0), Tlim=(0,2.0), Dlim=(-10,10)):
        self.Fig = plt.figure(figsize=(15/2.54, 7/2.54))
        self.Font = {'fontname' : 'Helvetica', 'fontsize' : 10}
        self.Fig.subplots_adjust(0.10, None, 0.9, .90, 0.15)

        self.SubFreq = self.Fig.add_subplot(121)
        self.SubFreq.clear()
        self.SubFreq.grid('on')
        self.SubFreq.set_xlabel(r'$\gamma$',self.Font)
        self.SubFreq.set_ylabel(r'$\theta$',self.Font)

        self.SubDamp = self.Fig.add_subplot(122, sharex=self.SubFreq)
        self.SubDamp.clear()
        self.SubDamp.grid('on')
        self.SubDamp.set_xlabel(r'$\gamma$',self.Font)
        self.SubDamp.set_ylabel(r'$\alpha \,(s^{-1})$',self.Font)
        self.SubDamp.yaxis.set_label_position('right')
        self.SubDamp.yaxis.set_ticks_position('right')
        self.setaxislim(Glim, Tlim, Dlim)
        self.Lines = []

    def setaxislim(self, Glim=None, Tlim=None, Dlim=None):
        if Glim!=None:
            self.Glim = Glim
        if Tlim!=None:
            self.Tlim = Tlim
        if Dlim!=None:
            self.Dlim = Dlim
        self.SubFreq.set_xlim(self.Glim)
        self.SubFreq.set_ylim(self.Tlim)
        self.SubDamp.set_xlim(self.Glim)
        self.SubDamp.set_ylim(self.Dlim)
        
        
    def addtoModesGraph(self, *args, **kwargs):
        kwargs.update({'scalex':False,'scaley':False})
        kwargs.update({'linewidth':2.0})
        kwargsT = kwargs.copy()
        kwargsT['label'] = '_nolegend_'

        if 'label' not in kwargs:
            kwargs['label'] = "$k_rL=%.1f$" % self.krL

        if hasattr(self,'wr')==False:
            self.wr = 1

        tmpi = []
        tmpr = []

        for elt in args:
            if array_equal(elt,elt.real)==False:
                tmpr.append(elt.real)
                tmpi.append((elt.imag/self.wr))
            else:
                tmpr.append(elt)
                tmpi.append(elt)

        argsT = tuple(tmpi)
        argsD = tuple(tmpr)
        LinesT = self.SubFreq.plot(*argsT, **kwargsT)
        LinesD = self.SubDamp.plot(*argsD, **kwargs)
        for n in range(len(LinesT)):
            LinesD[n].set_c(LinesT[n].get_c())
            LinesD[n].set_ls(LinesT[n].get_ls())
            LinesD[n].set_lw(LinesT[n].get_lw())
        self.Lines += LinesT+LinesD
        return LinesT+LinesD

    def show(self):
        self.setaxislim()
        plt.show()

    def savefig(self, basename, formats='eps', size=(15,7)):
        "Save figure into specified formats."
        w = size[0]/2.54
        h = size[1]/2.54
        self.Fig.set_size_inches(w, h) #(w,h) : cm to inches
        self.SubFreq.legend(loc=4)
        for ext in formats:
            if ext in ['eps', 'png', 'ps', 'svg']:
                print(basename + '.' + ext)
                self.Fig.savefig(basename + '.' + ext,orientation='portrait')
