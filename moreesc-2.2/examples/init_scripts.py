#! /usr/bin/python
# -*- coding: utf-8 -*-

import os, sys, time, warnings
import numpy as np
import scipy.io as io

# Change moreesc_dir if needed (must point to the folder containing setup.py and README)
moreesc_dir = os.path.abspath('..')
sys.path.insert(0, moreesc_dir)

import moreesc as mo
mp = mo.Profiles
mac = mo.AcousticResonator
mv = mo.Valve
ms = mo.Simulation

# Some convenient functions to properly name figure and data files according to 
# the name of the script.
data_dir = './data'
fig_dir = './fig'

for tmp in (fig_dir, data_dir):
    if not(os.path.isdir(tmp)):
        os.makedirs(tmp)
def filename(label=None, folder=fig_dir, ext=".pdf"):
    nom_base = os.path.basename(sys.argv[0])
    nom_base = nom_base[:nom_base.rfind('.py')].replace('.', '_')
    if label:
        nom_base += '_'+label
    return os.path.join(folder, nom_base + ext)
figfilename  = lambda *args, **kwargs: filename(folder=fig_dir, *args, **kwargs)
datafilename = lambda *args, **kwargs: filename(folder=data_dir, *args, **kwargs)


# Configuration of the Figure properties
show_fig = False
import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.style'] = 'normal'
mpl.rcParams['font.variant'] = 'normal'
mpl.rcParams['font.weight'] = 'normal'
mpl.rcParams['font.size'] = 14
mpl.rcParams['legend.fontsize'] = 14
mpl.rcParams['figure.figsize'] = (6, 4)
mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab

# Tempered scale
notes = [note+alt+octave  for octave in '123456' for note in 'CDEFGAB' for alt in ['','#']
    if not(alt is '#' and note in "BE")]
ts_frequencies = 440. * 2. ** ((np.arange(len(notes)) - notes.index('A3')) / 12.)
