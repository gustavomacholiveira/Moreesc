#! /usr/bin/python
# -*- coding: utf-8 -*-
# Copyright Fabrice Silva, 2010.

"""
"""

from init_scripts import *
path = lambda fname: os.path.join(data_dir, fname)

# Acoustic resonator
impedance = 'ImpedanceBaptiste'

# tmp hack (waiting for lsoda to reach debian unstable)
from scipy.integrate._ode import vode
ms.ode_solvers.lsoda = vode

try:
    sim = ms.load_simulation(os.path.join(data_dir, 'fig_trumpet_simulation_down.h5'))
except:
    raise IOError('Run fig.trumpet.simulation_(down|up).py first')

nfft = 1024
tif, fif = sim.get_instantaneous_frequency(mode='yin', bufsize=nfft, hopsize=64)
tif -= .5 * nfft / sim.fs

pm = sim.get_mouth_pressure()

fig, axs = plt.subplots(2, 1, sharex=True, figsize=(6,8))
axs[0].plot(sim.time, sim.pressure / pm.max())
axs[1].plot(tif, fif)
axs[0].set_xlim(6.25, 6.5)
axs[0].set_ylim(-4, 2)
axs[1].set_ylim(450, 600)
axs[1].set_yticks((450, 500, 550, 600))
axs[0].set_ylabel(r'$\tilde{p}(t)$')
axs[1].set_ylabel(r'$f\,(Hz)$')
axs[1].set_xlabel(r'$t\,(s)$')
fig.tight_layout()
fig.savefig(figfilename())

if show_fig:
    plt.show()
