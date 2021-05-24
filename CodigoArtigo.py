#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 15:31:04 2021

@author: gustavo
"""


# Teste apresentando no artigo: MoReeSC: a framework for the simulation and analysis
# of sound production in reed and brass instruments

import numpy as np
from moreesc import *
from moreesc.AcousticResonator import MeasuredImpedance


# Sampling frequency of output signals
fs = 44100

# Time range of the simulation
tsim = 2

# Load and estimate modal expansion of the input impedance
Bore = MeasuredImpedance(filename='data/trompette_Lionel.txt')
# para carregar a impedancia funcionou assim: 
# Bore = MeasuredImpedance(filename='ImpedanceBaptiste.mat', storage='mat_freqZ', fmin=60., Zc=Zc)
Bore.estimate_modal_expansion()

# Linearly decreasing resonance frequency of the lip reed
fr = Linear([0., tsim], [1000., 100.])
Lips = LipDynamics(wr = 2.*np.pi*fr, qr = 1, kr = 8e8)

# Mouth pressure (in Pa)
pm = 2e4

# Chanel section at rest (in m^2)
h0 = 1e-5

# Simulation Configuration
sim = TimeDomainSimulation (Lips, Bore, pm = pm, h0 = h0, fs = fs)

sim.integrator. set_integrator('lsoda')
sim.integrate(t=tsim)
