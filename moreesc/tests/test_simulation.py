#! /usr/bin/python
# -*- coding: utf-8 -*-
# Copyright Fabrice Silva, 2012.

"""
"""
import numpy as np
import numpy.random as rd
import numpy.testing as tt
import sys, os, warnings, time
sys.path.append('../../')
from moreesc import Profiles, Valve, AcousticResonator, Simulation
from . import test_profile as ttp

# Change to True it if Simulation.py is modified
#__test__ = False
def setup():
    warnings.simplefilter("ignore")
def teardown():
    warnings.resetwarnings()
    ttp.del_tmpfile()

def atest_simulation():
    __test__ = False
    D = Valve.ReedDynamics(wr=2*np.pi*1500., qr=0.4, kr=2e8)
    Ze = AcousticResonator.Cylinder(L=.3, r=7e-3, radiates=True, nbmodes=15)

    pm = Profiles.Linear([0.05, .9, .95], [0., 1000., 0.])
    h0 = Profiles.Constant(3e-6) # Channel section at rest (in m**2)

    sim = Simulation.TimeDomainSimulation(D, Ze, pm=pm, h0=h0)
    #sim.fs = 96000
    sim.set_initial_state()
    integrator = 'vode'
    sim.solver(time=1., integrator=integrator)
    sim.save('/tmp/simu.dat')
    for where in ('in', 'out'):
        sim.save_wav('/tmp/simu_%s.wav' % where, where=where)
    if False:
        sim.trace(tmax=.5, trace_all=True, fmax=5000.)
        import matplotlib.pyplot as plt
        plt.show()

def test_measured_impedance():
    folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/'))
    # Acoustic resonator
    impedance = 'ImpedanceBaptiste' #'trompette_Lionel'
    try:
        Z = AcousticResonator.load_impedance(
            os.path.join(folder, impedance + '_Z.h5'))
    except IOError:
        Zc = AcousticResonator.rho * AcousticResonator.c / (np.pi * (8.4e-3)**2)
        if impedance == 'trompette_Lionel':
            Z = AcousticResonator.MeasuredImpedance(
                filename=os.path.join(folder, impedance + '.txt'),
                storage='txt_realimag', fmin=60., Zc=Zc)
            fapp = [82, 231, 343, 454, 569, 675, 783, 902, 1023, 1148, 1268]
            Qapp = [16, 23, 30, 31, 32, 34, 32, 26, 21, 12, 10]
        elif 'Baptiste' in impedance:
            Z = AcousticResonator.MeasuredImpedance(
                filename=os.path.join(folder, impedance + '.mat'),
                storage='mat_freqZ', fmin=60., Zc=Zc)
            fapp = [87, 236, 352, 472, 590, 706, 820, 933, 1060, 1184, 1305, 1429]
            Qapp = None
        
        Z.estimate_modal_expansion(algorithm='bruteforce',
            lFapp=fapp, lQapp=Qapp)
        Z.residues *= Z.Zc
        Z.save(os.path.join(folder, impedance + '_Z.h5'))

    tsim = 3.
    fs = 44100.
    t = np.linspace(0, tsim, tsim * fs)

    # Sequence Bb3 - F4 - Bb4 - D5 - F5 - Ab5 - Bb5 - C6 - D6 - E6
    freqs = np.array([233.1, 349.2, 466.2, 587.3, 698.5, 830.6, 932.3,
             1046.5, 1174.7, 1318.5]) - 10.

    #Valve
    qr_values = np.r_[.02, np.linspace(.05, .2,9)]
    if False:
        # Constant
        idx = 0
        wr = 2. * np.pi * freqs[idx]
        qr = qr_values[idx]
    elif False:
        # Linear decrease
        wr = Profiles.Linear([0., 5., 15.], 200. * np.pi * np.array([5., 3., .5]))
        qr = Profiles.Linear([0., 3., 5.], [.04, .04, .1])
    else:
        # Phrase F4, Bb4, D5, Ab5, Bb4, Ab5, Bb5
        idx = [0,1,2,1,2,4]
        N = (t.size + 1) / len(idx)
        if False:
            wr = np.empty_like(t, dtype=float)
            # Low pass filtering/interpolating
            ind = 0
            for f in freqs:
                wr[ind: ind + N] = 2. * np.pi * f
                ind += N
            if ind < wr.shape[0] - 1:
                wr[ind:] = wr[ind-1]
            wr = Profiles.Signal(time=t, signal=wr, smoothness=.1)
        else:
            # Convolution by a rectangular window (C0 continuous)
            tau = .01
            freqs *= 2. * np.pi
            tmp = np.c_[t[N::N]-tau, t[N::N]+tau].ravel()
            wrapper = lambda arr: np.c_[arr[idx], arr[idx]].ravel()[1:-1]
            wr = Profiles.Linear(tmp, wrapper(freqs))
            qr = Profiles.Linear(tmp, wrapper(qr_values))
            del tmp, wrapper
    D = Valve.LipDynamics(wr=wr, qr=qr, kr=8e8)

    pm = Profiles.SmoothStep([0.0, .001, 14.95, 15.], 20000., 0.)
    h0 = Profiles.Constant(1e-5) # Channel section at rest (in m**2)
    # Different stiffness wrt pm and p
    # h0 = h0 - pm/7e8

    sim = Simulation.TimeDomainSimulation(D, Z, pm=pm, h0=h0, fs=fs,
        piecewise_constant=True)

    from moreesc.Simulation import cms
    print(sim.valve.beating_factor, cms.get_beating_factor())
    if sim.integrator.successful():
#        sim.integrator.set_integrator('vode', method='adams', nsteps=10000)
#        sim.integrator.set_integrator('vode', method='bdf', nsteps=200000)
        sim.integrator.set_integrator('lsoda', nsteps=200000)
        sim.integrate(t=tsim)

    sim.label += '_' + impedance
    if True:
        sim.save()
        for where in ('in', 'out'):
            sim.save_wav(sim.label + '_%s.wav' % where, where=where)
    Figs = None
    if False:
        Figs = sim.trace(fmax=5000., trace_signals=False, trace_spectrums=True)
        import matplotlib.pyplot as plt
        plt.show()
    return sim, Figs
    
    
if __name__=='__main__':
    sim, Figs = test_measured_impedance()
#    tt.run_module_suite()
#    ttp.del_tmpfile()
