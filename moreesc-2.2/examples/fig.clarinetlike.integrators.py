#!/usr/bin/python
# -*- coding: utf-8 -*-

#Copyright (C) 2012 Fabricio Silva

from init_scripts import *
from matplotlib.ticker import FuncFormatter

# Do not run this automatically!
sys.exit(None)

tsim, fs = 5., 44100.
zeta, gamma = 0.4, 1.
nfft = 8192

lIntegrators = {
    'Euler': ('euler', {'dt': 1e-2/fs},
        {'c':'r', 'ls':'-.', 'lw':2.}),
    'VODE':  ('vode', {'nsteps': 20000, 'method':'adams'},
        {'c':'b', 'lw':2., 'alpha':.3}),
    'LSODA': ('lsoda', {'nsteps': 20000},
        {'c':'g', 'ls':'-', 'lw':.2}),
    'RK45': ('dopri5', {'nsteps': 20000},
        {'c':'k', 'ls':'-.', 'lw':.5}),
    'RK58': ('dop853', {'nsteps': 20000},
        {'c':'k', 'ls':':', 'lw':.5}),
}

D = mv.ReedDynamics(wr=2*np.pi*1500., qr=.7, kr=8e8)
Z = mac.Cylinder(L=.5, r=7e-3, radiates=False, nbmodes=8)

h0 = .5 * D.stiffness() * mac.rho * (zeta / Z.Zc) ** 2
Pm0 = gamma * D.stiffness() * h0
H0 = mp.Constant(h0)
Pm = mp.Linear((0., tsim), (10., Pm0))

def get_envelop_filt(t, signal, fs=None, fmax=None):
    if fmax is None:
        nfft = int(2**np.ceil(np.log2(len(signal))))
        freq = np.linspace(0., fs, nfft+1)[:-1]
        fmax = freq[np.argmax(np.abs(np.fft.rfft(signal, nfft)))]
        del freq
    bands = (0, .05 * fmax / sim.fs, .5 * fmax / sim.fs, .5)
    import scipy.signal as ss
    order, wn = ss.cheb2ord(bands[1], bands[2], -10., -120.)
    b, a = ss.cheby2(order, 10., wn)
    sig_env = ss.lfilter(b, a, np.abs(signal))
    sig_env = ss.lfilter(b, a, sig_env)
    return t[len(b):], sig_env[len(b):]


fig, axs = plt.subplots(2, 1, sharex=False, squeeze=True)

keys = ('Euler', 'RK45', 'RK58', 'VODE', 'LSODA')
for k in keys:
    integ, kw, kw_plt = lIntegrators[k]
    label = integ
    if integ == 'euler':
        label += '_dt=%ffs' % (kw['dt']*fs)
    try:
        sim = ms.load_simulation(datafilename(label=label, ext='.h5'))
#        print(label + " loaded")
    except IOError:
#        print(label + 'integrating...')
        sim = ms.TimeDomainSimulation(D, Z, pm=Pm, h0=H0, fs=fs,
            piecewise_constant=False)
        sim.integrator.set_integrator(integ, **kw)
        sim.integrate(t=tsim, verbose=False)
        sim.save(datafilename(label=label, ext='.h5'))
        sim.save_wav(datafilename(label=label, ext='.wav'), where='in')
        sim.trace()
    print(k, sim.integration_time)
    tenv, env = get_envelop_filt(sim.time, sim.pressure, fs=sim.fs)
    tf, f = sim.get_instantaneous_frequency(mode='yin')
    envf = np.interp(tf, tenv, env)
    idx = envf > .5 * Pm(tf)

    axs[0].plot(Pm(tenv) / Pm0, env / Pm0, label=k, **kw_plt)
    axs[1].plot(Pm(tf[idx]) / Pm0, f[idx], label=k, **kw_plt)

#axs[0].plot(sim.time[::20], Pm(sim.time[::20]), 'k-.', label=r'$P_m$')
axs[0].legend(loc='upper center', ncol=5, prop={'size':'x-small'},
    columnspacing=1)
#axs[0].set_xlabel(r'$P_m/P_M$')
axs[0].set_ylabel(r'$|p|/P_M$')
axs[0].set_xlim(.47, .53)
axs[0].set_ylim(.00, .70)

axs[1].set_xlabel(r'$P_m/P_M$')
axs[1].set_ylabel(r'$f\ (Hz)$')
axs[1].set_xlim(.45, 1.)
plt.tight_layout()
fig.text(0.01, .98, "(a)", ha='left', va='top')
fig.text(0.01, .48, "(b)", ha='left', va='center')
fig.savefig(filename(), dpi=150)

if show_fig:
    plt.show()
