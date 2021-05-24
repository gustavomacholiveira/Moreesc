#!/usr/bin/python
# -*- coding: utf-8 -*-

#Copyright (C) 2012 Fabricio Silva

from init_scripts import *
from matplotlib.ticker import FuncFormatter

# Do not run this automatically!
sys.exit(None)

tsim, fs = 1., 44100.
zeta = 0.4
nfft = 8192
trace_freq = False

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

def dev(f1, f2):
    return 1200. * np.log2(f1/f2)

keys = ('LSODA', 'Euler', 'RK45', 'RK58', 'VODE')
lPm = []

if trace_freq:
    fig, axs = plt.subplots(1, 1, sharex=False, squeeze=True)
markers = 'x*osv'
colors = 'bgyrm'

for gamma in np.arange(.4, 1., .1):
    print("gamma=", gamma)
    Pm0 = gamma * D.stiffness() * h0
    H0 = mp.Constant(h0)
    Pm = mp.SmoothStep((0, .01, tsim-.05, tsim), Pm0, 0)
    lPm.append(Pm0)
    if trace_freq:
        ax2 = plt.figure().add_subplot(111)

    fref = None
    for indk, k in enumerate(keys):
        integ, kw, kw_plt = lIntegrators[k]
        label = integ + "_gamma=%.1f" % gamma
        if integ == 'euler':
            label += '_dt=%ffs' % (kw['dt']*fs)
        try:
            sim = ms.load_simulation(filename(folder="data/integrators_steps",
                label=label, ext='.h5'))
            print(label + " loaded")
        except IOError:
            print(label + ' integrating...'),
            sim = ms.TimeDomainSimulation(D, Z, pm=Pm, h0=H0, fs=fs,
                piecewise_constant=False)
            sim.integrator.set_integrator(integ, **kw)
            sim.integrate(t=tsim, verbose=False)
            sim.save(filename(folder="data/integrators_steps",
                label=label, ext='.h5'))
            sim.save_wav(filename(folder="data/integrators_steps",
                label=label, ext='.wav'), where='in')
            sim.trace()

        gamma_shift = gamma + .01 * indk - .02
        if trace_freq:
            if True:
                nfft = int(2**np.ceil(6+np.log2(sim.pressure.size)))
                freqs = np.fft.fftfreq(nfft, d=1./sim.fs)
                FFT = np.abs(np.fft.rfft(sim.external_pressure, n=nfft))
                freqs = np.abs(freqs[:FFT.size])
                fmax = freqs[np.argmax(FFT)]
                if fref is None:
                    fref = fmax
                l, = axs.plot((gamma_shift,), (dev(fmax,fref),),
                    marker=markers[indk], c=colors[indk])
                idx2 = freqs < 500.
                ax2.plot(freqs[idx2], FFT[idx2], c=colors[indk])
            else:
                tenv, env = get_envelop_filt(sim.time, sim.pressure, fs=sim.fs)
                if hasattr(sim, 'f_i'): del sim.f_i
                tf, f = sim.get_instantaneous_frequency(
                    mode='yin', bufsize=2048, hopsize=2048)
                envf = np.interp(tf, tenv, env)
                tsteady = tenv[env > .5 * env.max()]
                tsteady = (tsteady[0], tsteady[-1])
                assert tsteady[1] - tsteady[0] > .5
                idx = np.logical_and(tf > tsteady[0], tf < tsteady[1])
                fmean = np.mean(f[idx])
                fstd = np.std(f[idx])
                if fref is None:
                    fref = fmean
                l, = axs.plot((gamma_shift,), (dev(fmean,fref),),
                    marker=markers[indk], c=colors[indk])
                axs.plot([gamma_shift,]*2,
                    (dev(fmean-fstd, fref), dev(fmean+fstd, fref)),
                    c=colors[indk], marker='_')
                ax2.plot(tf[idx], f[idx], c=colors[indk])

            if len(lPm) == 1:
                l.set_label(integ)
    if trace_freq:
        ax2.set_title(str(gamma))

if trace_freq:
    axs.legend(loc='upper center', ncol=5, prop={'size':'x-small'},
        columnspacing=1, numpoints=1)
    axs.set_xlabel(r'$\gamma=\frac{P_m}{K_rH_0}$')
    axs.set_ylabel(r'$\mathrm{dev}(f/f_{ref})$')
    fig.savefig(filename(), dpi=150)

# Timings
fname = '../data/integrators_steps/timings2.csv'
data = np.loadtxt(fname, usecols=range(1,8))
keys = np.loadtxt(fname, dtype=str, usecols=(0,)).tolist()
fig, axs = plt.subplots(2,1, sharex=True)
for inds in range(1, 6):
    kw = {'c': colors[inds-1], 'marker': markers[inds-1]}
    axs[0].plot(data[0], data[inds].T, **kw)
    axs[1].plot(data[0], (data[inds] / data[1]).T, label=keys[inds], **kw)

axs[1].set_ylim(0., 3.)
axs[1].legend(keys[1:], loc='lower center', ncol=5, prop={'size':'x-small'})
axs[1].set_xlabel(r'$p_m/p_M$')
axs[0].set_title(r'Absolute computation time ($s$)')
axs[1].set_title(r'Relative computation time (ref. LSODA)')
fig.tight_layout()
if show_fig:
    plt.show()
