#!/usr/bin/python
# -*- coding: utf-8 -*-

#Copyright (C) 2008 Fabricio Silva
from init_scripts import *

tsim, fs = .5, 44100.
zeta, gamma = 0.3, 0.365

lqr = np.array([0.2, 0.4, 0.7 , 1.])[::-1]
fig, axs = plt.subplots(len(lqr), 1, sharex=True, figsize=(6,6))

for iqr, qr in enumerate(lqr):
    #print "\nqr = ", qr
    try:
        sim = ms.load_simulation(datafilename('_qr%0.1f' % qr, ext='.h5'))
    except IOError:
        D = mv.ReedDynamics(wr=2 * np.pi * 1500., qr=qr, kr=8e6)
        Z = mac.Cylinder(L=.30, r=7e-3, radiates=False, nbmodes=12)

        Kr = D.stiffness()
        h0 = (zeta / Z.Zc) ** 2 * Kr * mac.rho / 2.
        Pm0 = gamma * Kr * h0
        H0 = mp.Constant(h0)
        Pm = mp.SmoothStep([0.0001, .001, tsim - .01, tsim - .005], Pm0, 0)

        sim = ms.TimeDomainSimulation(D, Z, pm=Pm0, h0=h0, fs=fs,
            piecewise_constant=True)
        sim.integrator.set_integrator('lsoda', nsteps=20000)
        sim.integrate(t=tsim)
        if not(sim.integrator.successful()):
            plt.close(fig)
            raise ValueError("Simulation has not succeeded...")
        sim.save(datafilename('_qr%0.1f' % qr, ext='.h5'))

        sim.save_wav(datafilename('_qr%0.1f' % qr, ext='.wav'), where='in')

    nfft = 2048
    fmin, fmax = 0., 3000.
    tmp = np.r_[np.zeros(nfft), sim.pressure, np.zeros(nfft)]
    Pxx, fr, bins = mlab.specgram(tmp, NFFT=nfft, Fs=sim.fs, \
        window=np.hanning(nfft), noverlap=int(.98 * nfft))
    idx = np.logical_and(fr < fmax, fr > fmin)
    Pxx = 10. * np.log10(Pxx[idx, :] + 1e-9 * Pxx.max())
    tmp = Pxx[np.isfinite(Pxx)]
    vmax = tmp.max()

    ax = axs[iqr]
    im = ax.imshow(Pxx, aspect='auto', origin='lower', cmap=cm.hot_r, \
            extent=(0, tsim, fmin * 1e-3, fmax * 1e-3),
            interpolation='sinc', vmin=vmax - 80, vmax=vmax)
    ax.text(.98, .90, s=r'$q_r=%.1f$' % qr, va='top', ha='right',
            transform=ax.transAxes, bbox={'facecolor': 'w', 'alpha':.80})

axs[-1].set_xlabel(r'$t\ (s)$')
for ax in axs:
    ax.set_ylabel(r'$f\ (kHz)$')
fig.tight_layout()
fig.savefig(filename(), dpi=150)

if show_fig:
    plt.show()
