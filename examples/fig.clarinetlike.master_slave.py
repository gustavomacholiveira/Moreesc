#!/usr/bin/python
# -*- coding: utf-8 -*-

#Copyright (C) 2008 Fabricio Silva
from init_scripts import *

tsim, fs = .20, 44100.
zeta, gamma = 0.4, 0.45
nfft = 4096
Tmax = .15

t0 = .001
lTatt = np.array([1./fs, 1e-2])
lLabels = [r'$1/F_e$', '10ms']
lConfigs = {
    'fast': {'tatt': 1./fs, 'label':r'$1/fs$'},
    'slow': {'tatt': 1.e-2, 'label':r'$10\,ms$'},
}

for k, dic in lConfigs.items():
    fig = plt.figure(figsize=(6.,6.))

    try:
        sim = ms.load_simulation(datafilename(k, ext='.h5'))
    except IOError:
        D = mv.ReedDynamics(wr=2*np.pi*1500., qr=.4, kr=8e6)
        Z = mac.Cylinder(L=.5, r=7e-3, radiates=False, nbmodes=8)

        h0 = .5 * D.stiffness() * mac.rho * (zeta / Z.Zc) ** 2
        Pm0 = gamma * D.stiffness() * h0
        H0 = mp.Constant(h0)
        Pm = mp.SmoothStep([t0, t0 + dic['tatt'], tsim - .01, tsim - .005],
            Pm0, 0)

        sim = ms.TimeDomainSimulation(D, Z, pm=Pm, h0=H0, fs=fs,
            piecewise_constant=False)
        sim.integrator.set_integrator('lsoda', nsteps=20000)
        sim.integrate(t=tsim)
        if not(sim.integrator.successful()):
            plt.close(fig)
            raise ValueError("Simulation has not succeeded...")
        sim.save(datafilename(k, ext='.h5'))
        sim.save_wav(datafilename(k, ext='.wav'), where='in')

    # Signals
    Nmax = min(8, sim.Nac)
    indt = sim.time < Tmax
    time = sim.time[indt]
    Xa = ms.cc.get_acoustic_state_vector(sim.Nac, sim.result[:, indt]).real
    pM = sim.valve.stiffness() * sim.h0.max()

    ax0 = fig.add_subplot(Nmax, 2, 1)
    ax0.set_ylabel(r'$p$') #' + '\n' + r'$(kPa)$', ha='center')
    ax0.yaxis.labelpad += .05
    ax0.plot(time, sim.pressure[indt] / pM, 'b')
    ax0.plot(time, sim.pm[indt] / pM, 'r-.')
    lYTicks = [gamma,]

    for indn in range(Nmax - 1):
        ax = fig.add_subplot(Nmax, 2, 2 * indn + 3,
            ylabel='$p_{%d}$' % (indn + 1), sharex=ax0)
        pn = 2. * Xa[indn]
        ax.plot(time, pn / pM, 'b')
        lYTicks.append(pn.max() / pM)
    
    for ind, ax in enumerate(fig.get_axes()):
        tmp = np.abs(ax.get_ylim()).max()*1.05
        ax.set_ylim((-tmp, tmp))
        tmp = lYTicks[ind]
        ax.set_yticks([-tmp, tmp])
        ax.set_yticklabels(['-%.2f'% (tmp,), '+%.2f'% (tmp,)])
        ax.yaxis.get_label().set_rotation('horizontal')
        ax.set_xlim((0, Tmax))
        ax.label_outer()
        ax.set_xticks((0, Tmax / 2, Tmax))

    ax.set_xlabel(r'$t\,(ms)$')
    ax.set_xticklabels([int(tmp*1000) for tmp in ax.get_xticks()])
    fig.tight_layout()

    # Spectrogram
    fmin, fmax = 0., 3000.
    tmp = np.r_[np.zeros(nfft), sim.pressure, np.zeros(nfft)]
    Pxx, fr, bins = mlab.specgram(tmp, NFFT=nfft, Fs=sim.fs, \
        window=np.hanning(nfft), noverlap=int(.98 * nfft))
    idx = np.logical_and(fr < fmax, fr > fmin)
    Pxx = 10. * np.log10(Pxx[idx, :] + 1e-9 * Pxx.max())
    tmp = Pxx[np.isfinite(Pxx)]
    vmax = tmp.max()

    ax = fig.add_subplot(1, 2, 2, xlabel=r'$t\,(ms)$', ylabel=r'$f\,(kHz)$')
    im = ax.imshow(Pxx, aspect='auto', origin='lower', cmap=cm.hot_r, \
            extent=(0, tsim, fmin * 1e-3, fmax * 1e-3),
            interpolation='sinc', vmin=vmax - 80, vmax=vmax)
#    ax.text(.98, .98, s=dic['label'], va='top', ha='right',
#            transform=ax.transAxes, bbox={'facecolor': 'w', 'alpha':.80})
            
    for tmp in ax.yaxis.get_major_ticks():
        tmp.tick1On, tmp.tick2On = True, True
        tmp.label1On, tmp.label2On = False, True
    ax.yaxis.set_label_position('right')
    ax.yaxis.set_ticks_position('both')
    ax.set_xlim((0, Tmax))
    ax.set_xticks((0, Tmax / 2, Tmax))
    ax.set_xticklabels([int(tmp*1000) for tmp in ax.get_xticks()])
    
    fig.tight_layout()
    fig.savefig(filename(k), dpi=150)

if show_fig:
    plt.show()
