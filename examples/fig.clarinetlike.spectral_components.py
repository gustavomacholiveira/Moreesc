#!/usr/bin/python
# -*- coding: utf-8 -*-

#Copyright (C) 2009 Fabricio Silva

from init_scripts import *
trace_image = False
qr = 0.7
simfile = os.path.join(data_dir, 'fig_clarinetlike_reeddamping__qr%.1f' % qr)

try:
    sim = ms.load_simulation(simfile + '.h5')
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
    sim.save(simfile + '.h5')
    sim.save_wav(simfile + '.wav', where='in')

Nmax = min(6, sim.Nac)
fig, axs = plt.subplots(Nmax, 2)#, figsize=(10., 8.))

idx = np.logical_and(sim.time > 0.2, sim.time < 0.45)
time = sim.time[idx]

sigs = np.vstack([sim.pressure[idx], \
    ms.cc.get_acoustic_state_vector(sim.Nac, sim.result[:, idx]).real])
labels = ['$p$',] + ['$p_{%s}$' % tmp for tmp in range(1, 1 + Nmax)]
    
Nfft = int(2 ** np.ceil(np.log2(len(sigs[0]))))
freq = np.fft.fftfreq(Nfft, d=1./sim.fs)[:Nfft/2]
tmp = np.fft.rfft(sigs[1], n=Nfft)

indf = (freq < 2500)
freq = freq[indf] * 1e-3
# Detect fundamental frequency)
idx = np.nonzero(freq>.05)[0][0]
idx = np.argmax(np.abs(tmp[idx:])) + idx
nsamples = sim.fs / (freq[idx] * 1e3)
ylimsp = None

for idx in range(Nmax):
    sig = sigs[idx]
    sig_fft = np.fft.rfft(sig, n=Nfft)[indf]

    ax = axs[idx, 0]
    ax.plot(sig[:nsamples], 'b')

    ax.label_outer()
    ax.set_ylabel(labels[idx])
    tmp = np.abs(ax.get_ylim()).max()*1.05
    ax.set_ylim((-tmp, tmp))
    ax.set_yticks([])
       
    ax = axs[idx, 1]
    ax.plot(freq, 20 * np.log10(np.abs(sig_fft)))

    ax.set_ylabel(labels[idx].upper())
    tmp = ax.get_ylim()
    if ylimsp is None:
        ylimsp = [tmp[0] - 40., tmp[1] + 5.]
    ax.set_ylim(ylimsp)
    ax.set_yticks(ax.get_yticks()[::2])
#    for tmp in ax.yaxis.get_major_ticks():
#        tmp.tick1On, tmp.tick2On = True, True
#        tmp.label1On, tmp.label2On = False, True
        
    ax.yaxis.set_label_position('right')
    ax.yaxis.set_ticks_position('both')
    ax.label_outer()
#    ax.grid(True)
print(ax.get_yticks())
axs[-1, 0].set_xlabel(r'$\varphi$')
axs[-1, 0].set_xticks((0., nsamples-1))
axs[-1, 0].set_xticklabels((r'$0$', r'$2\pi$'))
axs[-1, 1].set_xlabel(r'$f\,(kHz)$')
fig.tight_layout()
fig.savefig(filename(), dpi=150)
if show_fig:
    plt.show()
