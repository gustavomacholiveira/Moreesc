#!/usr/bin/python
# -*- coding: utf-8 -*-

#Copyright (C) 2012 Fabricio Silva
from init_scripts import *

tsim, fs = .16, 44100.
zeta, gamma = 0.4, 1.15
nfft = 8192
spectro = False
ncols = 1 + spectro

try:
    sim = ms.load_simulation(datafilename(ext='.h5'))
except IOError:
    D = mv.ReedDynamics(wr=2*np.pi*1500., qr=.4, kr=8e8)
    Z = mac.Cylinder(L=.5, r=7e-3, radiates=False, nbmodes=8)

    h0 = .5 * D.stiffness() * mac.rho * (zeta / Z.Zc) ** 2
    Pm0 = gamma * D.stiffness() * h0
    H0 = mp.Constant(h0)

    # Measured mouth pressure profile with strong oscillation at 139.5Hz
    # using comb filter to remove oscillations
    import scikits.audiolab as au
    import scipy.signal as ss
    sig, fs2, enc = au.wavread('../data/Profil_Pbouche.wav')
    tmp = sig.copy()
    delay = int(np.round(fs2 / 139.5))
    b, a = [1. / delay,] * delay, [1,]
    sig = np.r_[np.zeros(delay/2), sig, sig[-1] * np.ones(delay/2)]
    sig = ss.lfilter(b, a, sig)[delay:-delay]
    t2 = np.arange(len(sig), dtype=float) / (100. * fs)
    idx = t2 < .0065
    t2 = t2[idx]
    Pm = mp.Signal(t2, sig[idx] * Pm0/sig[idx][-1], smoothness=.1)
#    Pm = mp.Signal(t2, sig[idx] * Pm0/sig.max(), smoothness=.1)
    Pm.coefs[Pm.coefs < 0.] = 0.
    Pm.instants += .002
    sim = ms.TimeDomainSimulation(D, Z, pm=Pm, h0=H0, fs=fs,
        piecewise_constant=False)
    sim.set_integrator('lsoda', nsteps=20000)
    sim.integrate(t=tsim)
    if not(sim.integrator.successful()):
        plt.close(fig)
        raise ValueError("Simulation has not succeeded...")
    sim.save(datafilename(ext='.h5'))
    sim.save_wav(datafilename(ext='.wav'), where='in')

# Mouth pressure profile
Pm = sim.get_mouth_pressure()
fig = plt.figure()
plt.xlabel(r'$t\,(ms)$')
plt.ylabel(r'$P_m\,(kPa)$')
t2 = sim.time
idx = t2 < 1e-2
plt.plot(t2[idx] * 1e3, Pm[idx] * 1e-3)
fig.set_size_inches(6., 3.)
plt.tight_layout()
fig.savefig(figfilename('profile'))

# Pressure components
fig = plt.figure(figsize=(6.,6.))
Nmax = min(8, sim.Nac)
idx_time = sim.time < tsim
time = sim.time[idx_time]
tsim = .15
Xa = ms.cc.get_acoustic_state_vector(sim.Nac, sim.result).real

ax0 = fig.add_subplot(Nmax, ncols, 1)
ax0.set_ylabel(r'$p$') #' + '\n' + r'$(kPa)$', ha='center')
ax0.yaxis.labelpad += .05
ax0.plot(time, sim.pressure[idx_time], 'b')
ax0.plot(time, sim.pm[idx_time], 'r')

nfft = int(2 ** np.ceil(np.log2(len(time))))
freq = np.fft.fftfreq(nfft, d=1./ sim.fs)
env = []

import scipy.signal as ss

def get_envelop_filt(t, signal):
    fmax = freq[np.argmax(np.abs(np.fft.rfft(pn, nfft)))]
    bands = (0, .05 * fmax / sim.fs, .5 * fmax / sim.fs, .5)
    order, wn = ss.cheb2ord(bands[1], bands[2], -10., -120.)
    b, a = ss.cheby2(order, 10., wn)
    sig_env = ss.lfilter(b, a, np.abs(pn))
    sig_env = ss.lfilter(b, a, sig_env)
    return t[len(b):], sig_env[len(b):]

def get_envelop(t, signal):
    tmp = np.abs(signal)
    idx_max = np.r_[False, (tmp[1:-1] >= tmp[:-2]) * (tmp[1:-1] > tmp[2:]), False]
    return t[idx_max], tmp[idx_max]

for indn in range(Nmax - 1):
    ax = fig.add_subplot(Nmax, ncols, ncols * (indn + 1) + 1,
        ylabel='$p_{%d}$' % (indn + 1), sharex=ax0)
    pn = 2. * Xa[indn, idx_time]
    t, e = get_envelop(time, pn)
    ax.plot(time, pn, 'b', lw=.1)
    ax.plot(t, +e, 'k', lw=1.)
    ax.plot(t, -e, 'k', lw=1.)
    env.append(e)
    

for ax in fig.get_axes():
    ax.grid(True)
    tmp = np.abs(ax.get_ylim()).max()*1.05
    ax.set_yticks([0, tmp])
    ax.set_yticklabels([])
#    ax.set_yticklabels(['-%.2f'% (tmp,), '0', '+%.2f'% (tmp,)])
#    ax.set_yticks([])
    ax.yaxis.get_label().set_rotation('horizontal')
    ax.label_outer()
ax.set_xlim(0., tsim)
from matplotlib.ticker import FuncFormatter
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: '%.0f' % (x*1e3)))
ax.set_xlabel(r'$t\,(ms)$')

if spectro:
    # Spectrogram
    fmin, fmax = 0., 3000.
    tmp = np.r_[np.zeros(nfft), sim.pressure, np.zeros(nfft)]
    Pxx, fr, bins = mlab.specgram(tmp, NFFT=nfft, Fs=sim.fs, \
        window=np.hanning(nfft), noverlap=int(.98 * nfft))
    idx = np.logical_and(fr < fmax, fr > fmin)
    Pxx = 10. * np.log10(Pxx[idx, :] + 1e-9 * Pxx.max())
    tmp = Pxx[np.isfinite(Pxx)]
    vmax = tmp.max()

    ax = fig.add_subplot(1, 2, 2, xlabel=r'$t\,(s)$', ylabel=r'$f\,(kHz)$')
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

fig.tight_layout()
fig.savefig(filename(), dpi=150)

if show_fig:
    plt.show()
