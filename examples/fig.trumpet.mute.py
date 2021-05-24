#! /usr/bin/python
# -*- coding: utf-8 -*-
# Copyright Fabrice Silva, 2012.

"""
"""
from init_scripts import *

tsim = 2.
nbperiods = 8 # Opening/Closing sequence
try:
    Z = mac.load_impedance(datafilename(label='Z', ext='.h5'))
except:
    try:
        dic = np.load(datafilename(label='snCn', ext='.h5'))
        poles, residues = dic['poles'], dic['residues']
        nmax = len(poles)
    except:
        dic = np.load(os.path.join('..', 'data', 'fig_trumpet_mute_mesP.npz'))
        freq = dic['f']
        dZ = np.asscalar(dic['Z'])
        presets = {
            'sourdine_0': (237., 342, 467, 577, 682, 782, 907, 1027, 1167, 1272),
            'sourdine_1': (231., 347, 457, 577, 687, 792, 917, 1042, 1162, 1277),
            'sourdine_2': (231., 346, 461, 576, 679, 788, 824, 929, 1052, 1175, 1292),
        #   'sourdine_3': (231., 344, 459, 574, 673, 747, 820, 929, 1048, 1171, 1294),
            'sourdine_4': (227., 357, 467, 592, 687, 802, 917, 1047, 1177, 1293),
        }
        if True:
            keys = sorted(presets.keys())
            Z = [None,] * len(keys)
            for indk, k in enumerate(keys[::-1]):
                Z[indk] = mac.MeasuredImpedance(freq, dZ[k], filename=k)
                Z[indk].estimate_modal_expansion( #algorithm='bruteforce', optfun='modulus',
                    fmin=150., fmax=1500., lFapp=presets[k])

        nmax = max([len(z.poles) for z in Z])
        nz = len(Z)
        poles = np.empty((nmax, nz), dtype=complex) + np.nan + 1.j * np.nan
        residues = np.empty_like(poles) + np.nan

        # Setting (sorted) reference
        snref, Cnref  = Z[0].poles, Z[0].residues
        ind = np.argsort(snref.imag)
        poles[:len(snref), 0] = snref[ind]
        residues[:len(snref), 0] = Cnref[ind]
        # Sorting other series to have some sontinuity in poles and residues series
        for iz, z in enumerate(Z):
            if iz == 0:
                continue
            sn, Cn = z.poles, z.residues
            dist = np.abs(sn[:, np.newaxis] - snref[np.newaxis, :])
            indices = np.asarray([True,] * len(sn))
            while np.any(indices):
                idx = np.nanargmin(dist)
                if np.isnan(idx):
                    poles[len(snref):, iz] = sn[indices]
                    residues[len(snref):, iz] = Cn[indices]
                    break
                i, j = np.unravel_index(idx, dist.shape)
                poles[j, iz] = sn[i]
                residues[j, iz] = Cn[i]
                indices[i] = False
                dist[i,:] = np.nan
                dist[:,j] = np.nan
        # Setting residue to 0 for non ubiquituous poles 
        for i in range(len(snref), nmax):
            idx = np.nonzero(np.isfinite(poles[i,:]))[0]
            i0, i1 = idx[0], idx[-1]
            poles[i, :i0] = poles[i, i0]
            residues[i, :i0] = 0.
            poles[i, i1+1:] = poles[i, i1]
            residues[i, i1+1:] = 0.
        np.savez(datafilename(label='snCn', ext='.h5'), poles=poles, residues=residues)

    # Building the time varying resonator
    sn, Cn = [], []
    instants = np.ravel(np.arange(4)[np.newaxis, :]
                        + 3 * np.arange(0, 2 * nbperiods).reshape(-1,1))
    instants = instants * tsim / instants[-1]

    for i in range(nmax):
        prof = mp.Profile()
        prof.instants = instants
        tmp = np.r_[[np.r_[poles[i, ::-1], poles[i,:]],] * nbperiods]
        prof.coefs = tmp.ravel()
        sn.append(prof)

        prof = mp.Profile()
        prof.instants = instants
        tmp = np.r_[[np.r_[residues[i, ::-1], residues[i,:]],] * nbperiods]
        prof.coefs = tmp.ravel()
        Cn.append(prof)

    sn = mp.GroupProfiles(sn)
    Cn = mp.GroupProfiles(Cn)
    Zc = mac.rho * mac.c / (np.pi * (8.4e-3)**2)
    Z = mac.TimeVariantImpedance(sn, Cn, reduced=True, Zc=Zc)
    Z.save(datafilename(label='Z', ext='.h5'))

try:
    sim = ms.load_simulation(datafilename(ext='.h5'))
except:
    wr = mp.Linear([0., .5*tsim, tsim], 2. * np.pi * np.array([500., 400., 500.]))
    D = mv.LipDynamics(wr=2. * np.pi * 500, qr=.1, kr=8e8)
    #D = mv.LipDynamics(wr=wr, qr=.1, kr=8e8)
    pm = mp.SmoothStep([0.0, .001, tsim - .01, tsim - .005], 40000., 0.)
    h0 = mp.Constant(1e-5)
    sim = ms.TimeDomainSimulation(D, Z, pm=pm, h0=h0, fs=44100.,
        piecewise_constant=True)

    if sim.integrator.successful():
        sim.integrator.set_integrator('lsoda', nsteps=200000)
        sim.integrate(t=tsim)

    sim.save(datafilename(ext='.h5'))

    for where in ('in', 'out'):
        sim.save_wav(datafilename(ext='.wav'), where='in')

# X view limits
x0 = tsim - tsim / nbperiods
x1 = tsim

# First fig : deviations
t, f = sim.get_instantaneous_frequency(method='yin', bufsize=1024,
    pitchmax=800, hopsize=64)
#deviate = lambda freq: 1200. * np.log2(freq / np.median(freq))
deviate = lambda freq: 1200. * np.log2(freq / freq[-1])

fig, ax0 = plt.subplots(1, 1, sharex=True)
ax0.plot(t, deviate(f), 'k-.', label=r'$f_{osc}$')
for indn, sn in enumerate(sim.resonator.poles):
    if indn < 2 or indn > 9: continue
    ax0.plot(sim.time[::10], deviate(np.abs(sn(sim.time[::10]))), \
        c=cm.jet(.1 * indn), label=r'$\omega_{%d}$' % indn)
ax0.set_ylabel(r'$dev(f)\,(cents)$')
ax0.set_ylim(-20, 40)
ax0.set_yticks(np.arange(-10, 40, 10))
ax0.legend(ncol=5, loc='lower center', prop={'size':'small'},
    columnspacing=1, frameon=False, borderaxespad=.1)
ax0.set_xlabel(r'$t\,(s)$')

kw = dict(va='center', size='x-small')
ax0.text(x0, 25, "Mute closing\nthe bell", ha='left', **kw)
ax0.text(x1, 25, "Mute closing\nthe bell", ha='right', **kw)
ax0.text(.5 * (x0+x1), 0, "Mute far\nfrom the bell", ha='center', **kw)

# Second figure : fundamental amplitude and spectral centroid frequency
def short_time_analyse(sig, bufsize=2048, hopsize=1024, padded_to=None, fs=None):
    if padded_to is None:
        padded_to = bufsize
    nframes = (sig.size - bufsize) / hopsize + 1
    scf = np.empty(nframes, dtype=float)
    ampl = np.empty(nframes, dtype=float)
    scf[:] = np.nan
    ampl[:] = np.nan
    i0 = 0
    frequency = np.fft.fftfreq(padded_to, d=1./fs)[:padded_to/2 + 1]
    frequency[-1] *= -1.
    for indn in range(nframes):
        signal = sig[i0: i0 + bufsize]
        fft = np.abs(np.fft.rfft(signal, n=padded_to))
        # Statistical estimator
        #scf[indn] = np.sum(frequency * fft) / np.sum(fft)
        
        # Partial-based estimator
        fmax = frequency[5:][np.argmax(fft[5:])]
        an = np.zeros(20)
        fn = np.zeros_like(an)
        for indp in range(an.size):
            tmp = (indp + 1) * fmax
            idxn = (frequency > .8 * tmp) * (frequency < 1.2 * tmp)
            fn[indp] = frequency[idxn][np.argmax(fft[idxn])]
            # Gaussian windowing around partial n
            an[indp] = np.sum(fft *
                              np.exp(-.5 * ((frequency - fn[indp])/20.) **2))
        scf[indn] = np.sum(fn * an)/np.sum(an)
        ampl[indn] = an[0]
            
        i0 += hopsize
    t = (np.arange(nframes) * hopsize + bufsize *.5) / fs
    return t, {'scf': scf, 'a1': ampl}

bufsize = 4096
t2, res = short_time_analyse(sim.pressure, bufsize=bufsize, hopsize=64,
    fs=sim.fs)
f2, a2 = res['scf'], res['a1']

fig2, (ax2, ax1) = plt.subplots(2, 1, sharex=True)
ax2.plot(t2, 20. * np.log10(a2 / np.nanmax(a2)), 'b')
ax2.set_ylabel(r'Amplitude ($dB$)')
ax2.set_ylim(-5, .2)
#ax2.set_yticks(range(-3, 1, 2))

ax1.plot(t2, f2 * 1e-3, 'b')
ax1.set_xlabel(r'$t\,(s)$')
ax1.set_ylabel(r'$SCF\,(kHz)$')
ax1.set_ylim(1.6, 2.1)
#ax1.set_yticks(range(1.6, 2.1, .2))

for ax in (ax0, ax1, ax2):
    ax.set_xlim(x0 - .1, x1 + .1)
    ax.set_xticks(np.linspace(x0, x1, 3))

fig.tight_layout()
fig.savefig(figfilename(label="dev"))

fig2.tight_layout()
fig2.savefig(figfilename())
if show_fig:
    plt.show()
