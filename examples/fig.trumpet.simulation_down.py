#! /usr/bin/python
# -*- coding: utf-8 -*-
# Copyright Fabrice Silva, 2012.

"""
"""
from init_scripts import *
path = lambda fname: os.path.join(data_dir, fname)
gpath = lambda fname: os.path.join('..', 'data', fname)

# Acoustic resonator
impedance = 'ImpedanceBaptiste'

if os.path.exists(datafilename(ext='.h5')): # gustavo (25/01/2021 - try não estava funcionando)
     sim = ms.load_simulation(datafilename(ext='.h5')) # ir na pasta e apagar o que foi criado para criar outro
else:

    tsim = 1.
    fs = 44100.
    t = np.linspace(0, tsim, int (tsim * fs))

    #Valve
    wr = mp.Linear([0., tsim], 2. * np.pi * np.array([1000., 100.]))
    qr = mp.Linear([0., tsim], [.1, .1])
    D = mv.LipDynamics(wr=wr, qr=qr, kr=8e8)

    pm = mp.SmoothStep([0.0, .001, tsim - .01, tsim - .005], 20000., 0.)
    h0 = mp.Constant(1e-5) # Channel section at rest (in m**2)
    # Different stiffness wrt pm and p
    # h0 = h0 - pm/7e8

    # if os.path.exists(path(impedance + '_Z.h5')):
    #     Z = mac.load_impedance(path(impedance + '_Z.h5'))
    # else:
    if True:
    
        Zc = mac.rho * mac.c / (np.pi * (8.4e-3)**2)
        Z = mac.MeasuredImpedance(filename=gpath(impedance + '.mat'),
                storage='mat_freqZ', fmin=60., Zc=Zc)

        fapp = [87, 236, 352, 472, 590, 706, 820, 933, 1060, 1184, 1305, 1429]
        Qapp = None
        
        Z.estimate_modal_expansion(algorithm='bruteforce', lFapp=fapp, lQapp=Qapp)
        Z.residues *= Z.Zc
        Z.save(path(impedance + '_Z.h5'))

    sim = ms.TimeDomainSimulation(D, Z, pm=pm, h0=h0, fs=fs, piecewise_constant=True)

    if sim.integrator.successful():
        sim.integrator.set_integrator('lsoda', nsteps=200000)
        sim.integrate(t=tsim)

    sim.save(datafilename(ext='.h5'))
    
    print(sim.extract_signals())

    for where in ('in', 'out'):
        sim.save_wav(datafilename(label=where, ext='.wav'), where=where)

if True:
    # Plot spectrogram
    Figs = sim.trace(fmax=5000., trace_signals=0, trace_spectrogram=1)

    fig = Figs['spectrogram']
    ax = fig.get_axes()[0]
    Pxx = ax.get_images()[0].get_array()
    xlim, ylim, clim = ax.get_xlim(), ax.get_ylim(), ax.get_images()[0].get_clim()
    plt.close('all')

    fig, ax = plt.subplots(1,1, squeeze=True)
    ax.set_xlabel(r'$t\, (s)$')
    ax.set_ylabel(r'$f\,(Hz)$')
    ax.imshow(Pxx[::-1, :], cmap=plt.cm.hot_r, interpolation='none',
        vmax=clim[1], vmin=clim[1]-80, origin='lower',
        extent=xlim+(0., sim.fs/2.), aspect='auto')
    t = sim.time
    for sn in sim.resonator.poles:
        fn = np.abs(sn(t)) / (2. * np.pi)
        L1 = ax.plot(t, fn, 'b', lw=.2, ls='-.')[0]
    fr = sim.valve.wr(t) / (2. * np.pi)
    L2 = ax.plot(t, fr, 'k', lw=.2)[0]
    ax.legend((L1, L2), ("Acoustic resonances", "Lip resonance"),
        loc='lower left')
    ax.set_ylim(0., 1200.)
    plt.tight_layout()
    fig.savefig(figfilename(label='spectro'))

if False:
    # Analyze results, in terms of playability and intonation
    t, freq = sim.get_instantaneous_frequency(mode='yin')

    fr = sim.valve.wr(t) / (2. * np.pi)
    fn = np.abs(sim.resonator.poles(0.))/(2.*np.pi)
    # Getting closest notes in tempered scale
    idx = np.argmin(np.abs(fn[:, np.newaxis]-ts_frequencies[np.newaxis,:]), axis=1)
    ft = ts_frequencies[idx]
    labels = [notes[i + 2] for i in idx] # written pitch for the Bb trumpet

    idx = np.argmin(np.abs(freq[:, np.newaxis] - np.asarray(fn)), axis=1)
    idx_maxp1 = idx.max() + 1
    figs = []
    axss = []
    dev = lambda f1, f2 : 1200. * np.log2(f1 / f2)
    dic = {
        'spectrolike': {
            'xlabel': r'$f_{Lips}$',
            'ylabel': r'$f$',
            'xconst': lambda f, fn, fr, ft: fr,
            'yconst': lambda f, fn, fr, ft: f,
            'axhline': True,
            'axvline': True,
        },
        'deviation': {
            'xlabel': r'$f_{Lips}$',
            'ylabel': r'$dev(f)\,(cents)$',
            'xconst': lambda f, fn, fr, ft: fr,
            'yconst': lambda f, fn, fr, ft: dev(f, ft),
            'axvline': True,
            'kwargs': dict(marker='o', ls=''),
        },
    }
    
    for key, el in dic.items():
        fig, axs = plt.subplots(1, 1, squeeze=True,
            subplot_kw={'xlabel': el['xlabel'], 'ylabel': el['ylabel']})
        figs.append(fig)
        axss.append(axs)
        for i in range(idx_maxp1):
            ind = (idx == i)
            x = el['xconst'](freq[ind], fn[i], fr[ind], ft[i])
            y = el['yconst'](freq[ind], fn[i], fr[ind], ft[i])
            axs.plot(x, y, **el.get('kwargs', {}))
            if len(y) :
                ii = np.nanargmin(y)
                if np.isfinite(ii) and np.isfinite(x[ii]) and np.isfinite(y[ii]):
                    axs.annotate(labels[i], xy=(x[ii], y[ii]),  xycoords='data',
                        xytext=(0, +5), textcoords='offset points',
                        ha='center', va='bottom',
                    arrowprops={'arrowstyle': "-", 'alpha': 0.})
        if el.get('axvline', False):
            axs.set_xticks(fn[:idx_maxp1])
            axs.set_xticklabels([r'$f_{ac}^{%d}$' % (i + 1) \
                                 for i in range(idx_maxp1)])
        if el.get('axhline', False):
            axs.set_yticks(fn[:idx_maxp1])
            axs.set_yticklabels([r'$f_{ac}^{%d}$' % (i + 1) \
                                 for i in range(idx_maxp1)])
            axs.set_ylim(0, fn[idx_maxp1])
        axs.invert_xaxis()
        axs.grid(True)
        fig.tight_layout(pad=1.2)
        fig.savefig(figfilename(label=key))

if show_fig:
    plt.show()
    
