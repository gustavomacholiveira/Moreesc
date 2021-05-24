#! /usr/bin/python
# -*- coding: utf-8 -*-
# Copyright Fabrice Silva, 2012.

"""
Compare intonation on upwards and downwards glissando
"""

from init_scripts import *

# Acoustic resonator
impedance = 'ImpedanceBaptiste'

try:
    su = ms.load_simulation(os.path.join(data_dir, 'fig_trumpet_simulation_up.h5'))
except:
    raise IOError('Run fig.trumpet.simulation_up.py first')
try:
    sd = ms.load_simulation(os.path.join(data_dir, 'fig_trumpet_simulation_down.h5'))
except:
    raise IOError('Run fig.trumpet.simulation_down.py first')

figs = {}
axss = {}
dev = lambda f1, f2 : 1200. * np.log2(f1 / f2)
dic = {
    'spectrolike': {
        'xlabel': r'$f_{Lips}$',
        'ylabel': r'$f$',
        'xconst': lambda f, fn, fr, ft: fr,
        'yconst': lambda f, fn, fr, ft: f,
        'axhline': True,
        'axvline': True,
        'sub': '(a)'
    },
#        'inst2': {
#            'xlabel': r'$f_{Lips}/f_{ac}^{closest}$',
#            'ylabel': r'$f/f_{ac}^{closest}$',
#            'xconst': lambda f, fn, fr, ft: fr/fn,
#            'yconst': lambda f, fn, fr, ft: f/fn,
#        },
#        'inst3': {
#            'xlabel': r'$f/f_{Lips}$',
#            'ylabel': r'$f/f_{ac}^{closest}$',
#            'xconst': lambda f, fn, fr, ft: f/fr,
#            'yconst': lambda f, fn, fr, ft: f/fn,
#        },
    'deviation': {
        'xlabel': r'$f_{Lips}$',
        'ylabel': r'$dev(f)\,(cents)$',
        'xconst': lambda f, fn, fr, ft: fr,
        'yconst': lambda f, fn, fr, ft: dev(f, ft),
        'axvline': True,
        'sub': '(b)'
    },
}


for inds, sim in enumerate((sd, su)):
    # Analyze results, in terms of playability and intonation
    if hasattr(sim, 'f_i'):
        print(inds)
        del sim.f_i
    t, freq = sim.get_instantaneous_frequency(mode='yin', smoothing=3)

    fr = sim.valve.wr(t) / (2. * np.pi)
    fn = np.abs(sim.resonator.poles(0.))/(2.*np.pi)
    # Getting closest notes in tempered scale
    idx = np.argmin(np.abs(fn[:, np.newaxis]-ts_frequencies[np.newaxis,:]), axis=1)
    labels = [notes[i + 2] for i in idx] # written pitch for the Bb trumpet
    ft = ts_frequencies[idx]

    idx = np.argmin(np.abs(freq[:, np.newaxis] - np.asarray(fn)), axis=1)
    idx_maxp1 = idx.max() + 1
    for key, el in dic.items():
        kwargs = el.get('kwargs', {})
        if key in figs.keys():
            fig = figs[key]
            axs = axss[key]
        else:
            fig, axs = plt.subplots(1, 1, squeeze=True,
                subplot_kw={'xlabel': el['xlabel'], 'ylabel': el['ylabel']})
            figs[key] = fig
            axss[key] = axs
        if inds:
            kwargs.update(dict(c='r', ls='.', lw=1.5, marker='.', ms=3))
        else:
            kwargs.update(dict(c='k', ls='.', lw=1.5, marker='+', ms=3.))

        for i in range(idx_maxp1):
            #kwargs['color'] = plt.cm.spectral(i / 8.)
            ind = (idx == i)
            x = el['xconst'](freq[ind], fn[i], fr[ind], ft[i])
            y = el['yconst'](freq[ind], fn[i], fr[ind], ft[i])
            if i == idx_maxp1 - 1:
                kwargs['label'] = ['downwards', 'upwards'][inds]
            axs.plot(x, y, **kwargs)
            kwargs.pop('label', None)
            if inds == 1 and len(x) :
                ii = np.nanargmin(np.abs(x-np.mean(x[np.isfinite(x)])))
                if np.isfinite(ii) and np.isfinite(x[ii]) and np.isfinite(y[ii]):
                    axs.annotate(labels[i], xy=(x[ii], y[ii]),  xycoords='data',
                        xytext=(-15, 5), textcoords='offset points',
                        ha='right', va='center',
                        arrowprops={'arrowstyle': "-", 'alpha': 0.})

        if el.get('sub', False):
            axs.text(.02, .98, el['sub'], ha='left', va='top',
                transform = axs.transAxes)

for key, el in dic.items():
    fig = figs[key]
    axs = axss[key]
    if el.get('axvline', False):
        axs.set_xticks(fn[:idx_maxp1])
        axs.set_xticklabels([r'$f_{ac}^{%d}$' % (i+1) for i in range(idx_maxp1)])
    if el.get('axhline', False):
        axs.set_yticks(fn[:idx_maxp1])
        axs.set_yticklabels([r'$f_{ac}^{%d}$' % (i+1) for i in range(idx_maxp1)])
        axs.set_ylim(0, fn[idx_maxp1])
    #axs.invert_xaxis()
    axs.grid(True)
    if key == 'deviation':
        axs.set_xlim(fn[3], fn[8])
        axs.set_ylim(-150, 150.)
    elif key == 'spectrolike':
        axs.set_xlim(fn[3], fn[8])
        axs.set_ylim(.5 * (fn[3] + fn[4]), .5 * (fn[8] + fn[9]))
        ld = axs.get_lines()[0]
        lu = axs.get_lines()[-1]
    axs.legend(loc='lower right', numpoints=6)
    fig.tight_layout()
    fig.savefig(figfilename(label=key))

if show_fig:
    plt.show()
