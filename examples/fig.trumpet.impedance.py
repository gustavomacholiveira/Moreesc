#! /usr/bin/python
# -*- coding: utf-8 -*-
# Copyright Fabrice Silva, 2012.

"""
"""
from init_scripts import *
path = lambda fname: os.path.join(data_dir, fname)

# Acoustic resonator
impedance = 'ImpedanceBaptiste'

try:
    Z = mac.load_impedance(path(impedance + '_Z.h5'))
except IOError:
    Zc = mac.rho * mac.c / (np.pi * (8.4e-3)**2)
    Z = mac.MeasuredImpedance(filename=os.path.join('..', 'data', impedance + '.mat'),
            storage='mat_freqZ', fmin=60., Zc=Zc)

    fapp = [87, 236, 352, 472, 590, 706, 820, 933, 1060, 1184, 1305, 1429]
    Qapp = None
    
    Z.estimate_modal_expansion(algorithm='bruteforce', lFapp=fapp, lQapp=Qapp)
    Z.residues *= Z.Zc
    Z.save(path(impedance + '_Z.h5'))

figZ, figR, f, Zval = Z.trace(reduced=True)
plt.close(figR)
axs = figZ.get_axes()
colors = 'brkyg'

leg = axs[0].get_legend()
for text in leg.texts:
    if text.get_text().startswith('$t='):
        text.set_text('Modal')
for line, c in zip(leg.get_lines(), colors):
    line.set_c(c)

yticks = [-90., -45., 0., 45., 90.]
axs[1].set_yticks(yticks)
axs[1].set_yticklabels([r'%.0f$^\circ$' % tmp for tmp in yticks])

for ax in axs:
    ax.yaxis.set_label_coords(-.1, 0.5)
    ax.grid(False)
    for l, c in zip(ax.get_lines(), colors):
        l.set_c(c)
figZ.tight_layout()
figZ.savefig(figfilename())

if False:
    idx = np.logical_and(f2 > 60., f2 < 1500.)
    f2, Z2 = f2[idx], Z2[idx] * Z.Zc
    Z = Z(2.j * np.pi * f2)
    div = .5 / (np.abs(Z2) + np.abs(Z))
    eps = (Z2 - Z) * div
    epsa = (np.abs(Z2) - np.abs(Z)) * div

    fig, axs = plt.subplots(3,1, sharex=axs[0])
    axs[0].plot(f2, np.abs(epsa), 'r')
    axs[1].plot(f2, np.abs(eps), 'r')
    axs[2].plot(f2, np.unwrap(np.angle(eps)), 'r')
    axs[1].set_xlim(0, 1500.)

    print("Complex error:", np.mean(eps), np.std(eps))
    print("Modulus error:", np.mean(epsa), np.std(epsa))

if show_fig:
    plt.show()
