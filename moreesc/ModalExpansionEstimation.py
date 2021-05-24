#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
:mod:`ModalExpansionEstimation` -- Tools for modelling real world bores
=======================================================================
.. moduleauthor:: Fabricio Silva <silva@lma.cnrs-mrs.fr.invalid>

"""
import numpy as np
import warnings
from . import AcousticResonator as acous
from . import c_acous_sys as sf
methode = 'Herzog'


def model_lorentz(freq, R1, Fr, Qr, G0):
    """
    Evaluate the model for given parameters at a set of frequencies.

    Parameters
    ----------
    freq : float or array-like
           Frequencies where the Lorentz' resonance model is evaluated
    R1, Fr, Qr, G0 : float
           Coefficients of the model

    Returns
    -------
    Z : array-like
        The values of the model as a frequency response.
    """
    freq = np.asarray(freq)
    return G0+R1/(1.+1.j*Qr*(freq/Fr-Fr/freq))


def modal_func(coefs, freq):
    """
    Evaluates the pole-residue model at a set of frequencies.

    Parameters
    ----------
    freq : float or array-like
           Frequencies where the Lorentz' resonance model is evaluated
    coefs : array-like
           Poles and residus of the modal expansion (4N real values)
           [re(s1)...re(sN) im(s1)...im(sN) re(C1)...re(CN) im(C1)...im(CN)]
    Returns
    -------
    Z : array-like
        The complex values of the model as a frequency response.
    """
    N = len(coefs)/4
    N = int(N)
    if np.any(coefs[:N] > 0):
        warnings.warn('Some pole has a positive real part.')
    l_sn = coefs[0*N:1*N]+1.j*coefs[1*N:2*N]
    l_Cn = coefs[2*N:3*N]+1.j*coefs[3*N:4*N]
    Z = np.empty(freq.shape, dtype=complex)
    return sf.modal_impedance(2.j*np.pi*freq, l_sn, l_Cn, Z)


def modal_jacob(coefs, freq):
    " J[n,m] = d[modal_func(coefs, valf[n])]/dcoefs[m] "
    N = len(coefs)/4
    l_sn = coefs[0*N:1*N]+1.j*coefs[1*N:2*N]
    l_Cn = coefs[2*N:3*N]+1.j*coefs[3*N:4*N]
    dZ, Z = sf.modal_jacb(2.j * np.pi * freq, l_sn, l_Cn)
    return dZ, Z


def modal_jacob_x(coefs, freq):
    " J[n,m] = d[modal_func(coefs, valf[n])]/dvalf[m] "
    N = len(coefs)/4
    jac = np.zeros(len(freq), dtype=complex)
    fun = np.zeros(len(freq), dtype=complex)
    for ind in range(N):
        sn = coefs[0*N+ind]+1.j*coefs[1*N+ind]
        Cn = coefs[2*N+ind]+1.j*coefs[3*N+ind]
        tmp1, tmp2 = (2.j*np.pi*freq-sn), (2.j*np.pi*freq-sn.conj())
        # nth component of Function
        fun += Cn/tmp1+Cn.conj()/tmp2
        jac += Cn/tmp1**2+Cn.conj()/tmp2**2
    jac *= -2.j*np.pi
    return jac, fun

"""
Resonance in a frequency response may be represented by a circle in the complex
plane. This library is based on this assertion, which is exact for single
Lorentz peak:

.. math::

   H(f) = \\frac{R_1}{1+jQ_r\left(f/Fr-Fr/f\\right)}

For a multimodal transfer function, the frequency response is locally
approximated by the sum of a Lorentz peak and a constant residue :math:`G_0`:

.. math::

   f\sim F_r \Rightarrow H(f) \sim G_0+\\frac{R_1}{1+jQ_r\left(f/Fr-Fr/f\\right)}

"""


def single_circle(freq, Z, meth_C=None, meth_f=None, trace=False):
    """
    Estimation of the parameters of a Kennelly's circle representing a
    Lorentz resonance by a geometrical and angle LSQ fit procedure
    from complex valued frequential data.

    .. math::

       H(f) = \\frac{R_1}{1+jQ_r\left(f/Fr-Fr/f\\right)}

    Initial version: 22/6/98, Ph. Herzog

    Parameters
    ----------
    freq, Z : float or array-like
        Data points which will be fitted
    meth_C : str
        A string defining the maximum adjustment option:

        - «hermitian» (to constrain the hermitian property of the transfer function),
        - «aligned» (assuming a common possibly complex factor),
        - «complex»  (without constraints, the maximums of the tranfer function and of the model are supposed equal).

    meth_f : str or float
        A value defining the limitation of the range used in the frequential parametrization of the circle (see comments in code for details).

    Returns
    -------
    R1, Fr, Qr, G0 : float
        Coefficients of the model
    J1, J2 : float
        Indicators of the quality of the fits.
    """
    vecf = np.asarray(freq, dtype=float)
    vecZ = np.asarray(Z, dtype=complex)

    # Ajustement du cercle géométrique par minimisation du critère
    # J = ((Rce-re(Z))**2+(Xce-im(Z))**2-rayon**2).sum()
    s_re, s_im = vecZ.real, vecZ.imag
    nbpt = len(vecf)
    PR1, PR2, PR3 = s_re.sum(), (s_re**2).sum(), (s_re**3).sum()
    PX1, PX2, PX3 = s_im.sum(), (s_im**2).sum(), (s_im**3).sum()
    PK11 = (s_re*s_im).sum()
    PK12, PK21 = (s_re*s_im**2).sum(), (s_re**2*s_im).sum()

    Pr, Px = PR1**2/nbpt-PR2, PX1**2/nbpt-PX2
    P21 = PK21+PX3-(PR2*PX1+PX2*PX1)/nbpt
    P12 = PK12+PR3-(PR1*PX2+PR1*PR2)/nbpt
    P11 = PR1*PX1/nbpt-PK11
    D = 2.0*(Pr*Px-P11**2)

    # Coordonnées du cercle de Kennelly
    Rce, Xce = (P21*P11-P12*Px)/D, (P12*P11-P21*Pr)/D
    C = Rce+1.j*Xce
    rayon = np.sqrt(np.abs(C)**2+(PR2+PX2-2*Rce*PR1-2*Xce*PX1)/nbpt)

    # Exactitude de l'ajustement géométrique
    J1 = np.abs(np.abs(s_re+1.j*s_im-C)/rayon-1).max()

    # Inclinaison du cercle et coefficient de la lorentzienne
    # Plusieurs choix possibles (argument meth_C):
    # - forcer le caractère hermitien (R1 réel) : «Herzog» ou «hermitian»
    # - forcer l'alignement de l'origine, du centre et du «max»
    #   (multiplicateur de l'impédance localement cst) : «aligne»
    # - quelconque : le «max» de la lorentzienne est le module max
    #   sans considération d'influence d'autres pics : «complexes» ou «complex»
    if meth_C is None:
        meth_C = methode
    if meth_C.find('Herzog') > -1 or meth_C.find('hermitian') > -1:
        arg = 0.
    elif meth_C.find('aligne') > -1:
        arg = np.angle(C)
    elif meth_C.find('complex') > -1:
        iMax = np.argmax(np.abs(vecZ))
        arg = np.arg(vecZ[iMax]-C)
    G0 = C-rayon*np.exp(1.j*arg)
    R1 = 2.*(C-G0)

    if trace:
        import matplotlib.pyplot as plt
        plt.figure(20)
        L, = plt.plot(s_re, s_im, 'o')
        Zz = G0 + R1/2. * (1 + np.exp(1.j * np.linspace(0, 2 * np.pi, 256)))
        plt.plot(Zz.real, Zz.imag, c=L.get_c())
        #plt.arrow(G0.real, G0.imag, R1.real, R1.imag, \
        #    ec=L.get_c(), fc=L.get_c())

    # Redressement du cercle en fonction du choix meth_C
    # pour que la paramétrisation fréquentielle fonctionne
    simp = (vecZ-G0)*np.exp(-1.j*arg)

    # Réduction de l'intervalle pour la paramétrisation fréquentielle
    # Plusieurs choix possibles (argument meth_f)
    # - considérer les points à droite du centre : «Herzog»
    # - considérer les points assez loin de l'origine (G_0)
    #   donner une fraction du rayon
    #   ("1" pour les points à un minimum de un rayon de l'origine)
    if meth_f is None:
        meth_f = methode
    if str(meth_f).find('Herzog') == -1:
        try:
            tx = float(meth_f)
        except:
            meth_f = 'Herzog'
            print('Invalid argument meth_f : using "Herzog" method.')
        else:
            idx = (np.abs(simp) > tx * rayon)
    if str(meth_f).find('Herzog') > -1:
        idx = simp.real > rayon

    freq, Val = vecf[idx], simp[idx]
    N = len(freq)
    if N < 2:
        fmoy = vecf.mean()
        tmp = "Can't fit angle curve near %.1fHz:\n" % (fmoy,)
        tmp += "meth_f is too much restrictive: "
        if N == 0:
            tmp += "no point selected."
        else:
            tmp += "only point selected."
        raise ValueError(tmp)
    # Ajustement des paramètres du modèle de phase
    ta2 = Val.imag/Val.real
    F1, F2 = (freq**2).sum(), (1./freq**2).sum()
    T1, T2 = (ta2*freq).sum(), (ta2/freq).sum()
    # Paramètres fréquentiels
    Fr = np.sqrt((N*T1-T2*F1)/(T1*F2-N*T2))
    Qr = np.sqrt((N*T1-T2*F1)*(T1*F2-N*T2))/(F1*F2-N**2)

    # Exactitude de l'ajustement fréquentiel
    #J2 = ((Qr*(freq/Fr-Fr/freq)+ta2)**2).sum()
    J2 = ((Qr*(freq/Fr-Fr/freq)/ta2)).std()
    #J2 = np.abs(ta2[-1]-ta2[0])

    if trace:
        plt.figure(20)
        plt.plot(vecZ.real[idx], vecZ.imag[idx], 'o', c=L.get_c(), mec='y')
        #Zz = G0+rayon*tx*np.exp(.5j*np.pi*np.linspace(-1,1, 256))
        #plt.plot(Zz.real, Zz.imag, '-.', c=L.get_c())

        plt.figure(40)
        plt.plot(freq, -Val.imag / Val.real, 'o', c=L.get_c())
        try:
            Fnr = np.linspace(freq[0], freq[-1], 256)
            plt.plot(Fnr, Qr*(Fnr/Fr-Fr/Fnr), c=L.get_c())
        except:
            pass

    return R1, Fr, Qr, G0, J1, J2


def multiple_circles(freq, Z, lFapp=None, output_snCn=False,
                     meth_C=None, meth_f=None, meth_seq=None, trace=False):
    """
    Estimate the modal expansion of a frequency response with multiple
    resonances.

    Parameters
    ----------
    lFapp : array-like
        List of approximated resonances frequencies. It may be unsorted in order to mention a specific processing order.
    meth_seq : str
        It defines the sequential algorithm used to minimize the side effect of adjacent resonances when estimating a single peak's parameters:

        - if it contains 'eliminate', then the previously estimated resonances are sequentially removed from the analyzed data;
        - if it contains 'automagnitude', approximated guess will be sorted by associated magnitude values, and the algorithm will process resonances by decreasing values.

        Combinations are possible, e.g. for example 'eliminate_automagnitude'
    output_snCn : bool
        A boolean defining whether the model coefficients (False) or the pole-residue pairs (True) are returned.

    For other parameters, see single_circle docstring.

    Returns
    -------
    Zn, Fn, Qn : three arrays
        The coefficient, natural frequency and quality factor of the resonances
    sn, Cn : two arrays
        The estimated poles and the residues of the tranfer function.
    """
    meth_seq = str(meth_seq)
    valZ = np.atleast_1d((1.+0.j)*Z)
    Zn, Fn, Qn = [], [], []

    if lFapp is None or len(lFapp) == 0:
        # Graphical selection of frequencies to process
        lFapp = graphic_fapp(freq, valZ)
        print('Approximate frequencies (in Hz):')
        print(', '.join(['%.0f' % tmp for tmp in lFapp]))
    lFapp = np.array(lFapp)
    tmplFapp = np.atleast_1d(lFapp).tolist()

    while len(tmplFapp) > 0:
        # Sélection séquentielle de la plus forte résonance du résidu ?
        if meth_seq.find('automagnitude') > -1:
            # Évaluation du résidu de l'impédance aux fréquences proches
            # des fréquences approchées, et sélection de la plus forte valeur
            tmp = np.atleast_2d(tmplFapp)
            idx = np.argmin(np.abs(freq-tmp.T), axis=1)
            idx = np.argmax(np.abs(valZ[idx]))
            fapp = tmplFapp.pop(idx)
            del tmp, idx
        else:
            fapp = tmplFapp.pop(0)
        print('Processing resonance near %.1fHz.' % fapp)

        # Restricting the frequency range around the processed frequency
        # Looking for nearest minima (less and greater than fapp)
        # to limit frequency range where a circle is fitted.
        try:
            # Frequencies less than fapp
            fmax = np.max(lFapp[lFapp < fapp])
            imin = np.logical_and(freq > fmax, freq < fapp)
            fminl = freq[imin][np.argmin(np.abs(valZ[imin]))]
            del fmax, imin
        except:
            fminl = np.min(freq)
        try:
            # Frequencies greater than fapp
            fmax = np.min(lFapp[lFapp > fapp])
            imin = np.logical_and(freq < fmax, freq > fapp)
            fminr = freq[imin][np.argmin(np.abs(valZ[imin]))]
            del fmax, imin
        except:
            fminr = np.max(freq)
        # Preventing unwanted huge one-sided frequency range.
        deltaf = np.sqrt((fminr-fapp)*(fapp-fminl))
        #print (fminl, fminr, deltaf)
        idx = np.logical_and(
            freq > max(fminl, fapp - deltaf),
            freq < min(fminr, fapp + deltaf))
        tmp = single_circle(freq[idx], valZ[idx],
                            meth_C=meth_C, meth_f=meth_f, trace=trace)
        Zn.append(tmp[0])
        Fn.append(float(tmp[1]))
        Qn.append(float(tmp[2]))
        if meth_seq.find('eliminate') > -1:
            # Eliminate the last estimated peak
            # (without the residue of fit G0)
            modZ = model_lorentz(freq, tmp[0], tmp[1], tmp[2], 0.)
            valZ -= modZ
        if trace:
            import matplotlib.pyplot as plt
            plt.show()
            plt.close(20)
            plt.close(40)

    if output_snCn:
        # Extraction of poles and residues
        Fn, Qn, Zn = [np.array(el) for el in (Fn, Qn, Zn)]
        poles, residues = acous.wnqnZn2snCn(wn=2.*np.pi*Fn, qn=1./Qn, Zn=Zn)
        return poles, residues
    else:
        return Zn, Fn, Qn


def bruteforce_optimization(freq, Z,
                            lFapp=None, lQapp=None, lZapp=None,
                            output_snCn=False, trace=False,
                            optfun='complex', fmin=None, fmax=None):
    """
    Estimate the modal expansion of a frequency response with multiple
    resonances.

    Parameters
    ----------
    lFapp : array-like
        List of approximated resonances frequencies (open GUI if none).
    lZapp : array-like
        List of approximated resonances magnitudes (local max search if none).
    lQapp : array-like
        List of approximated resonances quality factors (from phase rotation if none).
    output_snCn : bool
        A flag indicating if the result of the optimization must be returned
        in poles-residus values (True) or frequency-damping-magnitude
        values (False, default).
    optfun : str
        You can either optimize upon the complex impedance values (use optfun='complex'):

        .. math::

           \min{J} \quad\mbox{with}\quad J=\sum |\mbox{model} - \mbox{ref}|^2,

        or only on the modulus of impedance (optfun='modulus'):

        .. math::

           \min{J} \quad\mbox{with}\quad J=\sum (|\mbox{model}| - |\mbox{ref}|)^2.

    Returns
    -------
    Zn, Fn, Qn : three arrays
        The coefficient, natural frequency and quality factor of the resonances
    sn, Cn : two arrays
        The estimated poles and the residues of the tranfer function.
    """
    freq = np.array(freq, dtype=float)
    valZ = np.array(Z, dtype=complex)
    flag_Fapp, flag_Zapp, flag_Qapp = False, False, False
    if lFapp is None:
        flag_Fapp = True
        # Graphical selection of frequencies to process
        lFapp = graphic_fapp(freq, valZ)
    lFapp = np.asanyarray(lFapp, dtype=float).ravel()

    if lZapp is None:
        flag_Zapp = True
        lZapp = np.empty(lFapp.shape, dtype=complex)
    lZapp = np.asanyarray(lZapp, dtype=complex).ravel()

    if lQapp is None:
        flag_Qapp = True
        lQapp = np.empty_like(lFapp)
    lQapp = np.asanyarray(lQapp, dtype=float).ravel()

    if flag_Zapp or flag_Qapp or flag_Fapp:
        # If lQapp and lZapp are not provided, a raw estimate is computed
        # lFapp is refined by looking for the maximum of the windowed data.
        tmp = np.r_[freq.min(), lFapp, freq.max()]
        for ind in range(len(lFapp)):
            # Windowing : distorted Hanning window
            idx1 = np.logical_and(freq >= tmp[ind], freq <= tmp[ind + 1])
            idx2 = np.logical_and(freq > tmp[ind + 1], freq < tmp[ind + 2])
            idxw = idx1+idx2
            N1, N2 = idx1.sum(), idx2.sum()
            win = np.empty(N1+N2)
            win[:N1] = np.sin(np.pi/2.*np.arange(N1)/(N1-1))
            win[N1:] = np.sin(np.pi/2.*np.arange(N2)[::-1]/(N2-1))
            # Searching the maximum of windowed data
            imax = np.argmax(win*np.abs(valZ[idxw]))
            if flag_Fapp:
                lFapp[ind] = freq[idxw][imax]
            if flag_Zapp:
                lZapp[ind] = valZ[idxw][imax]
            if flag_Qapp:
                # Raw estimation of quality factor : fitting phase shift
                NbPt = 5
                idxn = np.arange(max(0, imax-NbPt), min(N1+N2, imax+NbPt+1))
                if optfun == 'complex':
                    tmpf, tmpZ = freq[idxw][idxn], np.angle(valZ[idxw][idxn])
                    N, Sx, Sxx = len(idxn), tmpf.sum(), (tmpf**2).sum()
                    Sy, Sxy = tmpZ.sum(), (tmpf * tmpZ).sum()
                    P = np.polyfit(tmpf, tmpZ, 1)
                    lQapp[ind] = -.5 * lFapp[ind] * P[0]
                elif optfun == 'modulus':
                    tmpf, tmpZ = freq[idxw][idxn], np.abs(valZ[idxw][idxn])
                    P = np.polyfit(tmpf, tmpZ, 2)
                    lFapp[ind] = -.5 * P[1] / P[0]
                    lZapp[ind] = (lFapp[ind] * P[0] + P[1]) * lFapp[ind] + P[2]
                    lQapp[ind] = .5 / np.sqrt(1 - P[2] / P[0] / lFapp[ind] ** 2)

    print("Bruteforce initialization values (Fapp, Qapp, Zapp)")
    for x, y, z in zip(lFapp, lQapp, lZapp):
        print('Fapp=%.1fHz\t Qapp=%.2f\t Zapp=%.1f+j%.1f' % (x, y, z.real, y.imag))
    # 4N-coefficients optimization (each complex mode has 4 coefficients).
    snapp, Cnapp = acous.wnqnZn2snCn(wn=2*np.pi*lFapp, qn=1./lQapp, Zn=lZapp)
    coefs_0 = np.r_[snapp.real, snapp.imag, Cnapp.real, Cnapp.imag]

    if optfun == 'complex':
        # Observations are real and imag parts of the impedance
        data_y = np.array([valZ.real, valZ.imag])

        def func(coefs, x):
            tmp = modal_func(coefs, x)
            return np.array([tmp.real, tmp.imag])

        def jacb(coefs, x):
            tmp = modal_jacob(coefs, x)[0]
            return np.array([tmp.real, tmp.imag])

        def jacd(coefs, x):
            tmp = modal_jacob_x(coefs, x)[0]
            return np.array([tmp.real, tmp.imag])

    elif optfun == 'modulus':
        # Observations are the modulus of the impedance
        data_y = np.abs(valZ)

        def func(coefs, x):
            tmp = modal_func(coefs, x)
            return np.abs(tmp)

        def jacb(coefs, x):
            jac, fun = modal_jacob(coefs, x)
            return np.real(fun/np.abs(fun)*jac)

        def jacd(coefs, x):
            jac, fun = modal_jacob(coefs, x)
            return np.real(fun/np.abs(fun)*jac)

    elif optfun == 'modulusYZ':
        # Observations are the modulus of the impedance and of the admittance
        data_y = np.abs(valZ+1./valZ)

        def func(coefs, x):
            tmp = modal_func(coefs, x)
            return np.abs(tmp)

        def jacb(coefs, x):
            jac, fun = modal_jacob(coefs, x)
            return np.real(fun/np.abs(fun)*jac)

        def jacd(coefs, x):
            jac, fun = modal_jacob(coefs, x)
            return np.real(fun/np.abs(fun)*jac)

    else:
        raise IOError("Can not understand optfun argument.")

    if False:
        # using leastsq from scipy.optimize (Levenberg-Marchand optimization)
        from scipy.optimize import leastsq
        residual = lambda coefs, x: func(coefs, x) - data_y
        jacobian = jacb
        res = leastsq(residual, coefs_0, args=(freq, valZ), Dfun=jacobian,
                      xtol=1e-12, full_output=False)
        if res[1] not in (1, 2, 3, 4):
            return False, res
        if res[1] == 4:
            print("Ununderstandable message about jacobian...")
            return False, res
        print(res[1])
        coefs_opt = res[0]
    else:
        # using odr from scipy.odrpack (trust-region optimization)
        import scipy.odr as odr
        data = odr.Data(freq, data_y)
        model = odr.Model(func)  # , fjacb=jacb, fjacd=jacd)
        syst = odr.ODR(data, model, beta0=coefs_0,
                       partol=1e-3, sstol=1e-5)
        syst.set_iprint(init=0, iter=0, iter_step=0, final=0)
        # syst.set_job(deriv=2)
        # TODO: Problem with the derivatives (not used).
        syst.set_job(fit_type=2)
        output = syst.run()
        coefs_opt = output.beta

    N = len(coefs_opt)/4
    N = int(N)
    sn_opt = coefs_opt[0*N:1*N]+1.j*coefs_opt[1*N:2*N]
    Cn_opt = coefs_opt[2*N:3*N]+1.j*coefs_opt[3*N:4*N]
    print("Post optimization\n", np.c_[sn_opt/(2.*np.pi), Cn_opt])
    import matplotlib.pyplot as plt
    if trace:
        plt.figure(8)
        plt.suptitle('Nyquist plane : position of poles')
        plt.plot(snapp.real, snapp.imag, 'bo')
        plt.plot(sn_opt.real, sn_opt.imag, 'rx')
        plt.figure(9)
        plt.suptitle("Result of optimization")
        plt.plot(freq, np.abs(valZ), 'b',
                 freq, np.abs(modal_func(coefs_opt, freq)), 'r')
        for ind in range(N):
            plt.axvline(np.abs(sn_opt[ind]/2./np.pi), c='k', ls='-.')
    if output_snCn:
        return True, (sn_opt, Cn_opt)
    else:
        wn, qn, Zn = acous.snCn2wnqnZn(sn=sn_opt, Cn=Cn_opt)
        return True, (wn, qn, Zn)


def graphic_fapp(freq, Z):
    """
    Plot the frequency response data to allow the graphical input of
    approximated resonance frequencies.

    Parameters
    ----------
    freq, Z : float or array-like
        Data points which will be fitted

    Returns
    -------
    lFapp : list
        The approximated frequencies the user graphically selected.
    """
    import matplotlib.pyplot as plt
    Fig = plt.figure()
    Fig.clf()
    S1 = Fig.add_subplot(211, ylabel=r'$|Z|$')
    S2 = Fig.add_subplot(212, ylabel=r'$\arg(Z)$', xlabel=r'$f$', sharex=S1)
    L, = S1.plot(freq, np.abs(Z))
    tmp = np.unwrap(np.angle(Z))
    tmp -= np.ceil(np.mean(tmp)/(2.*np.pi))*2.*np.pi
    S2.plot(freq, tmp, c=L.get_c())
    S1.label_outer()

    # Acquisition
    S1.set_title('Left click: add a value - '
                 'Right: remove last point\nmiddle: validate')
    tmp = plt.ginput(n=0, timeout=30)
    x, y = zip(*tmp)
    return x

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.close(1)
    plt.close(20)
    plt.close(40)
    X1 = (10e5, 100., 12., 0.)
    X2 = (1e5, 150., 30., 0.)
    Nbpt = 128
    f = np.linspace(10, 500, Nbpt)
    Z = model_lorentz(f, *X1)+model_lorentz(f, *X2)
    Z += np.random.randn(Nbpt) * np.max(np.abs(Z)) / 100. \
        * np.exp(2.j * np.pi * np.random.rand(Nbpt))
    plt.figure(1)
    plt.plot(f, np.abs(Z), '+', label='Original')
    Zn, Fn, Qn = multiple_circles(f, Z, lFapp=(), meth_f=.3)
    Fn, Qn, Zn = [np.array(el) for el in (Fn, Qn, Zn)]
    print("Reference")
    print(np.array([X1, X2]))
    print("Estimated")
    print(np.array([Zn, Fn, Qn], dtype=float).T)
    f2 = np.linspace(0., f.max(), 16*len(f))
    ind = 0
    Zapp = model_lorentz(f2, Zn[ind], Fn[ind], Qn[ind], 0)

    sn, Cn = acous.wnqnZn2snCn(wn=2.*np.pi*Fn, qn=1./Qn, Zn=Zn)
    Zres = acous.ResonateurModesComplexes(poles=sn, residus=Cn,
                                          dimensionne=True, Zc=1.)

    plt.figure(1)
    plt.plot(f2, np.abs(Zapp), label='Kennelly')
    plt.plot(f2, np.abs(Zres(s=2.j*np.pi*f2)), label='ModesComplexes')
    plt.legend()
    plt.show()
