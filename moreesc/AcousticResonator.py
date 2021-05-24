#!/usr/bin/python
# -*- coding: utf-8 -*-
#Copyright (C) 2007 Fabricio Silva

"""
:mod:`AcousticResonator` -- Specifiyng the acoustical resonator
===============================================================
.. moduleauthor:: Fabricio Silva <silva@lma.cnrs-mrs.fr.invalid>

This module provides the class objects useful for the modal description of
the acoustic resonator. The base class for the definition is the
:class:`AcousticResonator.Impedance` described by its complex modes.
Instances are constructed using numerical values or Profile parametrization
of poles and residues..

.. math::

   Z(s=\\alpha+j\omega) = \sum_{n=1}^N \\frac{C_n}{s-s_n}
   +\\frac{C_n^*}{s-s_n^*}

usually evaluated on the frequency axis

.. math::

   Z(\omega) = \sum_{n=1}^N \\frac{C_n}{j\omega-s_n}
   +\\frac{C_n^*}{j\omega-s_n^*}.


.. warning::

   Only the positive frequency poles (i.e. the poles having positive imaginary
   part) have to be handled, as Hermitian symmetry is assumed.

"""

import numpy as np
from . import Profiles
from . import utils
from .c_acous_sys import acous_sys, acous_sys_profiles, modal_impedance

#: Coefficients of the approximated radiation impedance
#: for an unflanged cylinder
radiation_a = (0.800, 0.266, 0.0263, .5)
radiation_b = (0.6133, 0.0599, 0.238, -0.0153, .00150)

#: Wave speed in free space (:math:`m/s`)
c = 346.19
#: Density of the air (:math:`kg/m^3`)
rho = 1.1851
#: Shear coefficient (:math:`kg/m/s`)
nu = 1.831e-5
#: Viscous boundary layer thickness (:math:`m`)
lv = 4.463e-8
#: Thermal conductivity (:math:`Cal/(m.s.^\circ C)`)
kappa = 6.24
#: Thermal boundary layer thickness (:math:`m`)
lt = 6.34e-8
#: Specific heat under constant pressure (:math:`Cal/(kg.^\circ C)`)
Cp = 240.
#: Specific heat at constant volume (:math:`Cal/(kg.^\circ C)`)
Cv = 171.
#: Ratio of the specific heats :math:`C_p/C_v`
Cpv = 1.402
#: Reference temperature 0^\circ C"
Temp0 = 273.16


def physical_constants(Temp=298.):
    """
    Updates the acoustical constants according to the specified temperature.

    Parameters
    ----------
    Temp : float
          The experiment temperature (in Kelvin).
    """
    global rho, c, nu, lv, lt, Cp, Cv, Cpv, kappa, lt
    rho = 1.2929 * Temp0 / Temp
    c = 331.45 * np.sqrt(Temp / Temp0)
    nu = 1.708e-5 * (1 + .0029 * (Temp - Temp0))
    lv = nu / (rho * c)
    Cp = 0.24 * 1000.
    Cpv = 1.402
    Cv = Cp / Cpv
    kappa = 5.77e-5 * (1 + .0033 * (Temp - Temp0)) * 100.
    lt = kappa / (rho * c * Cp)

physical_constants(Temp=298.)

raman = False
debug = False


class TimeVariantImpedance(object):
    """
    A general class to represent the input impedance of a acoustic resonator.

    Attributes
    ----------
    Zc : Profile
      Characteristic impedance of the resonator (used for reduced residues).
    poles : GroupProfiles
      Poles :math:`s_n` of the input impedance.
    residus : GroupProfiles
      Residues :math:`C_n` associated to the  poles.
    nbmodal : integer
      Number of modes in the instance.

    """

    def __init__(self, sn=None, Cn=None, reduced=False, Zc=None):
        """

        Parameters
        ----------
        sn: array or :class:`Profiles.GroupProfiles`
          Poles of the acoustic resonator. If it is time-invariant, a list
          or array of N complex values (one per resonance) is sufficient.
          If not, provide a:class:`Profiles.GroupProfiles` instance
          (possible complex-valued).
        Cn: array or :class:`Profiles.GroupProfiles`
          Residue of the input impedance of the acoustic resonator. See
          :attr:`sn` for explanation on format.
        Zc: float or Profile
          The (possibly time-variant) characteristics impedance of the
          acoustic resonator. Optional if the residues are not dimensionless,
          as it is then only used to compute reflection coefficient.
        reduced: bool
          Flag indicating whether residue value are dimensionless or not. If
          True, value of the characteristics impedance is mandatory and residus
          are then dimensioned according to Zc.

        """

        # Handling dimensionnality of residues and characteristics impedance.
        isReduced = bool(reduced)
        if Zc is None:
            if isReduced:
                raise IOError('Zc needed if the residues are dimensionless.')
            Zc = 1.
        self.Zc = Profiles.toProfile(Zc)
        # Transformation of poles and residus
        self.poles = Profiles.GroupProfiles(sn)
        self.residues = Profiles.GroupProfiles(Cn)
        for el in (self.poles, self.residues):
            el.coefs_array = np.asanyarray(el.coefs_array, dtype=complex)
        if len(self.poles) != len(self.residues):
            raise IOError(
                "Lengths of poles (%d) and residues (%d) not matching."
                % (len(self.poles), len(self.residues)))
        self.nbmodes = len(self.poles)

        if isReduced:
            self.residues *= self.Zc

    def __str__(self, t=0.):
        tmp = '%s at %s (%d modes)' % (self.__class__, id(self), self.nbmodes)
        Nmax = min(40, self.nbmodes)
        t_damping = - 1. / np.asarray(self.poles(t)).real
        frequencies = np.asarray(self.poles(t)).imag / (2. * np.pi)
        for indn in range(Nmax):
            tmp += 'Mode %d:\t%.1f Hz\t%.3f ms\n' % \
                (indn, frequencies[indn], t_damping[indn])
        if self.nbmodes > Nmax:
            tmp += '...\n'
            tmp += 'Last Mode:\t%.1f Hz\t%.3f ms\n' % \
                (frequencies[-1], 1000. * t_damping[-1])
        return tmp

    def __eq__(self, other):
        return isinstance(other, TimeVariantImpedance) \
            and self.nbmodes == other.nbmodes \
            and self.Zc == other.Zc \
            and self.poles == other.poles \
            and self.residues == other.residues

    def __ne__(self, other):
        return not(self == other)

    def __instantaneous_operator(self, func, t):
        " Private method to apply function for several time values. "
        sn = np.atleast_2d(np.r_[self.poles(t)].T)
        Cn = np.atleast_2d(np.r_[self.residues(t)].T)
        Zc = np.atleast_1d(self.Zc(t))
        result = [func(*args) for args in zip(sn, Cn, Zc)]
        if np.isscalar(t):
            return result[0]
        return result

    def __call__(self, s, t=[0., ]):
        """
        Evaluates the expression :math:`Z(s)` for the specified values of
        :keyword:`s`. The evaluation is done using the compiled function
        modal_impedance.

        Parameters
        ----------
        s : array-like
            List of values of Laplace variable at which transfer function
            is evaluated. If purely real valued, it is interpreted as
            a array of frequencies (and thus multiplied by :math:`2\pi`)
        t : scalar, optional
            Instant to consider when evaluating the coefficients of the
            transfer function. Default is 0.

        Returns
        -------
        Z : list of frequency responses (one per instant required).
        """
        if not(np.iscomplexobj(s)):
            print("""
                When calling a TransferFunction instance, argument must be
                given in terms of Laplace variable s. It is assumed real
                frequencies have been provided here.
                """)
            s = 2.j * np.pi * np.array(s)
        else:
            s = np.asanyarray(s)

        Z = np.empty(s.shape, dtype=complex)
        def func(sn, Cn, Zc):
            return modal_impedance(s, sn, Cn, Z)
        return self.__instantaneous_operator(func, t)

    def trace(self, f=None, figs=None, linlog='lin', t=[0., ], reduced=False):
        """
        Plots representations of the frequency response. The graphics
        properties can be modified directly on the returned object.

        Parameters
        ----------
        f : array-like, optional
            Frequencies at which frequency response is evaluated.
        figs : list, optional
            list of two Figure instances in which Z and R are shown.
        linlog : string, optional
            The modulus is displayed with a linear axis if ``lin``, with a
            logarithmic axis if ``log``, and an exception is raised otherwise.
        t : scalar, optional
            Instant to consider when evaluating the coefficients of the
            transfer function. Default is 0.

        Returns
        -------
        Fig : :class:`matplotlib.Figure`
            The figure object

        """
        import matplotlib.pyplot as plt
        if figs is None:
            FigZ = plt.figure()
            FigR = plt.figure()
        else:
            FigZ, FigR = figs
        absZ = FigZ.add_subplot(211)
        argZ = FigZ.add_subplot(212, sharex=absZ)
        absR = FigR.add_subplot(211, sharex=absZ)
        argR = FigR.add_subplot(212, sharex=absR)

        Zdata, Rdata = [], []
        for valt in t:
            fres = np.atleast_1d(self.poles(valt)).imag / (2 * np.pi)
            NbModes = len(fres[fres > 0.])
            if not(f is None):
                freq = np.asarray(f, dtype=float)
            elif linlog == 'lin':
                fmin = 0.1 * np.min(np.abs(fres))
                if NbModes != 1:
                    fmax = 1.5 * max(abs(fres))
                else:
                    fmax = 3 * max(abs(fres))
                freq = np.linspace(fmin, fmax, len(fres) * 200)
            elif linlog == 'log':
                eps = 1e-24
                logfmin = np.floor(np.log10(max(eps, np.min(fres))))
                logfmax = np.ceil(np.log10(max(eps, np.max(fres))))

                if np.isnan(logfmin):
                    logfmin = logfmax - 3
                if np.isnan(logfmax):
                    logfmax = logfmin + 3
                freq = np.logspace(logfmin, logfmax, 200)
            else:
                raise ValueError("linlog argument should be 'lin' or 'log'.")

            Zdata_t = self(2.j * np.pi * freq, t=[valt, ])[0]
            Zc = self.Zc(valt)
            Rdata_t = (Zdata_t - Zc) / (Zdata_t + Zc)
            if reduced:
                Zdata_t = Zdata_t / Zc

            Zdata.append(Zdata_t)
            Rdata.append(Rdata_t)

            Mag = np.abs(Zdata_t)
            Arg = np.angle(Zdata_t) * 180.0 / np.pi
            MagR = np.abs(Rdata_t)
            ArgR = np.unwrap(np.angle(Rdata_t)) * 180.0 / np.pi

            if linlog == 'lin':
                absZ.plot(freq, Mag)
                argZ.plot(freq, Arg)
                absR.plot(freq, MagR)
                argR.plot(freq, ArgR)
            else:
                absZ.semilogx(freq, 20.0 * np.log10(Mag))
                argZ.semilogx(freq, Arg)
                absR.semilogx(freq, MagR)
                argR.semilogx(freq, ArgR)
            absZ.get_lines()[-1].set_label(r'$t=%.3f$' % valt)

        absZ.set_ylabel(r'$|\,Z\,|$')
        absR.set_ylabel(r'$|\,R\,|$')
        argZ.set_ylabel(r'$\arg{(Z)}$', va='bottom')
        argR.set_ylabel(r'$\arg{(R)}$', va='bottom')
        argZ.set_yticks(np.arange(-120, 120, 30))
        for ax in (argZ, argR):
            ax.grid('on')
            ax.set_xlabel(r'$f$ (Hz)')
        for ax in (absZ, absR):
            ax.grid('on')
            ax.label_outer()
            ax.legend()
        return FigZ, FigR, freq, Zdata

    def __getstate__(self):
        return {'Zc': self.Zc, 'nbmodes': self.nbmodes,
            'poles': self.poles, 'residues': self.residues}

    def __setstate__(self, dic):
        self.__dict__.update(dic)

    def save(self, filename):
        """
        Saves instance to file using pickle [pick]_.
        See :func:`load_impedance` for loading.
        """
        utils._pickle(self, filename)

load_impedance = lambda s: utils._unpickle(s)
load_impedance.__doc__ = utils.__pickle_common_doc \
     % {'class': 'Time(Inv|V)ariantImpedance',
        'output': 'TimeInvariantImpedance or TimeVariantImpedance'}
load_impedance.__name__ = 'load_impedance'


class TimeInvariantImpedance(object):
    """
    Simplified class for time invariant acoustic resonator.

    Attributes
    ----------
    Zc : float
      Characteristic impedance (required for reduced residues).
    poles : complex array
      Poles :math:`s_n`
    residus : complex array
      Residues :math:`C_n` associated to the  poles
    nbmodal : integer
      Number of positives modes in the instance.

    """
    _as_ltv = None

    def __init__(self, sn=None, Cn=None, reduced=False, Zc=None):
        """

        Parameters
        ----------
        sn: array
          Poles of the acoustic resonator.
        Cn: array
          Residue of the input impedance of the acoustic resonator.
        Zc: float
          The characteristics impedance of the acoustic resonator.
          Optional if the residues are not dimensionless,
          as it is then only use to compute reflection coefficient.
        reduced: bool
          Flag indicating whether residue value are dimensionless or not. If
          True, value of the characteristics impedance is mandatory and residus
          are then dimensioned according to Zc.

        """
        #Handling dimensionnality of residues and characteristics impedance.
        isReduced = bool(reduced)
        if Zc is None:
            if isReduced:
                raise IOError('Zc needed if the residues are dimensionless.')
            Zc = 1.
        self.Zc = float(Zc)

        # Transformation of poles and residus
        self.poles, self.residues = [
            np.atleast_1d(np.asanyarray(el, dtype=complex).copy())
            for el in (sn, Cn)]
        if len(self.residues) != len(self.poles):
            raise IOError(
                "Lengths of poles (%d) and residues (%d) not matching."
                % (len(self.poles), len(self.residues)))
        self.nbmodes = len(self.poles)
        if isReduced:
            self.residues *= self.Zc

    def __eq__(self, other):
        return isinstance(other, TimeInvariantImpedance) and \
            self.nbmodes == other.nbmodes and \
            np.abs(self.Zc - other.Zc) < 1e-8 * max(self.Zc, other.Zc) and \
            np.all(self.poles == other.poles) and \
            np.all(self.residues == other.residues)

    def __ne__(self, other):
        return not(self == other)

    def __call__(self, s):
        """
        Evaluates the expression :math:`Z(s)` for the specified values of
        :keyword:`s`. The evaluation is done using the fortran extension
        modal_impedance_fortran (src/modal_impedance.f in source package).

        Parameters
        ----------
        s : array-like
            List of values of Laplace variable at which transfer function
            is evaluated. If purely real valued, it is interpreted as
            a array of frequencies (and thus multiplied by :math:`2\pi`)

        Returns
        -------
        Z : frequency response.
        """
        if not(np.iscomplexobj(s)):
            print("""
                When calling a TransferFunction instance, argument must be
                given in terms of Laplace variable s. It is assumed real
                frequencies have been provided here.
                """)
            s = 2.j * np.pi * np.array(s)
        else:
            s = np.asanyarray(s)
        Z = np.empty(s.shape, dtype=complex)
        return modal_impedance(s, self.poles, self.residues, Z)

    def as_timevariantimpedance(self):
        obj = TimeVariantImpedance(self.poles, self.residues, False, self.Zc)
        self._as_ltv = obj
        return obj

    def __getattr__(self, key):
        if key in dir(TimeVariantImpedance):
            if getattr(self, "_as_ltv", None) is None:
                self.as_timevariantimpedance()
            return getattr(self._as_ltv, key)
        else:
            raise AttributeError(
                'TimeInvariantImpedance does not have %s attribute.' % (key,))

    def __getstate__(self):
        return {'Zc': self.Zc, 'nbmodes': self.nbmodes,
            'poles': self.poles, 'residues': self.residues}

    def save(self, filename):
        """
        Saves instance to file using pickle [pick]_.
        See :func:`load_impedance` for loading.
        """
        utils._pickle(self, filename)


def wnqnZn2snCn(wn=None, qn=None, Zn=None):
    """
    Converts the tuple (:keyword:`wn`, :keyword:`qn`, :keyword:`Zn`)
    into (:keyword:`sn`, :keyword:`Cn`).

    Parameters
    ----------
    wn : float or array
         Natural angular frequency of the resonance
    qn : float or array
         Damping of the resonance (quality factor: :math:`1/qn`)
    Zn : complex or array-like
         Gain of the resonance (peak magnitude)

    Returns
    -------
    sn : complex array
         Pole (with positive imaginary part)
    Cn : complex array
         Residue
    """
    qn, wn = [np.asarray(np.atleast_1d(tmp), dtype=float) for tmp in (qn, wn)]
    Zn = np.array(np.atleast_1d(Zn), dtype=complex)
    flag = (qn <= 2.)
    sn = (-.5 * qn + 1.j * np.sqrt(1. - qn ** 2 / 4.)) * wn
    Cn = .5 * qn * Zn * wn * (1. + .5j * qn / np.sqrt(1. - qn ** 2 / 4.))

    if not(np.all(flag)):
        print("""
               At least one mode has over-critical damping and is not
               associated with a pair of complex conjugate poles. Only
               really complex modes are returned.
               """)
    return np.asarray(sn)[flag], np.asarray(Cn)[flag]


def snCn2wnqnZn(sn=None, Cn=None):
    """
    Converts the tuple (:keyword:`sn`, :keyword:`Cn`) into
    (:keyword:`wn`, :keyword:`qn`, :keyword:`Zn`).

    Parameters
    ----------
    sn : complex or array-like
         Pole (with positive imaginary part)
    Cn : complex or array-like
         Residue

    Returns
    -------
    wn : array
         Natural angular frequency of the resonance
    qn : array
         Damping of the resonance (quality factor: :math:`1/qn`)
    Zn : complex array
         Gain of the resonance (peak magnitude)
    """
    sn, Cn = [np.asarray(np.atleast_1d(tmp), dtype=complex) for tmp in (sn, Cn)]
    tmp = sn * Cn.conj()
    flag = np.abs(tmp.real / tmp.imag) < 1e-8
    wn = np.asarray(np.abs(sn))[flag]
    qn = np.asarray(2. * np.abs(sn.real / sn))[flag]
    Zn = np.asarray(-Cn.real / sn.real)[flag]

    if not(np.all(flag)):
        print("""
            At least one mode has non-zero value at zero frequency,
            and thus can not be expressed as (wn,qn,Zn) tuple.""")
    return wn, qn, Zn


class Cylinder(TimeInvariantImpedance):
    """
    Model a cylindrical bore, possibly radiating.

    .. warning::

       By now this configuration is static, i.e. some still need to be done
       to enable linear profiles (or more complex ones, like spline) for the
       attributes of the cylindre (i.e. length and radius). If such case are
       wanted, please consider defining an initial cylinder and a final one,
       and selecting the way the poles and residues are meant to evolve between
       these two states. Be aware that interpolated configurations may not
       correspond to the impedance of an intermediate length cylinder...

    Attributes
    ----------
    r : float
      The inner radius of the cylindrical bore (in :math:`m`)
    L : float
      The geometrical length (in :math:`m`)
    radiates : bool
      Boolean indicating whether radiation from the open end is considered.


    """

    def __init__(self, L=1., r=7e-3, radiates=True, nbmodes=10,
        losses="visco-thermal"):
        """
        Convenient class to define a cylindrical resonator (radius r, length L,
        nbmodes modes) with possible radiating open end (if radiates==True).
        """
        self.r = float(r)
        self.L = float(L)
        self.radiates = bool(radiates)
        self.nbmodes = int(nbmodes)
        self.losses = str(losses).lower()
        for el in ' -_':
            self.losses = self.losses.replace(el, '')
        if self.losses not in ('viscothermal', 'raman'):
            raise ValueError("Invalid losses specification: %s" % losses)
        self.Zc = rho * c / (np.pi * self.r ** 2)
        self.estimate_poles()
        self.eval_residues()
#        self.poles = Profiles.GroupProfiles(self.poles)
#        self.residues = Profiles.GroupProfiles(self.residues)

    def __getstate__(self):
        tmp = TimeInvariantImpedance.__getstate__(self)
        tmp.update({'L': self.L, 'r': self.r, 'radiates': self.radiates,
            'nbmodes': self.nbmodes, 'Zc': self.Zc})
        return tmp

    def __repr__(self):
        return "Analytically defined: L=%.1em and r=%.1em (%d modes) %s" % \
            (self.L, self.r, self.nbmodes,
            {True: 'radiating', False: 'without radiation'}[self.radiates])

    def gamma(self, s, derive=False):
        """
        wave propagation constant
        G(s) = s/c + beta1*np.sqrt(s/c)
        where beta1 accounts for visco-thermic effects.

        """
        # Frequency-dependent losses
        beta1 = (np.sqrt(lv) + (Cpv - 1.0) * np.sqrt(lt)) / self.r
        if self.losses == 'viscothermal':
            beta2 = 0.0
        elif self.losses == 'raman':
            # Constant losses
            beta2 = beta1 * np.sqrt(1.j * np.pi / (2. * self.L))
            beta1 = 0.
        Gs = s / c + beta1 * np.sqrt(s / c) + beta2
        if derive:
            dGs = 1. / c + beta1 * 0.5 / np.sqrt(s * c)
            return Gs, dGs
        else:
            return Gs

    def eta_r(self, s, derive=False):
        """
        Radiation output impedance
        Compute eta_r as defined by Z_r = Z_c coth(eta_r)
        as a function of s=jw with convention exp(+jwt).
        R_r = -|R_r| exp(-2jkr*L/r) = exp(-2 eta_r)
        """
        if self.radiates:
            a, b = radiation_a, radiation_b
            # X=sr/c = jkr -> X**2 = -(kr)**2
            X = s * self.r / c
            X2 = X ** 2
            Rden = 1. - (a[3] + a[0]) * X2 + a[1] * X2 ** 2 - a[2] * X2 ** 3
            Rrabs = (1. - a[0] * X2) / Rden
            Lrden = 1. - b[2] * X2 + b[3] * X2 ** 2 - b[4] * X2 ** 3
            Lr = b[0] * (1 - b[1] * X2) / Lrden

            etar = -.5 * np.log(Rrabs) - 1.j * np.pi / 2. + X * Lr
            #Rr = -Rrabs*np.exp(-2*X*Lr)
            #etar = -0.5*(np.log(np.abs(Rr))-1.j*np.angle(Rr))

            if derive:
                # If X = sr/c then
                # deta_r/ds = r/c*R_r(s)
                #    *(1/|R_r|*d|R_r|/dX-2(L/r+X*d(L/r)/dX))
                dRrabs = -2. * X * (-a[3] + 2. * a[1] * X2
                    - (3. * a[2] + a[0] * a[1]) * X2**2
                    + 2. * a[0] * a[2] * X2**3)
                dRrabs /= Rden ** 2

                dLr = -2. * b[0] * X * (
                        b[1] - b[2] + 2 * b[3] * X2
                        - (b[1] * b[3] + 3 * b[4]) * X2 ** 2
                        + 2 * b[1] * b[4] * X2 ** 3)
                dLr /= Lrden ** 2

                detar = self.r / c * (-.5 * dRrabs / Rrabs + Lr + X * dLr)

        else:
            etar = -.5j * np.pi + np.zeros_like(s)
            detar = np.zeros_like(s)

        if derive:
            return etar, detar
        else:
            return etar

    def fct_poles(self, X, zero):
        """
        Looking for roots of sinh(Gamma(s)L+eta_r(s)).
        i.e. roots of fct(s) = Gamma(s)L+eta_r(s)-j n pi.
        """
        s = X[0] + 1.j * X[1]
        fct = self.gamma(s) * self.L + self.eta_r(s) - zero
        return fct.real, fct.imag

    def estimate_poles(self, S0=None):
        """
        Try to locate the poles in the :math:`s` plane. It uses
        :func:`scipy.optimize.fsolve` which needs initial guess.

        Parameters
        ----------
        S0 : array, optional
          Initial guess for poles. If not specified, the default are the poles
          of the open-closed lossless cylindrical bore.

        Raises
        ------
        ValueError
          When the root finding is not successfull for one of the poles.
        """
        if S0 is None:
            tmp = np.arange(self.nbmodes)
            # Step 1: Neumann/Dirichlet boundary conditions
            S0 = 1.j * np.pi * (tmp + .5) * c / self.L
            # Step 2: Neumann/Radiation boundary conditions
            S0 += c / self.L * (1.j * np.pi * tmp - self.eta_r(S0)
                                - self.gamma(S0) * self.L)

        from scipy.optimize.minpack import fsolve
        self.poles = np.zeros(S0.shape, dtype=complex)
        for ind, val in enumerate(S0):
            pole, dic, ier, msg = fsolve(
                self.fct_poles,
                [val.real, val.imag],
                1.j * ind * np.pi,
                xtol=1e-11, full_output=True)

            if ier == 1 and pole[1] > 1e-8:
                self.poles[ind] = complex(pole[0], pole[1])
            else:
                raise ValueError('Problem with pole %d' % ind)
        idx = np.argsort(self.poles.imag)
        self.poles = self.poles[idx]

    def eval_residues(self, x=0., xs=0., poles=None):
        """
        Evaluates the residues of the previously estimated poles. Analytical
        expression of the residues is given in [Silva:PhD]_.

        Parameters
        ----------
        x, xs : float, optional
          positions of the observer and of the source. Default are 0,
          so that it computes the input impedance.
        poles : array, optional
          If not specified, the residus associated to each pole are computed.

        Returns
        -------
        residues : optional
          If poles are selected, the methods outputs the associated residus.
        """
        if poles is None:
            poles = self.poles
            flag = False
        else:
            poles = np.atleast_1d(poles)
            flag = True
        residus = np.zeros_like(poles)
        for ind, sn in enumerate(poles):
            etar, detar = self.eta_r(sn, derive=True)
            gs, dgs = self.gamma(sn, derive=True)
            residus[ind] = self.Zc * np.cosh(gs * x) * np.cosh(gs * xs)
            residus[ind] /= dgs * self.L + detar
        if flag:
            return residus
        else:
            self.residues = residus

    def eval_analytically(self, s, x=0., xs=0.):
        """
        Evaluates the transfer function between :math:`U(x_s)` and :math:`P(x)`
        using the compact analytical expression.

        Parameters
        ----------
        s : complex array
          Values where to evaluate the transfer function.
        x, xs : float, optional
          Position of the pressure observer and of the volume flow source.
          Default are 0 to get the input impedance.

        Returns
        -------
        Z : complex array
          The values of :math:`Z(s)`.
        """
        X = np.sort([x, xs])
        g = np.cosh(self.gamma(s) * X[0])
        g *= np.cosh(self.gamma(s) * (self.L - X[1]) + self.eta_r(s))
        g /= np.sinh(self.gamma(s) * self.L + self.eta_r(s))
        return g * self.Zc


class MeasuredImpedance(TimeInvariantImpedance):
    """

    Attributes
    ----------
    frequences : array
      The vector of frequency where the impedance has been evaluated
    valeurs : array
      The vector of values of the impedance
    """

    def __init__(self, *args, **kwargs):
        """
        Loads an input impedance from raw and/or binary data file.
        You can either pass two numpy arrays for frequencies and complex
        impedance values (as arguments, with optional filename keyword
        argument for labelling), or either load data from a file whose format
        is specified with keyword argument 'storage' with one of the following
        values:
        - storage='txt_realimag': text archive files (freq, real, imag)
            ex: measurements with CTTM device
        - storage='txt_absangle': text archive files (freq, modulus, angle)
            ex : measurements with IKW device
        - storage='mat_freqZ': matlab archive with freq and Z variables.
        - storage='mat_wmZm': matlab archive with w_m and Z_m variables.
            ex: measurements by V. Debut (LAUM, 2003/2004, Laloe's clarinet).
        - storage='mat_Z3N': matlab archive (size Nx3: freq,reZ, imZ).
            ex: measurements by B. Vericel (IRCAM, 2010, Trumpet).

        'fmin' and 'fmax' allows to specify bounds to the frequency range.
        Extra keyword arguments are passed to the loading methods
        """
        self.Zc = kwargs.pop('Zc', 1.)
        fmin = kwargs.pop('fmin', 0.)
        fmax = kwargs.pop('fmax', np.inf)
        if len(args) == 2:
            f, Z = args
            filename = kwargs.get('filename', 'No filename')
        elif len(args) == 0:
            storage = kwargs.pop('storage', 'txt_realimag')
            dic = {
                'txt_realimag': self.load_txt_realimag,
                'txt_absangle': self.load_txt_absangle,
                'txt_absangledeg': self.load_txt_absangle,
                'mat_wmZm': self.load_wmZm,
                'mat_freqZ': self.load_freqZ,
                'mat_Z3N': self.load_mat3N}
            try:
                loader = dic[storage]
            except KeyError:
                raise IOError("Incorrect 'storage' argument.")
            if storage.find('deg'):
                kwargs['angle'] = 'deg'

            try:
                filename = kwargs.pop('filename')
                f, Z = loader(filename, **kwargs)
            except KeyError:
                raise IOError("Can't load data.")
        else:
            raise IOError("""
                Invalid number of input arguments. Choose either:
                - two arguments (freq and Z) plus optional 'filename' kwarg,
                - none (keywords arguments are then used).
                """)
        valid = (f <= fmax) * (f >= fmin) * (Z.real > 0.)
        self.frequencies = np.asarray(f)[valid]
        self.values = np.asarray(Z)[valid]
        self.label = filename
        self.version = 'Experimental data (%s).' % self.label

    def __getstate__(self):
        tmp = TimeInvariantImpedance.__getstate__(self)
        tmp.update({
            'file': self.label,
            'frequencies': self.frequencies,
            'values': self.values,
            'label': self.label,
            'version': self.version})
        return tmp

    def __repr__(self):
        return "Measured impedance: %s" % self.label

    def load_txt_realimag(self, filename, **kwargs):
        """
        Loader of text archive files (freq, real part, imag part)
        ex: measurements with CTTM device
        """
        kwargs.pop('angle', None)
        freq, real, imag = np.loadtxt(filename, unpack=True, **kwargs)
        return freq, real + 1.j * imag

    def load_txt_absangle(self, filename, **kwargs):
        """
        Loader of text archive files (freq, modulus, angle in rad)
        ex : measurements with IKW device
        """
        ang_unit = kwargs.pop('angle', 'rad')
        freq, mag, ang = np.loadtxt(filename, unpack=True, **kwargs)
        if ang_unit == 'deg':
            ang *= np.pi / 180.
        return freq, mag * np.exp(1.j * ang)

    def load_freqZ(self, filename, **kwargs):
        """
        Loader by freq and Z variables in .mat file.
        """
        import scipy.io as io
        kwargs.pop('angle', None)
        tmp = io.loadmat(filename, **kwargs)
        return tmp['freq'], tmp['Z']

    def load_wmZm(self, filename, **kwargs):
        """
        Loader by w_m and Z_m variables in .mat file.
        ex: measurements by V. Debut (LAUM, 2003/2004, Laloe's clarinet).
        """
        import scipy.io as io
        kwargs.pop('angle', None)
        tmp = io.loadmat(filename, **kwargs)
        return tmp['w_m'] / (2. * np.pi), tmp['Z_m']

    def load_mat3N(self, filename, **kwargs):
        """
        Loader by Z variable from a Nx3 array (freq,reZ, imZ) in .mat file.
        ex: measurements by B. Vericel (IRCAM, 2010, Trumpet).
        """
        import scipy.io as io
        kwargs.pop('angle', None)
        tmp = io.loadmat(filename, **kwargs)['Z']
        return tmp[:, 0], tmp[:, 1] + 1.j * tmp[:, 2]

    def _test_dimension(self):
        return (np.any(np.abs(self.values) > 500.))

    def estimate_modal_expansion(self, **kwargs):
        """
        Perform the estimation of a modal expansion of the loaded data.

        Parameters
        ==========
        algorithm : str 'Kennelly' or 'bruteforce'
            Algorithm used to compute the modal expansion.
        kwargs : passed to computational routines.
        """
        kwargs['output_snCn'] = True
        method = kwargs.pop('algorithm', 'Kennelly')
        from . import ModalExpansionEstimation as mod

        freq, valZ = self.frequencies, self.values
        fmin = kwargs.pop('fmin', self.frequencies[0])
        fmax = kwargs.pop('fmax', self.frequencies[-1])
        mask_opt = np.logical_and(freq > fmin, freq < fmax)
        freq, valZ = freq[mask_opt], valZ[mask_opt]

        if method.lower() == 'kennelly':
            tmp = mod.multiple_circles(freq, valZ, **kwargs)
            print("Kennelly fitting is over, please check the result!")
            self.poles, self.residues = tmp
        elif method.lower() == 'bruteforce':
            flag, tmp = mod.bruteforce_optimization(freq, valZ, **kwargs)
            if flag:
                print('Modal expansion estimation seems successful.')
                self.poles, self.residues = tmp
            else:
                print('Modal expansion estimation not successful...')
                self.poles, self.residues = np.array([]), np.array([])
        else:
            raise NotImplementedError('Algorithm %s does not exist.' % method)

        # Remove active modes
        idx = (self.poles.real <= 0.)
        self.poles = self.poles[idx]
        self.residues = self.residues[idx]
        self.nbmodes = len(self.poles)

    def trace(self, *args, **kwargs):
        if not('f' in kwargs):
            kwargs['f'] = self.frequencies
        self.as_timevariantimpedance()
        tmp = self._as_ltv.trace(*args, **kwargs)
        FigZ, FigR, freq, Zdata = tmp

        absZ, argZ = FigZ.get_axes()
        absR, argR = FigR.get_axes()

        freq = self.frequencies
        valZ = self.values
        valR = (valZ - 1.) / (valZ + 1.)
        Mag = np.abs(valZ)
        if self._test_dimension():
            Mag /= self.Zc
        Arg = np.angle(valZ) * 180. / np.pi
        MagR = np.abs(valR)
        ArgR = np.unwrap(np.angle(valR)) * 180.0 / np.pi
        if kwargs.get('linlog', 'lin') is 'lin':
            argZ.plot(freq, Arg)
            absZ.plot(freq, Mag, label='Measured')
            absR.plot(freq, MagR)
            argR.plot(freq, ArgR)
        else:
            argZ.semilogx(freq, Arg)
            absZ.semilogx(freq, 20. * np.log10(Mag))
            absR.semilogx(freq, MagR)
            argR.plot(freq, ArgR)

        absZ.legend()
        return FigZ, FigR, freq, Zdata
##############
# Obsolete API
##############


class ResonateurModesComplexes(TimeInvariantImpedance):
    " Obsolete class. See TimeInvariantImpedance"
    pass


class ResonateurAnalytique(Cylinder):
    " Obsolete class. See Cylinder."
    pass


class ResonateurExperimental(MeasuredImpedance):
    " Obsolete class. See MeasuredImpedance."
    pass


def restore(filename):
    """
    Restore a ResonateurModesComplexes instance from file.

    .. warning::

       Obsolete! Allow one to restore impedance created with previous
       versions of Moreesc

    Parameters
    ----------
    filename : str or file
      A handle for the file to load.

    Returns
    -------
    Ze : :class:`TimeInvariantImpedance`
      A :class:`TimeInvariantImpedance` instance constructed from old class.

    """
    import scipy.io as io
    dic = io.loadmat(filename, squeeze_me=True)
    return TimeInvariantImpedance(
        dic['sn'], dic['Cn'], not(dic['adim']), dic['Zc'])
