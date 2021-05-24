#!/usr/bin/python
# -*- coding: utf-8 -*-

#Copyright (C) 2006 Fabricio Silva

"""
:mod:`Valve` -- Specifiyng the mechanical resonator
===================================================
.. moduleauthor:: Fabricio Silva <silva@lma.cnrs-mrs.fr.invalid>

This module provides the class objects useful for the description of the valve.

.. warning::

   An important point is that the Valve instance that is passed to the
   :class:`Simulation.TimeDomainSimulation` object relates the pressure difference
   :math:`p_m-p_e` and the valve channel cross-section area :math:`S`.

"""

import warnings as wa
import numpy as np
import numpy.lib.polynomial as poly
from . import Profiles
from . import utils

class TransferFunction:
    """
    A general class to represent systems by means of Laplace transform. It
    describes a linear dynamical system by numerator and denominator
    polynomials of the :math:`s` variable.

    .. math::

       F(s=\\alpha+j\omega)
       = \\frac{\sum_{m=0}^M b_m s^{M-m}}{\sum_{=0}^N a_n s^{N-n}}

    with :math:`M<N`, usually evaluated on the frequency axis
    :math:`s=j\omega`.

    Attributes
    ----------
    num,den : :class:`Profiles.GroupProfiles`
      contain the (possibly time variable) coefficients of the numerator and the denominator of the :class:`TransferFunction` instance.

    """

    def __init__(self, num, den):
        """
        Constructor of the TransferFunction class.

        Parameters
        ----------
        num, den : list of scalars or list of Profiles
            Sequence of the coefficients (potentially time varying)
            of numerator and denominator polynomials from highest powers
            of the Laplace variable :math:`s` to lowest.

        """
        self.num = Profiles.GroupProfiles(num)
        self.den = Profiles.GroupProfiles(den)
        try:
            assert self.den[0].value==1.
        except (AttributeError, AssertionError,KeyError):
            raise ValueError("Highest power coefficient in denominator must be 1.")

    def save(self, filename):
        """
        Saves instance to file using pickle [pick]_.
        See :func:`Profiles.load_groupprofiles` for loading.
        """
        utils._pickle(self, filename)

    def __instantaneous_operator(self, func, t):
        " Private method to apply function for several time values. "
        num = np.atleast_2d(np.r_[self.num(t)].T)
        den = np.atleast_2d(np.r_[self.den(t)].T)
        result = [func(n, d) for n, d in zip(num, den)]
        if np.isscalar(t): result = result[0]
        return result

    def __call__(self, s, t=0.):
        """
        Evaluate the transfer function for the given values of the Laplace
        variable s. As the coefficients may (slowly) depend on time, the
        frequency response is itself dynamic (depends on time).

        Parameters
        ----------
        s : array-like
            List of values of Laplace variable at which transfer function
            is evaluated.
        t : scalar, optional
            Instant to consider when evaluating the coefficients of the
            transfer function. Default is 0.

        Returns
        -------
        H : list of frequency responses (one per instant required).
        """
        if not(np.iscomplexobj(s)):
            wa.warn("""
            VALVE: When calling a TransferFunction instance, argument
            must be given in terms of Laplace variable s. It is assumed
            real frequencies have been provided here.""")
            s = 2.j*np.pi*np.array(s)
        else:
            s = np.asarray(s)

        func = lambda a,b: poly.polyval(a, s)/poly.polyval(b, s)
        return self.__instantaneous_operator(func, t)

    def __str__(self):
        return "%s instance (num of order %d, den of order %d)" % \
            (self.__class__.__name__, len(self.num), len(self.den))

    def __repr__(self):
        from np.polynomial import Polynomial as P
        Brepr = P(self.num).__repr__()
        Arepr = P(self.den).__repr__()
        return "TransferFunction H(s)=B(s)/A(s)\n\tB(s)=%s\n\tA(s)=%s" % (Brepr, Arepr)

    def __eq__(self, other):
        if not(isinstance(other, self.__class__)):
            return False
        return self.den==other.den and self.num==other.num

    def __ne__(self, other):
        return not(self.__eq__(other))

    def get_poles(self, t=0.):
        """
        Evaluate the location of the poles of the transfer function.
        As the coefficients may (slowly) depend on time, the poles may vary.

        Parameters
        ----------
        t : scalar, optional
            Instant to consider when evaluating the coefficients of the
            transfer function. Default is 0.

        Returns
        -------
        H : list of lists of poles (one per instant required).
        """
        func = lambda a,b: poly.roots(b)
        return self.__instantaneous_operator(func, t)

    def get_zeros(self, t=0.):
        """
        Evaluate the location of the zeros of the transfer function.
        As the coefficients may (slowly) depend on time, the zeros may vary.

        Parameters
        ----------
        t : scalar, optional
            Instant to consider when evaluating the coefficients of the
            transfer function. Default is 0.

        Returns
        -------
        H : list of lists of zeros (one per instant required).
        """
        func = lambda a,b: poly.roots(a)
        return self.__instantaneous_operator(func, t)


    def trace(self, f=None, linlog='lin', t=[0.,]):
        """
        Plots representations of the frequency response. The graphics
        properties can be modified directly on the returned object.

        Parameters
        ----------
        f : array-like, optional
            Frequencies at which frequency response is evaluated.
        linlog : string, optional
            The modulus is displayed with a linear axis if ``lin``, with a
            logarithmic axis if ``log``. An exception is raised in others cases.
        t : scalar, optional
            Instant to consider when evaluating the coefficients of the
            transfer function. Default is 0.

        Returns
        -------
        Fig : :class:`matplotlib.Figure`
            The figure object

        """
        import matplotlib.pyplot as plt
        Fig = plt.figure()
        Module = Fig.add_subplot(211)
        Phase = Fig.add_subplot(212, sharex=Module)
        poles = self.get_poles(t)
        Zdata = []
        for indt,valt in enumerate(np.atleast_1d(t)):
            Puls = np.array(poles[indt]).imag
            NbModes = len(Puls[Puls>=0])

            if not(f==None):
                wdata = 2.*np.pi*f
            elif linlog=='lin':
                wmin = 0.1*np.min(np.abs(Puls))
                if NbModes!=1:
                    wmax = 1.1*np.max(np.abs(Puls))
                else:
                    wmax = 3*np.max(np.abs(Puls))
                wdata = np.linspace(wmin, wmax, len(Puls)*200)
            elif linlog=='log':
                logwmin = max(-4, np.floor(np.log10(np.min(Puls))))
                logwmax = min(6, np.ceil(np.log10(np.max(Puls))))

                if not(np.isfinite(logwmin)):
                    logwmin = logwmax-3
                if not(np.isfinite(logwmax)):
                    logwmax = logwmin+3
                wdata = np.logspace(logwmin, logwmax, 200)
            else:
                raise ValueError("linlog argument should only be 'lin' or 'log'.")

            Zdatat = self(1.j*wdata, t=[valt,])[0]
            Zdata.append(Zdatat)

            Mag = 20.0*np.log10(np.abs(Zdatat))
            Arg = np.arctan2(Zdatat.imag, Zdatat.real)*180.0/(np.pi)

            if linlog=='lin':
                Phase.plot(wdata/(2*np.pi), Arg)
                Module.plot(wdata/(2*np.pi), np.abs(Zdatat))
            else:
                Phase.semilogx(wdata/(2*np.pi), Arg)
                Module.semilogx(wdata/(2*np.pi), Mag)
            Module.get_lines()[-1].set_label(r'$t=%.3f$'%valt)

        Phase.grid('on')
        Phase.set_ylabel(r'Phase (deg)', va='bottom')
        Phase.set_xlabel(r'Frequency (Hz)')
        Phase.set_yticks(np.arange(-120, 120, 30))
        Module.grid('on')
        Module.set_ylabel(r'Magnitude')
        Module.label_outer()
        Module.legend()
        if f==None:
            return Fig
        else:
            return Fig, f, Zdata

    def stiffness(self, t=0.):
        return self.den[-1](t) / self.num[-1](t)

class OneDOFOscillator(TransferFunction):
    """
    A TransferFunction subclass intended to model a single degree of freedom
    oscillator.

    .. math::

       H(s) = \\frac{H0}{1+q_r\\frac{s}{\omega_r}+\\frac{s^2}{\omega_r^2}}.

    Attributes
    ----------
    wr : Profile
      Natural angular frequency of the oscillator
    qr : Profile
      Damping of the oscillator
    HO : Profile
      Low frequency gain of the oscillator

    """

    def __init__(self, wr, qr, H0=1., beating_factor=0.):
        self.wr = Profiles.toProfile(wr)
        self.qr = Profiles.toProfile(qr)
        self.H0 = Profiles.toProfile(H0)
        self.beating_factor = float(beating_factor)
        wr2 = self.wr**2
        num = [self.H0*wr2,]
        den = [1., self.qr*self.wr, wr2]
        TransferFunction.__init__(self, num, den)

    def __repr__(self):
        tmp = "One degree of freedom oscillator (lowpass 2th order filter):\n"
        tmp += "at t=0: Natural frequency: %.1fHz\t Damping:%.2e\tStatic gain:%.3e" % \
            (self.wr(0.)/(2.*np.pi), self.qr(0.), self.H0(0.))
        return tmp

    def __str__(self):
        return "One DOF oscillator (lowpass 2th order filter)"

    def __eq__(self, other):
        return TransferFunction.__eq__(self, other) and \
            all([getattr(self,k)==getattr(other,k) for k in ('wr', 'qr', 'H0')])
    
    def __getstate__(self):
        return {'wr': self.wr, 'qr': self.qr, 'H0': self.H0, \
                'beating_factor': self.beating_factor,
                'num': self.num, 'den': self.den}

    def stiffness(self, t=0.):
        return 1. / self.H0(t)
        

commondoc = r"""
:class:`%(classname)s` provides a one degree of freedom mechanical
oscillator that models the %(valve)s. It defines the characteristics of
the transfer function between the pressure difference (between
mouth :math:`p_m` and mouthpiece :math:`p_e`) and the %(valve)s channel
cross-section area :math:`S`.

.. math::

   \frac{d^2 S}{dt^2}+q_r\omega_r\frac{dS}{dt}
   +\omega_r^2 \left(S-S_0\right)
   = \frac{\omega_r^2}{K_r}\left(p_e(t)-p_m(t)\right)

with an %(valvetype)s valve behaviour: :math:`K_r%(signK)s0`.

Attributes
----------
wr : Profile
  Natural angular frequency of a vibrational mode of the %(valve)s,
qr : Profile
  Damping of the same vibrational mode,
Kr : Profile
  Mechanical stiffness of the %(valve)s (quasi static value, relating the
  static pressure difference to the static cross-section area).


"""

class ReedDynamics(OneDOFOscillator):
    __doc__ = commondoc % {'classname':'ReedDynamics', 'signK':'>', \
        'valve':'reed', 'valvetype':'inward'}

    def __init__(self, wr, qr, kr=1e6, beating_factor=1.e3):
        kr = Profiles.toProfile(kr)
        kr.coefs = np.abs(kr.coefs)
        H0 = kr ** -1
        OneDOFOscillator.__init__(self, wr, qr, H0, beating_factor)

    def __repr__(self):
        return "Reed valve - " + OneDOFOscillator.__repr__(self)

    def __str__(self):
        return "Reed valve"

    def __eq__(self, other):
        return OneDOFOscillator.__eq__(self, other)

class LipDynamics(OneDOFOscillator):
    __doc__ = commondoc % {'classname':'LipDynamics', 'signK':'<', \
        'valve':'lip', 'valvetype':'outward'}

    def __init__(self, wr, qr, kr=1e6, beating_factor=0.):
        kr = Profiles.toProfile(kr)
        kr.coefs = -np.abs(kr.coefs)
        H0 = kr ** -1
        OneDOFOscillator.__init__(self, wr, qr, H0, beating_factor)

    def __repr__(self):
        return "Lip valve - " + OneDOFOscillator.__repr__(self)

    def __str__(self):
        return "Lip valve"

    def __eq__(self, other):
        return OneDOFOscillator.__eq__(self, other)

load_transferfunction = lambda s: utils._unpickle(s)
load_transferfunction.__doc__ = utils.__pickle_common_doc \
    % {'class':'TransferFunction', 'output':'TransferFunction'}
load_transferfunction.__name__ = 'load_transferfunction'
