#!/usr/bin/python
# -*- coding: utf-8 -*-

#Copyright (C) 2008 Fabricio Silva
#Copyright (C) 2009 Sami Karkar

"""
:mod:`Simulation` -- Time-domain simulation
===========================================
.. moduleauthor:: Fabricio Silva <silva@lma.cnrs-mrs.fr.invalid>



Pay attention to the various attributes created by the differents
methods. They are exhibited within the functions which generate them.

"""

###import sys
import warnings
from time import strftime, time
import numpy as np
import scipy.optimize as opt
from scipy.integrate import ode
import moreesc.Profiles as Profiles
import moreesc.Valve as mec
import moreesc.AcousticResonator as ac
import moreesc.c_acous_sys as cas 
import moreesc.c_mech_sys as cms
import moreesc.c_coupling as cc
from . import utils
from . import ode_solvers
###import AnalyseLineaireStabilite as als

###debug = False


class TimeDomainSimulation(object):
    """
    :class:`TimeDomainSimulation` gathers all the informations about the
    configuration of the numerical experimentation.

    Attributes
    ----------
    valve
       A valve object
    resonator
       An acoustic resonator object.
    mouth_pressure, opening : :class:`Profiles.Profile`
      The time-varying control parameters.
    fs : float
      The apparent sampling frequency.
    time_vector : array
      The time vector.
    X0 : array
      The initial condition state vector  :math:`X(0)`.
    Nx : int
      The lenght of the state vector :math:`X`
    Nac : int
      Number of oscillating acoustical resonances.
    """

    def __init__(self, valve=None, resonator=None, fs=44100.,
                 integrator='vode', kw_integrator=None,
                 piecewise_constant=False,
            **kwargs):
        """
        Instantiate a new TimeDomainSimulation object given a Valve and a
        acoustical resonator. Additional arguments are

        Parameters
        ----------
        pm : Profile
            Mouth pressure profile
        h0 : Profile
            Channel's opening at rest
        fs : float
            the (apparent) sampling frequency of the simulation
        integrator : 'vode', 'dopri5', 'dop853', 'lsoda', 'Euler', 'EulerRichardson'
            the name of the integrator used (see scipy.integrate.ode)
        kw_integrator : dict
            dictionnary of options to pass to the integrator
        piecewise_constant : bool
            flag specifying whether the control parameter are kept constant
            between two consecutive samples.


        Note that the last three parameters can easily be changed during the
        simulation (see the integrator attribute and its set_integrator method)

        Examples
        --------
        >>> D  = mec.ReedDynamics(wr=6280., qr=0.4, kr=1e6, W=1.5e-2)
        >>> Ze = ac.Cylinder(L=3., r=7e-3, radiates=False, nbmodes=50)
        >>> pm = Profiles.Linear(instants=[0., 1.], values=[0., 1e3])
        >>> h0 = Profiles.Constant(8e-4)
        >>> sim = TimeDomainSimulation(valve=D, resonator=Ze, pm=pm, h0=h0)

        """
        # Handling control parameters
        # Note : labeled GroupProfiles can not ensure control parameters order
        #self.sim_args_profiles = Profiles.GroupProfiles(kwargs)
        # Order must be explicit:
        self.sim_args_profiles = Profiles.GroupProfiles(
            [kwargs.pop(tmp) for tmp in ('pm', 'h0')])
        if len(kwargs) != 0:
            print('Unhandled arguments: {}' + str(kwargs.keys()))

        if isinstance(valve, mec.TransferFunction):
            self.valve = valve
        else:
            raise TypeError('Valve can not be %s' % type(valve))

        if isinstance(resonator, ac.TimeVariantImpedance):
            self.resonator = resonator
        elif isinstance(resonator, ac.TimeInvariantImpedance):
            self.resonator = resonator.as_timevariantimpedance()
        else:
            raise TypeError('Resonator can not be %s' % type(valve))

        self.Nac = self.resonator.nbmodes
        self.Nx = 2 * self.Nac + len(self.valve.den) - 1
        self.X0 = np.zeros(self.Nx, dtype=float)

        # Setting global values
        cc.set_twodivbyrho(ac.rho)
        cc.set_reed_motion_coef(getattr(self.valve, 'reed_motion_coef', 0.))
        cms.set_beating_factor(getattr(self.valve, 'beating_factor', 0.))

        self.set_initial_state()

        self.fs = int(fs)
        self.time = np.array([], dtype=np.float64)
        self.result = np.empty((self.Nx, 0), dtype=np.float64)

        # Initialization of the integrator
        self.piecewise_constant = bool(piecewise_constant)

        # Ensuring intra state vector contiguity
        kw_sv = dict(dtype=np.float64, order='F')
        dX = np.zeros(self.Nx, **kw_sv) # Pre-allocated dX
        jX = np.zeros((self.Nx, self.Nx), **kw_sv) # Pre-allocated jX

        if piecewise_constant:
            self.integrator = ode(f=cc.global_sys, jac=cc.global_jac)
            f_args = (
                dX,
                self.resonator.poles(0.), self.resonator.residues(0.),
                valve.num(0.), valve.den(0.),
                self.sim_args_profiles(0.))
            jac_args = (
                jX,
                self.resonator.poles(0.), self.resonator.residues(0.),
                valve.num(0.), valve.den(0.))
        else:
            self.integrator = ode(
                f=cc.global_sys_profiles,
                jac=cc.global_jac_profiles)
            sn, Cn = self.resonator.poles, self.resonator.residues
            bn, an = valve.num, valve.den
            f_args = (
                dX,
                sn.sizes_array, sn.instants_array, sn.coefs_array,
                Cn.sizes_array, Cn.instants_array, Cn.coefs_array,
                bn.sizes_array, bn.instants_array, bn.coefs_array,
                an.sizes_array, an.instants_array, an.coefs_array,
                self.sim_args_profiles.sizes_array,
                self.sim_args_profiles.instants_array,
                self.sim_args_profiles.coefs_array)
            jac_args = (
                jX,
                sn.sizes_array, sn.instants_array, sn.coefs_array,
                Cn.sizes_array, Cn.instants_array, Cn.coefs_array,
                bn.sizes_array, bn.instants_array, bn.coefs_array,
                an.sizes_array, an.instants_array, an.coefs_array)
        self.integrator.set_f_params(*f_args)
        self.integrator.set_jac_params(*jac_args)
        dt = 1. / self.fs
            
        self.integrator_desc = ""
        tmp = {'first_step': dt / 10., 'max_step': dt / 2.}
        if kw_integrator:
            if not(isinstance(kw_integrator, dict)):
                raise TypeError('kw_integrator should be a dictionary.')
            tmp.update(kw_integrator)
        self.integrator.set_integrator(integrator, **tmp)
        self.integration_time = 0.
        
        self.attributes = [
            'valve', 'resonator', 'Nx', 'Nac', 'fs',
            'X0', 'sim_args_profiles', 'time', 'result',
            'integrator_desc', 'integration_time', 'piecewise_constant']

    def __getattr__(self, key):
        if key is 'result':
            raise AttributeError('No results yet. '
                'You may need to run the "solver" method first.')
        # Control parameters
        try:
            return self.sim_args_profiles[key]
        except IndexError:
            raise ValueError("Invalid control parameter index: %s" % key)
        except:
            raise AttributeError('No "%s" attribute.' % key)

    def __str__(self):
        return "<TimeDomainSimulation object at 0x%x>" % id(self)

    def __repr__(self):
        tmp = "Time Domain Simulation (sampling at %dHz" % self.fs
        if hasattr(self, "time"):
            tmp += ", [%.2e - %.2e] time range)" % (self.time[0], self.time[-1])
        else:
            tmp += ")"
        tmp += "\nAcoustic Resonator:\n"
        for line in self.resonator.__repr__().split('\n'):
            tmp += '\t' + line
        tmp += "\nValve:\n"
        for line in self.valve.__repr__().split('\n'):
            tmp += '\t' + line + '\n'
        for k, p in zip(("Mouth pressure", "Rest opening"), 
                        self.sim_args_profiles):
            tmp += '\n' + k + ':' + p.__repr__()
        return tmp

    # -----------
    # Persistence
    def __dir__(self):
        return list(self.attributes)

    def __getstate__(self):
        tmp = {}
        for el in self.attributes:
            tmp[el] = getattr(self, el)
        return tmp

    def __setstate__(self, dic):
        self.__dict__.update(dic)
        self.attributes = list(dic.keys())
        cc.set_twodivbyrho(ac.rho)
        if hasattr(self, 'result'):
            self.extract_signals()

    def save(self, filename=None):
        """
        Saves instance to file using pickle [pick]_.
        See :func:`load_impedance` for loading.

        Examples
        --------
        >>> sim.save('/tmp/simulation.dat')
        >>> sim = moreesc.Simulation.load_simulation('/tmp/simulation.dat')

        """
        if filename is None:
            if utils.method == 'hdf5':
                filename = self.label + '.h5'
            else:
                filename = self.label + '.dat'
        if filename.endswith('.mat'):
            raise NotImplementedError('Saving to mat format is not yet available.')
        else:
            utils._pickle(self, filename)

    # Accessing to physical quantities
    def get_mouth_pressure(self, t=None):
        " Get the mouth pressure profile evaluated at given instants. "
        if t is None:
            t = self.time
        return self.sim_args_profiles[0](t)

    def get_rest_opening(self, t=None):
        " Get the rest opening profile evaluated at given instants. "
        if t is None:
            t = self.time
        return self.sim_args_profiles[1](t)

    def get_pressure(self, X=None):
        " Get the opening from a state vector (or an array of state vectors). "
        if X is None:
            X = self.result
        assert X.shape[0] == self.Nx
        Xa = cc.get_acoustic_state_vector(self.Nac, X)
        return cas.get_pressure(Xa)

    def get_opening(self, X=None):
        " Get the opening from a state vector (or an array of state vectors). "
        if X is None:
            X = self.result
        assert X.shape[0] == self.Nx
        return cms.get_opening(cc.get_mechanic_state_vector(self.Nac, X))
        
    def get_gamma(self, t=None):
        """
        Get the reduced gamma parameter (see [Kergomard:1995]_) evaluated 
        at given instants. 
        """
        if t is None:
            t = self.time
        k = np.abs(self.valve.stiffness(t))
        pm = self.get_mouth_pressure(t)
        h0 = self.get_rest_opening(t)
        return pm / (k * h0)
        
    def get_zeta(self, t=None):
        """
        Get the reduced zeta parameter (see [Kergomard:1995]_) evaluated 
        at given instants. W Z_c \sqrt{\\frac{2 h_0}{\\rho K_r}}
        """
        if t is None:
            t = self.time
        k = self.valve.stiffness(t)
        h0 = self.get_rest_opening(t)
        Zc0 = self.resonator.Zc(t)
        return Zc0 * np.sqrt(2. * h0 / (ac.rho * k))
    
    def get_instantaneous_frequency(self, mode='yin',
                                    nwin=512, noverlap=256, **kwargs):
        """
        Get the instantaneous frequency from the pressure signal using Aubio.
        To recompute using another mode, delete the f_i attribute.
        """
        if hasattr(self, 'f_i'):
            return self.f_i

        try:
            import aubio
            alg = aubio.pitch(mode, nwin, noverlap, self.fs)
        except:
            print("Aubio (and its latest python bindings) are required to"
                  " estimate the instantaneous frequency (see www.aubio.org).")
            return [np.zeros(0), ] * 2
        
        alg.set_unit('freq')
        if 'yin' in mode and 'tolerance' in kwargs:
            alg.set_tolerance(kwargs['tolerance'])
        hopsize = nwin - noverlap
        nframes = min(1, int((self.pressure.shape[-1] - nwin) / hopsize))
        res = []
        for i0 in range(0, nframes * hopsize, hopsize):
            frame = self.pressure[i0: i0 + nwin].astype(np.float32)
            res.append(alg(frame))
        freq = np.array(res)[:, 0]
        freq[freq<0] = np.nan
        t = (np.arange(freq.size) * hopsize + nwin * .5) / float(self.fs)
        self.f_i = np.r_['0,2', t, freq]
        return self.f_i

    #-----------------------------------------
    # Initial conditions and equilibrium state
    def equilibrium_state(self, t=0., verbose=False):
        pm, h0 = self.sim_args_profiles(t)
        Z, D = self.resonator, self.valve
        sn, Cn = Z.poles(t), Z.residues(t)
        bn, an  = D.num(t), D.den(t)
        kr = D.stiffness(t)  # should equal an[-1] / bn[-1]

        X0 = np.zeros(self.Nx, dtype=float)
        dX0 = np.zeros(self.Nx, dtype=float)

        # Initial guess for opening (with null modal pressures)
        Xm0 = cc.get_mechanic_state_vector(self.Nac, X0)
        dXm0 = np.zeros_like(Xm0)
        Xm0[0] = h0 - pm / kr
        args = (dXm0, 0. - pm, h0, bn, an)
        func_m = lambda X: cms.mech_sys(t, X, *args)
        tmp = opt.fsolve(func_m, x0=Xm0, xtol=1e-6)
        opening = cms.get_opening(tmp)

        # Initial guess for volumeflow and pressure components
        flow = cc.nl_coupling(pm - 0., opening, 0.)
        modal_pressures = -Cn * flow / sn
        pressure = cas.get_pressure(modal_pressures)
        opening = h0 + (pressure - pm) / kr
        cc.set_acoustic_state_vector(self.Nac, X0, modal_pressures)
        args = (dX0, sn, Cn, bn, an, self.sim_args_profiles(t))
        func_g = lambda X: cc.global_sys(t, X, *args)
        tmp = opt.fsolve(func_g, X0, xtol=1e-3, full_output=True)
        if tmp[2] != 1:
            print("Failed to set initial state to equilibrium.\n")
            if verbose:
                print(tmp[1])
        return tmp[0]

    def set_initial_state(self, X0=None, *args, **kwargs):
        """
        Initialize the state vector. If a vector :keyword:`X0` is given,
        it is assigned to the initial conditions within the :attr:`X0`
        attribute. If not, the static state vector associated to the mouth
        pressure and the valve opening at instant :math:`t=0` is evaluated.

        Parameters
        ----------
        X0 : array-like, optional
            If specified, it must have the same shape as the state vector.

        Examples
        --------
        >>> sim.set_initial_state(X0=np.ones(sim.Nx))

        or

        >>> sim.set_initial_state()

        """
        if X0 is None:
            self.X0[:] = self.equilibrium_state(*args, **kwargs)
        else:
            self.X0[:] = np.asarray(X0, dtype=float)

    def set_integrator(self, name, **kwargs):
        tmp = {'first_step': .1 / self.fs, 'max_step' : .5 / self.fs}
        tmp.update(kwargs)
        self.integrator.set_integrator(name, **tmp)
        self.integrator_desc += r'\n%s\n' % name

    def integrate(self, t=1., verbose=True):
        """
        Solves the set of ordinary differential equations associated to the
        configuration using the solver :func:`scipy.integrate.ode`.

        Parameters
        ----------
        time : float
            Setting the time range over which integration is performed.

        Examples
        --------
        >>> sim.integrate(1.)

        You can interrupt the computation using CTRC-C. It will however update
        the following attributes according to the last step computed and extract
        signals (with the extract_signals method).

        Attributes
        ----------
        time : array
            Discrete time vector
        result : 2D array
           Raw data results.
        label : str
           A timestamp of the simulation


        """
        # Time vector
        if len(self.time):
            t_prev = self.time[-1]
            r_prev = self.result[:, -1]
        else:
            t_prev = 0.
            r_prev = self.X0
        if not(hasattr(self, 'integrator')):
            raise AttributeError('Set the integrator first...')
        self.integrator.set_initial_value(r_prev)
        self.integrator.t = t_prev

        dt = 1. / float(self.fs)
        time_range = np.arange(t_prev, float(t) + .5 * dt, dt)[1:]

        # Ensuring intra state vector contiguity
        kw_sv = dict(dtype=np.float64, order='F')
        result = np.empty((self.Nx, time_range.size), **kw_sv)
        result[:, :] = np.nan

        sn, Cn = self.resonator.poles, self.resonator.residues
        bn, an = self.valve.num, self.valve.den
        tic = time()
        indt = 0
        self.integrator_desc += "t=%.3e - " % self.integrator.t
        while indt < time_range.size:
            t = self.integrator.t
            if self.piecewise_constant:
                tmp = (sn(t), Cn(t), bn(t), an(t), self.sim_args_profiles(t))
                for ind, el in enumerate(tmp):
                    self.integrator.f_params[1+ind][:] = el
                    if ind == len(tmp) - 1:
                        break
                    self.integrator.jac_params[1+ind][:] = el
            if verbose and indt % 1000 == 0:
                print("Solver t[%s]=%.3f" % (indt, t))

            try:
                self.integrator.integrate(time_range[indt])
            except KeyboardInterrupt:
                # Allows to stop integration but yet keeps computed results.
                break

            if not(self.integrator.successful()):
                break
            result[:, indt] = np.squeeze(self.integrator.y)
            indt += 1

        toc = time()
        if hasattr(self, 'integration_time'):
            self.integration_time += toc - tic
        else:
            self.integration_time = toc - tic
            
        if verbose:
            mn, sec = divmod(toc-tic, 60)
            print("Integration duration : %d min %d sec." % (mn,sec))

        if indt > 0:
            self.time = np.r_[self.time, time_range[:indt]]
            self.result = np.c_[self.result, result[:, :indt]]
            if not(hasattr(self, 'label')):
                self.attributes += ['label', ]
            self.label = strftime("Simulation.%Y%m%d.%Hh%M")
            self.extract_signals()
        self.integrator_desc += "%.3e\n" % self.integrator.t
        return self.result

    def extract_signals(self):
        """
        Extract the signals of mouthpiece pressure, volume flow and tip
        opening from the raw solution of the set of ODE. An estimation of the
        radiated pressure is also computed.

        They are then available with through the following attributes

        Attributes
        ----------
        pressure : array
           The total pressure in the mouthpiece, i.e. the sum of components.
        pm : array
           The excitation pressure in the mouth.
        flow : array
           The volume flow entering through the reed channel.
        opening : array
           The varying tip opening.
        h0 : array
           The tip opening at rest.
        external_pressure : array
           An approximation of the radiated sound pressure.

        """
        X = self.result
            
        self.pressure = self.get_pressure(X)
        self.pm = self.get_mouth_pressure()

        self.opening = self.get_opening(X)
        self.h0 = self.get_rest_opening()

        if self.pressure.ptp()==0:
            print("No variations in the pressure signal.")

        cc.set_twodivbyrho(ac.rho)
        # Bernoulli flow only
        self.flow = cc.nl_coupling(self.pm - self.pressure,
                                   self.opening,
                                   np.zeros_like(self.opening))

        # Filtered derivate 
        from scipy.signal import lfilter
        Zc = self.resonator.Zc(self.time)
        Pplus = 0.5 * (self.pressure + Zc * self.flow)
        w0Te = 2. * np.pi * 2000. / self.fs
        B, A = (1., -1.), (1., -1. / (1. + w0Te))
        self.external_pressure = lfilter(B, A, Pplus)

    def reconstruct_spatial_field(self, vectors, decimate=1):
        """
        Reconstructs the time-evolving pressure field within the resonator
        from a simulation result and knowing the eigenvectors related to the
        poles of the acoustic resonator.

        Parameters
        ----------
        vectors : array-like
           The list of the (possibly complex) eigenvectors
           Shape: self.Nac x number of nodes
        decimate : int, optional
           A decimation factor for sampling frequency.

        Returns
        -------
        Ptx : 2d array
           The array containing the pressure value for each spatial point
           and time instant (time signals as columns, instantaneous pictures
           as rows)

        """
        assert vectors.shape[0] == self.Nac
        nt, nx = len(self.time)/int(decimate), vectors.shape[1]
        ptx = np.empty((nt, nx), dtype=float)

        Xa = cc.get_acoustic_state_vector(self.result)
        for indt in range(nt):
            # TODO: What about purely real poles? and truncature correction?
            ptx[indt,:] = 2. * (Xa[:, indt] * vectors).real
        return ptx

###    def analyse_stabilite(self):
###        """
###        Performs a linear stability analysis of the static regime. It computes
###        the oscillation threshold using the module
###        ``AnalyseLineaireStabilite``.[Sil:08]_
###
###        .. [Sil:08] Interaction of reed and acoustic resonator in clarinet-like systems, F. Silva, J. Kergomard, Ch. Vergez and J. Gilbert. J. Acoust. Soc. Am. 124(5), pp.3284-3295, 2008.
###        """
###        if self.profil_ouverture.pts.shape[1]>1:
###            warnings.warn(u"Profil d'ouverture non constant : le maximum est retenu.")
###            h0 = self.profil_ouverture.pts[1,:].max()
###        else:
###            h0 = self.profil_ouverture.pts[1,0]
###        # Du fait de l'impédance non nulle à fréquence nulle, il faut
###        # ajouter à la valeur gamma_p la valeur moyenne de la pression
###        # dans le bec
###        PM = self.k*h0
###        gamma_p, sol = als.recherche_seuil(self)
###        if np.isnan(gamma_p):
###            warnings.warn(u"Trouble in threshold computation.")
###            return np.nan, np.nan*1.j, np.nan
###        X = self._regime_statique(PM*gamma_p, h0)
###        p0 = 2.*X[:-3:2].sum()
###        # En première approximation de l'équation
###        #   p_bou-p_bec(p_bou)=gamma_p*PM
###        pbouche_seuil = gamma_p*PM+p0
###        if debug:
###            print u"Threshold pressure: %.1f Pa (gamma=%.3f)" \
###                %(pbouche_seuil, pbouche_seuil/PM)
###            print u"Complex solution: ", sol
###        return pbouche_seuil, sol, PM

    def trace(self, tmin=None, tmax=None, trace_signals=True,
            trace_components=False, trace_spectrogram=False,
            trace_spectrums=False, trace_all=False,
            trace_instantaneous_frequency=False,
            fmax=5000., verbose=False, **kwargs):
        """
        Plots several figures, with possible time range reduction

        Parameters
        ----------
        tmin, tmax : float, optional
           Lower and upper bounds of time range to display.
        trace_signals : bool, optional
           whether to plot a figure with pressure, opening and volume flow.
        trace_components : bool, optional
           whether to plot the first components of the pressure field.
        trace_spectrogram : bool, optional
           whether to plot a spectrogram of the pressure signal
        trace_spectrums : bool, optional
           whether to plot spectrums of pressure, opening and volume flow.
        trace_all : bool, optional
            whether to plot all the previously mentioned figures.
        fmax : float, optional
           Maximal frequency to display if :keyword:`spec` is True.

        """
        import matplotlib.pyplot as plt
        tmin = np.nanmax(np.array([tmin, self.time.min()], dtype=float))
        tmax = np.nanmin(np.array([tmax, self.time.max()], dtype=float))
        indt = np.logical_and(self.time > tmin, self.time < tmax)
        indt_wo_ds = np.copy(indt)

        # Downsampling required ?
        if 'cairo' in plt.rcParams['backend'].lower():
            # Cairo maximum number of points
            nmax = 18980 # Hard-coded in matplotlib/backends/backend_cairo.py 
            if indt.sum() > nmax:
                ndec = np.ceil(1. * indt.sum() / nmax)
                tmp = np.zeros_like(indt)
                tmp[::ndec] = True
                indt *= tmp
                del tmp
                if verbose:
                    print("Downsampling display: only taking one point "
                          "out of %d." % ndec)
            
        t = self.time[indt]
        label_kw = {'rotation': 'vertical', 'size': 'smaller'}
        Figs = {}
        axref = True

        if trace_signals or trace_all:
            Fig, axs = plt.subplots(4, 1, sharex=True, squeeze=True)

            axs[0].plot(t, self.pm[indt], 'r', label=r'$p_{m}$')
            axs[0].plot(t, self.pressure[indt], label=r'$p_{e}$', **kwargs)
            axs[0].set_ylabel(r'$p\ (Pa)$', **label_kw)

            axs[1].plot(t, 1e6 * self.h0[indt], 'r', label=r'$h_0$')
            axs[1].plot(t, 1e6 * self.opening[indt], label=r'$h$', **kwargs)
            axs[1].set_ylabel(r'$x\ (mm^2)$', **label_kw)

            axs[2].plot(t, 60e3 * self.flow[indt], label=r'$u$', **kwargs)
            axs[2].set_ylabel(r'$u\ (L/min)$', **label_kw)

            axs[3].plot(t, self.external_pressure[indt], **kwargs)
            axs[3].set_ylabel(r'$p_{ext}$', **label_kw)
            axs[3].set_xlabel(r'$t\ (s)$')

            for ax in axs:
                ax.legend()
                ax.label_outer()
            Figs['signals'] = Fig
            axref = axs[0]

        if trace_components or trace_all:
            nmax = min(10, self.Nac)
            Fig, axs = plt.subplots(nmax, 1, squeeze=False,
                                    subplot_kw={'sharex': axref})

            ax = axs[0, 0]
            ax.plot(t, self.pressure[indt], **kwargs)
            ax.plot(t, self.pm[indt], 'r')
            ax.set_ylabel('$p$', **label_kw)
            ax.label_outer()

            Xac = cc.get_acoustic_state_vector(self.Nac, self.result)
            for ind, ax in enumerate(axs[1:, 0]):
                ax.plot(t, 2. * Xac[ind,indt].real, **kwargs)
                Vmax = np.abs(ax.get_yticks()).max()
                ax.set_yticks([-Vmax, 0, Vmax])
                ax.set_ylabel(r'$Re(p_{%d})$' % (ind + 1), **label_kw)
                ax.label_outer()

            axs[-1, 0].set_xlabel(r'$t\ (s)$')
            Figs['components'] = Fig
            if axref is False:
                axref = axs[0, 0]

        N = 13
        if trace_spectrogram or trace_all:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            Fig, ax = plt.subplots(1, 1, squeeze=True,
                                   subplot_kw={'sharex': axref})
            divider = make_axes_locatable(ax)

            # Pressure signal on top axes
            cax = divider.append_axes("top", size="10%", pad=0.05, sharex=ax)
            cax.plot(t, self.pressure[indt], **kwargs)
            for label in cax.get_xticklabels() + cax.get_yticklabels():
                label.set_visible(False)
            cax.set_ylabel(r'$p$', **label_kw)

            # Pressure spectrogram on center axes
            nTot = indt_wo_ds.sum()
            nFFT = 2 ** N
            nFram_max = 1024
            nOver = max(0, int(nFFT - (nTot - nFFT) / nFram_max))
            nOver = min(nOver, nFFT-1)
            nFram = int((nTot - nFFT) / (nFFT - nOver))
            if verbose:
                print("FFT size: %d\t\tOverlap: %d\t\t# Frames: %d"
                      % (nFFT, nOver, nFram))
            Im = ax.specgram(
                self.pressure[indt_wo_ds], Fs=self.fs * 1e-3,
                    NFFT=nFFT, noverlap=nOver, pad_to=nFFT,
                    xextent=(tmin, tmax),
                    cmap=plt.cm.hot_r, interpolation='none')[-1]
            ax.set_xlabel(r'$t\ (s)$')
            ax.set_ylim(0., fmax * 1e-3)
            tmp = Im.get_clim()
            Im.set_clim(tmp[1] - 100, tmp[1])
            # Adding resonances
            for pole in self.resonator.poles:
                fn = np.abs(pole(t)) / (2. * np.pi)
                L1 = ax.plot(t, 1.e-3 * fn, 'b', lw=.2)[0]
            if isinstance(self.valve, mec.OneDOFOscillator):
                fr = self.valve.wr(t) / (2. * np.pi)
                L2 = ax.plot(t, 1.e-3 * fr, 'k', lw=.2)[0]
            ax.legend((L1, L2), ("Acoustic resonances", "Valve resonance"))
            for label in ax.get_yticklabels():
                label.set_visible(False)
            # Colorbar on the right side
            cbax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(Im, cax=cbax)

            # Resonator impedance on the left side
            Zax = divider.append_axes("left", size="10%", pad=0.05, sharey=ax)
            Zax.set_ylabel(r'$f\ (kHz)$')
            f = np.linspace(0., fmax, 8192)
            Zdata = self.resonator(s=2.j * np.pi * f)[0]
            Zax.semilogx(np.abs(Zdata), f * 1e-3)
            ax.set_ylim(0., fmax * 1e-3)
            for label in Zax.get_xticklabels():
                label.set_visible(False)

            Figs['spectrogram'] = Fig
            if axref is None:
                axref = ax

        if trace_spectrums or trace_all:
            Fig, axs = plt.subplots(3, 1, sharex=True, squeeze=True)
            from matplotlib.mlab import psd
            NFFT = 2 ** N
            kw_psd = dict(
                NFFT=NFFT, Fs=self.fs,
                noverlap=NFFT / 4, pad_to=NFFT,
                window=np.hanning(NFFT))
            DSPp, vecf = psd(self.pressure, **kw_psd)
            DSPh = psd(self.opening, **kw_psd)[0]
            DSPu = psd(self.flow, **kw_psd)[0]
            vecf = vecf[vecf >= 0]
            idx = (vecf < fmax)
            
            for ax, dsp, label in zip(axs, [DSPp, DSPh, DSPu],
                    [r'$|P_{e}|$', r'$|H|$', r'$|U|$']):
                ax.plot(vecf[idx], 20. * np.log10(dsp[idx]), **kwargs)
                ax.set_ylabel(label, **kwargs)
                ax.label_outer()

            axs[-1].set_xlabel('$f\ (Hz)$')
            Figs['spectrums'] = Fig

        if trace_instantaneous_frequency or trace_all:
            Fig, axs = plt.subplots(2, 1, squeeze=True,
                                    subplot_kw={'sharex': axref})

            t, freq = self.get_instantaneous_frequency()
            indt2 = (t > tmin) * (t < tmax)
            axs[0].plot(self.time[indt], self.pressure[indt], **kwargs)
            axs[0].set_ylabel('$p$', **label_kw)
            axs[0].label_outer()

            axs[1].plot(t[indt2], freq[indt2], **kwargs)
            axs[1].set_ylabel('$f\ (Hz)$', **label_kw)
            axs[1].set_xlabel('$t\ (s)$')
            Figs['finst'] = Fig
            if axref is None:
                axref = axs[0]

        return Figs

    # Audio I/O
    def save_wav(self, filename=None, fmt=None, where='out',
                remove_peaks=False):
        """
        Record the pressure into an audio file.

        Parameters
        ----------
        filename : File or str
          A description of file (will be over-written if existing).
        format : audiolab.Format, str or list of strings
          An audio format suitable to audiolab Format class
        where : str 'in' or 'out'
          The signal used to create the wav file ('in' for mouthpiece pressure,
          'out' for a pseudo-radiated presure).
        remove_peaks : bool
          Whether to remove peaks
        """
        try:
            import scikits.audiolab as au
        except ImportError:
            warnings.warn(
                'Scikits.audiolab is recommended for writing wav files.\n'
                'Using Python standard module wave instead.')
            au = None
            import wave

        if where.lower() == 'out':
            sig = self.external_pressure
        elif where.lower() == 'in':
            sig = self.pressure
        else:
            raise ValueError("Invalid micro argument (in/out): %s." % where)

        if filename is None:
            filename = self.label + ".wav"
            print("Saving %s Wav to: %s" % (where, filename))

        sig = np.array(sig, dtype='<f4', copy=True)
        if remove_peaks:
            # Compression: prevent loss of dynamic due to spurious peaks.
            idx = (np.abs(sig - sig.mean()) > 10. * sig.std())
            sig[idx] = 0.
        
        idx = ~np.isfinite(sig)
        sig[idx] = 0.

        # Normalization
        sig /= (1.1*np.abs(sig).max())

        if au:
            # Using scikits.audiolab
            if isinstance(fmt, au.Format):
                pass
            elif isinstance(fmt, str):
                fmt = au.Format(fmt)
            elif fmt is None:
                fmt = au.Format('wav', 'pcm32')
            else:
                fmt = au.Format(*fmt)
            f = au.Sndfile(filename, 'w', fmt, 1, self.fs)
            f.write_frames(sig)
            f.close()
        else:
            # Using wave pcm32 by default
            dtype = np.int32
            f = wave.open(filename, 'w')
            f.setparams((1, dtype().itemsize, self.fs, sig.size,
                        'NONE', 'uncompressed'))
            sig *= np.iinfo(dtype).max
            f.writeframes(sig.astype(dtype).tostring())
            f.close()

    def play(self, where='in'):
        """
        Plays the sound produced by the computations.

        Parameters
        ----------
        where : str 'in' or 'out'
          The signal to be played ('in' for mouthpiece pressure,
          'out' for a pseudo-radiated presure).

        """
        try:
            import scikits.audiolab as au
        except ImportError:
            warnings.warn(
                'Scikits.audiolab is required for playing wav files.\n'
                'http://pypi.python.org/pypi/scikits.audiolab/')
            return

        if where.lower() == 'out':
            sig = self.external_pressure
        elif where.lower() == 'in':
            sig = self.pressure
        else:
            raise ValueError("Invalid micro argument (in/out): %s." % where)

        au.play(sig / (1.1*np.abs(sig).max()), fs=self.fs)

load_simulation = lambda s: utils._unpickle(s)
load_simulation.__doc__ = utils.__pickle_common_doc \
     % {'class': 'TimeDomainSimulation', 'output': 'TimeDomainSimulation'}
load_simulation.__name__ = 'load_simulation'

if __name__=='__main__':
    D = mec.LipDynamics(wr=2*np.pi*1500., qr=0.1, kr=1e9)
    Ze = ac.Cylinder(L=3., r=7e-3, radiates=False, nbmodes=2)

    pm = Profiles.Linear([0.05, .1, .9, .95], [1000., 3e3, 3e3, 0.])
    h0 = Profiles.Constant(3e-6) # Channel section at rest (in m**2)

    sim = TimeDomainSimulation(D, Ze, pm=pm, h0=h0)
    sim.set_initial_state()
    print(sim.X0)

#    a.grandeurs_controle()

    if True:
        sim.solver()
#        a.extrait_signaux()
#        #fichier = a.sauvegarde()
#        #print "Sauvegarde de l'instance Simulation dans %s" %fichier
#        #a.record_wavfile(location='in')
#        import matplotlib.pyplot as plt
#        #a.affiche(spec=False, pn=False, freqinst=False)

#    print "Gamma_max : %.3f\t Zeta_max : %.3f" % \
#        (a.gamma.max(), a.zeta.max())
