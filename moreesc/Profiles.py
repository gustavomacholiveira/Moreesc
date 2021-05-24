#!/usr/bin/python
# -*- coding: utf-8 -*-

#Copyright (C) 2010 Fabricio Silva

"""
:mod:`Profiles` -- Time-varying quantities
================================================
.. moduleauthor:: Fabricio Silva <silva@lma.cnrs-mrs.fr.invalid>

**Moreesc** is intended to deal with time-varying control parameters
and time-evolving characteristics of resonator and valve.
This module provide the framework for modelling simple or real-world data, and
be able to perform efficient computation on them. It builds instances of
:class:`Profile` (or one of its subclass) that can easily be evaluated within
the calculation of self-sustained oscillations.

In addition to basic constant and linear profiles :class:`Constant`
and :class:`Linear`, :class:`Spline` is based on B-splines [BSplines]_,
which are transformed without any approximation into a sequence of Bezier
curves [Bezier]_ for a cheaper evaluation during the calculation of
oscillations. Measured signals can thus be parametrized and used with the
computation (see :class:`Signal`). :class:`Spline` (and its subclass
:class:`Signal`) can easily be manipulated by means of a graphical
:meth:`Spline.editor`.

"""

import sys
#import warnings as wa
import numpy as np
import scipy.interpolate as ii
from . import utils
from . import c_profiles as sf

eps = 1.e-8
__all__ = ['Profile', 'GroupProfiles', 'toProfile', \
    'load_profile', 'load_groupprofiles', \
    'Constant', 'Linear', 'Spline', 'Signal', 'SmoothStep', 'C2_Step']


class Profile(object):
    r"""
    General class intended to provide a parametrization of temporal evolution
    of coefficients of the model. The method used here bases on a decomposition
    into a sequence of Bezier curves in the (t,value) plane.

    The :attr:`coefs` attribute consists of a :math:`(2,4n)`-shaped numpy
    array, the two lines containing the instants and the values of the nodes,
    respectively. Groups of 4 rows defines the four Bezier nodes of each Bezier
    curves (of degree 3). Thus the n-th curves of the Profile is made from
    the nodes

    .. math::

       \forall\ i\ \in\ [0,3],
       t=\mathrm{coefs}[0,4*n+i],
       \quad value=\mathrm{coefs}[1,4*n+i].

    Examples
    --------
    >>> f = Profiles.Profile()  # Dummy empty profile, constant 0.
    >>> print f
    <moreesc.Profiles.Profile object at 0xb05e78c> empty
    >>> print f.coefs           # No coefficients in the empty profile
    []

    Attributes
    ----------
    coefs : array
      The coefficients that parametrizes the :class:`Profile` as a sequence of Bezier curves.


    """
    debug_flag = False

    def __init__(self, dtype=float):
        " Dummy constructor. See Constant, Spline and Signal classes. "
        if self.__class__.__name__ == "Profile":
            self.instants = np.empty((0), dtype=float)
            self.coefs = np.empty((0), dtype=dtype)

    def is_empty(self):
        " Check whether Profile has no coefficients. "
        return len(self.instants) == 0

    def atleast_Constant(self):
        " Fill empty Profile to zero valued Constant. "
        if self.is_empty():
            self.__class__ = Constant
            self.value = 0.

    def __copy__(self):
        tmp = Profile()
        for attr in self.__dict__:
            setattr(tmp, attr, getattr(self, attr))
        tmp.instants = self.instants.__copy__()
        tmp.coefs    = self.coefs.__copy__()
        tmp.__class__ = self.__class__
        return tmp

    def __str__(self):
        tmp = object.__repr__(self) + ' '
        N = len(self.instants)
        if N < 1:
            return tmp + "empty."
        elif (N % 4 != 0):
            return tmp + "%d point(s)." % N
        else:
            return tmp + "%d Bezier curves" % (N / 4)

    def __call__(self, t):
        """
        Evaluate the profile at various instants using the compiled functions
        :func:`(d|z)profile_eval`.

        Parameters
        ----------
        t: array-like
          Sequence of instants where to evaluate the profile.

        Returns
        -------
        out: array
          The values of the profile at those instants. Returned with the same
          shape as the input.

        Examples
        --------
        >>> print f(0.0) # Initial (at t=0) value of f
        0.0
        >>> t = np.linspace(0,1, 1024) # Time vector
        >>> print f(t) # f sampled at 1024Hz.
        array([ 0.,  0.,  0., ...,  0.,  0.,  0.])
        """
        tmp = np.asarray(np.atleast_1d(t), dtype=float).flatten('C')
        val = np.empty(tmp.shape[0], dtype=self.dtype)

        if self.is_empty():
            val[:] = 0.
        elif self.dtype == 'd':
            val = sf.dprofile_eval_vectorized(tmp, self.instants, self.coefs, val)
        elif self.dtype == 'D':
            val = sf.zprofile_eval_vectorized(tmp, self.instants, self.coefs, val)
        else:
            raise TypeError("Can not evaluate Profile with type %s." \
                % self.dtype)

        if isinstance(t, np.ndarray):
            return val.reshape(t.shape)
        elif np.isscalar(t):
            return np.asscalar(val)
    
    def integrate(self, t, tf=None):
        """
        Integrate the profile between various bounds using the compiled routines
        :func:`(d|z)profile_integrate`.

        Parameters
        ----------
        t: array-like
          Sequence of upper bounds for integrating the profile (various
          integrals are computed if t is a nontrivial array, and tf is
          not provided)
        tf: float.
          If provided, :param:`t` must be a float too, and the lower and upper
          bounds of the integration are t and tf, respectively.

        Returns
        -------
        out: array (or float if tf is provided)
          The integrals of the profile for the given upper bounds. Returned
          with the same shape as the input.
        """
        if tf is None:
            t = np.asanyarray(t, dtype=float)
        else:
            t = np.array([float(t), float(tf)], dtype=float)

        val = np.empty(t.shape, dtype=self.dtype)
        if self.is_empty():
            val = 0.
        elif self.dtype == 'd':
            val = sf.dprofile_integrate_vectorized(t, self.instants, self.coefs, val)
        elif self.dtype == 'D':
            val = sf.zprofile_integrate_vectorized(t, self.instants, self.coefs, val)
        else:
            raise TypeError("Can not integrate Profile with type %s." \
                % self.dtype)

        if tf is not None:
            return np.asscalar(val[1] - val[0])
        return val

    def debug(self, *args):
        " Print message according to debug level. "
        if self.debug_flag:
            print("DEBUG: {}".format(args))
            sys.stdout.flush()

    @property
    def dtype(self):
        " Datatype of the Profile. "
        return self.coefs.dtype

    def __getattr__(self, key):
        raise KeyError("type object %s has no attribute '%s'." \
            % (self.__class__, key))

    def __getstate__(self):
        return {'coefs': self.coefs, 'instants': self.instants}

    def __setstate__(self, dic):
        for k, v in dic.items():
            setattr(self, k, v)

    def save(self, filename):
        """
        Saves instance to file using pickle [pick]_. Note that this format
        may not be the more appropriate for exchange with other scientific
        software (see numpy and scipy I/O).

        See :func:`Profiles.load_profile` for loading.

        Examples
        --------
        >>> f.save('/tmp/profile.dat')
        >>> g = Profiles.load_profile('/tmp/profile.dat')
        >>> print g
        <moreesc.Profiles.Profile at 0xb070ccc> ...

        """
        utils._pickle(self, filename)

    def __eq__(self, other):
        if not(isinstance(other, self.__class__)):
            return False
        return np.all(self.coefs == other.coefs) \
           and np.all(self.instants == other.instants)

    def __ne__(self, other):
        return not(self.__eq__(other))

    __add__ = lambda s, o: Profile.__bfunc(s, o, np.add)
    __sub__ = lambda s, o: Profile.__bfunc(s, o, np.subtract)
    __mul__ = lambda s, o: Profile.__bfunc(s, o, np.multiply)
    __div__ = lambda s, o: Profile.__bfunc(s, o, np.divide)
    __truediv__ = __div__
    __pow__ = lambda s, o: Profile.__bfunc(s, o, np.power)
    __inv__ = lambda s: s ** (-1)

    __radd__ = lambda s, o: Profile.__bfunc(s, o, np.add)
    __rsub__ = lambda s, o: -Profile.__bfunc(s, o, np.subtract)
    __rmul__ = lambda s, o: Profile.__bfunc(s, o, np.multiply)
    __rdiv__ = lambda s, o: o * (s ** -1)
    __rtruediv__ = __rdiv__

    __iadd__ = lambda s, o: Profile.__bfunc(s, o, np.add, inplace=True)
    __isub__ = lambda s, o: Profile.__bfunc(s, o, np.subtract, inplace=True)
    __imul__ = lambda s, o: Profile.__bfunc(s, o, np.multiply, inplace=True)
    __idiv__ = lambda s, o: Profile.__bfunc(s, o, np.divide, inplace=True)
    __itruediv__ = __idiv__

    def __neg__(self):
        retobj = self.__copy__()
        if self.is_empty():
            raise ValueError("Won't operate by Empty Profile.")
        elif isinstance(self, Spline) and not(isinstance(self, Signal)):
            for el in retobj.c:
                el *= -1
            return retobj
        retobj.coefs = -retobj.coefs
        return retobj

    def __bfunc(self, other, operator, inplace=False):
        " Try to apply binary operator. "
        if isEmpty(self) or isEmpty(other):
            raise ValueError("Won't operate by Empty Profile.")
        if inplace:
            retobj = self
        else:
            retobj = self.__copy__()
        retobj.atleast_Constant()
        other = toProfile(other)
        if isinstance(retobj, Constant) and isinstance(other, Constant):
            retobj.value = operator(self.value, other.value)
            return retobj
        if isinstance(other, Constant) and (operator != np.power):
            # Operation with Constant: linear unless op is power.
            val = toScalar(other)
            retobj.coefs = operator(retobj.coefs, val)
            return retobj

        if isinstance(retobj, Constant) \
             and isinstance(other, Profile) \
             and not isinstance(other, Constant):
            # __r***__ method is called on a more complex Profile subclass
            # to directly change coefs (see above)
            return NotImplemented

        if operator in (np.add, np.subtract):
            # Operation of coefficients after generating a common mesh
            smis, omis = self._common_mesh(other)[1:3]
            tmp2 = other.__copy__()
            tmp2.atleast_Constant()

            #retobj.__class__ = Profile
            retobj._insert_single_knot(smis)
            tmp2._insert_single_knot(omis)
            retobj.coefs = operator(retobj.coefs, tmp2.coefs)
            if retobj.instants.dtype != 'd' or tmp2.instants.dtype != 'd':
                raise TypeError
            return retobj
        if isinstance(other, Profile):
            # Remaining cases: nonlinear operation
            return Profile.__appfunc(self, other, operator, inplace=inplace)
        return NotImplemented

    def __appfunc(self, other, operator, inplace=False, N=256):
        """
        Performs operation between Profiles with an approximation since the
        non linear operation may rise the degree of the Profiles (which would
        then not be represented by Bezier curve).
        Evaluates each Profile on a mesh containing :param:`N` points
        between the nodes of the minimal common mesh, performs the desired
        operation :parma:`op` element-wise on these evaluations,
        and then computes the Spline profile approximating the result of the
        operation.
        """
        if isEmpty(self) or isEmpty(other):
            raise ValueError("Won't operate by Empty Profile.")
        if inplace:
            retobj = self
        else:
            retobj = self.__copy__()
        other = toProfile(other)
        comm = retobj._common_mesh(other)[0]

        def consecutives(l):
            el0 = l[0]
            for el1 in l[1:]:
                yield (el0, el1)
                el0 = el1

        pts = [np.linspace(x, y, N+1)[:-1] for x, y in consecutives(comm)]
        pts = np.r_[np.hstack((pts)), comm[-1]]
        self.debug('Approximation: %s(%s,%s)->Profile' %
            (operator.__name__,
             self.__class__.__name__,
             other.__class__.__name__))
        spts, opts = self(pts), other(pts)
        if operator == np.divide and not(isInversible(other, opts)):
            raise ValueError("Won't invert Profile with zero crossing.")
        if operator == np.power and \
            not(isInversible(self, spts)) and np.any(opts < 0.):
            raise ValueError("No negatively power if Profile's sign changes.")

        val = operator(spts, opts)
        sig = Signal(time=pts, signal=val, smoothness=1e-10)
        # Creating simple Profile (not Signal)
        self.debug("Casting from %s to Spline." % self.__class__.__name__)
        retobj.__class__ = Spline
        for el in ['c', 't', 'k']:
            setattr(retobj, el, getattr(sig, el))
        return retobj

    def _insert_single_knot(self, missing):
        " Insert knot in Bezier curves. "
        instants, coefs = self.instants, self.coefs
        for t in np.asarray(missing):
            tl, tr = instants[0::4], instants[3::4]
            if t in tl or t in tr:
                continue
            if t < instants[0]:
                tmp0 = np.linspace(t, instants[0], 4)
                tmp1 = np.empty(tmp0.shape, dtype=self.dtype)
                tmp1[:] = coefs[0]
                if len(instants) == 1:
                    instants = tmp0
                    coefs    = tmp1
                else:
                    instants = np.r_[tmp0, instants]
                    coefs    = np.r_[tmp1, coefs]
                continue
            elif t > instants[-1]:
                tmp0 = np.linspace(instants[-1], t, 4)
                tmp1 = np.empty(tmp0.shape, dtype=self.dtype)
                tmp1[:] = coefs[-1]
                if len(instants) == 1:
                    instants = tmp0
                    coefs    = tmp1
                else:
                    instants = np.r_[instants, tmp0]
                    coefs    = np.r_[coefs,    tmp1]
                continue
            else:
                idx = np.nonzero(np.logical_and(t > tl, t < tr))[0] * 4
                # Points to move
                tmpi = instants[idx: idx + 4]
                tmpc = coefs[idx: idx + 4]
                a = (t - tmpi[0]) / (tmpi[3] - tmpi[0])
                ac = 1. - a
                # Using de Casteljau's algorithm for instants
                tmp1 = a * tmpi[1:] + ac * tmpi[:-1]
                tmp2 = a * tmp1[1:] + ac * tmp1[:-1]
                tmp3 = a * tmp2[1:] + ac * tmp2[:-1]
                # New Bezier curve on the left of the new point
                cl = np.r_[tmpi[0], tmp1[0], tmp2[0], tmp3[0]]
                # New Bezier curve on the right of the new point
                cr = np.r_[tmp3[-1], tmp2[-1], tmp1[-1], tmpi[-1]]
                # Recombination of all Bezier curves
                instants = np.r_[instants[: idx], cl, cr, instants[idx + 4:]]

                # Using de Casteljau's algorithm for coefs
                tmp1 = a * tmpc[1:] + ac * tmpc[:-1]
                tmp2 = a * tmp1[1:] + ac * tmp1[:-1]
                tmp3 = a * tmp2[1:] + ac * tmp2[:-1]
                # New Bezier curve on the left of the new point
                cl = np.r_[tmpc[0], tmp1[0], tmp2[0], tmp3[0]]
                # New Bezier curve on the right of the new point
                cr = np.r_[tmp3[-1], tmp2[-1], tmp1[-1], tmpc[-1]]
                # Recombination of all Bezier curves
                coefs = np.r_[coefs[: idx], cl, cr, coefs[idx + 4:]]

        self.instants = instants
        self.coefs = coefs

    def _common_mesh(self, other):
        """
        Return a subdivision containing knots from self and from other

        Parameters
        ----------
        self, other: Profile instances

        Returns
        -------
        comm: array
            Complete sequence as the unique union of arguments sequences
        smis, omis: arrays
            Lists of elements to insert in arguments sequence to complete them.

        """
        sorg = np.r_[self.instants[::4],  self.instants[-1]]
        oorg = np.r_[other.instants[::4], other.instants[-1]]
        comm = np.union1d(sorg, oorg)
        # Knots missing in each sequence to form common subdivision
        smis, omis = np.setdiff1d(comm, sorg), np.setdiff1d(comm, oorg)
        return comm, smis, omis
    
    @property
    def real(self):
        if self.dtype == 'd':
            return self
        retobj = self.__copy__()
        retobj.coefs = self.coefs.real
        return retobj
    
    @property
    def imag(self):
        retobj = self.__copy__()
        retobj.coefs = self.coefs.imag
        return retobj


class GroupProfiles(object):
    """
    Defines a group of Profiles that can all be evaluated in a single call.
    The coefficients of each Profile are concatenated in on big array
    (:attr:`coefs_array`), and information about their respective shapes
    are grouped in the attribute :attr:`shapes_array`.

    Individual Profiles stored within an instance of this class can be accessed
    as if :class:`GroupProfiles` is a simple list, or even as if it is a
    dictionnary if a dictionnary or keys are provided.
    """

    def __init__(self,  profiles, keys=None):
        if isinstance(profiles, dict):
            keys, profiles = profiles.keys(), profiles.values()
        elif keys is None:
            keys = np.arange(len(profiles))

        self.list = [toProfile(el) for el in profiles]
        if len(keys) != len(self.list):
            raise TypeError("Number of keys and Profiles do not match.")
        self.keys = keys
        self.__set_arrays()

    def __set_arrays(self):
        self.sizes_array = np.array([s.coefs.shape[0] for s in self.list])
        ntotal = sum(self.sizes_array)
        self.instants_array = np.zeros(ntotal, 'float')
        if any([np.iscomplexobj(s.coefs) for s in self.list]):
            self.coefs_array = np.zeros(ntotal, 'complex')
        else:
            self.coefs_array = np.zeros(ntotal, 'float')

        pos = 0
        for p in self.list:
            n = np.size(p.instants)
            self.instants_array[pos: pos + n] = p.instants
            self.coefs_array[pos: pos + n]    = p.coefs
            pos += n
        if pos != ntotal:
            tmp = pos, ntotal, self.sizes_array
            raise ValueError("Shape mismatch in GroupProfiles:\n%s" % tmp)

    @property
    def dtype(self):
        return self.coefs_array.dtype

    def __str__(self):
        return '%s at %s (%d profiles)'  \
            % (self.__class__, hex(id(self)), len(self.list))

    def __eq__(self, other):
        return isinstance(other, GroupProfiles) and \
            len(self) == len(other) and \
            np.allclose(self.sizes_array,    other.sizes_array) and \
            np.allclose(self.instants_array, other.instants_array) and \
            np.allclose(self.coefs_array,    other.coefs_array) and \
            all([self[i] == other[i] for i in range(len(self))])

    def __ne__(self, other):
        return not(self == other)

    def __copy__(self):
        return GroupProfiles([el.__copy__() for el in self.list])

    # Some arithmetics on GroupProfiles
    def __iadd__(self, other):
        for ind in range(len(self.list)):
            self.list[ind] += other
        self.__set_arrays()
        return self

    def __isub__(self, other):
        for ind in range(len(self.list)):
            self.list[ind] += -other
        self.__set_arrays()
        return self

    def __imul__(self, other):
        for ind in range(len(self.list)):
            self.list[ind] *= other
        self.__set_arrays()
        return self

    def __idiv__(self, other):
        for ind in range(len(self.list)):
            self.list[ind] /= other
        self.__set_arrays()
        return self
    __itruediv__ = __idiv__
    
    @property
    def real(self):
        return GroupProfiles([tmp.real for tmp in self.list], keys=self.keys)
    
    @property
    def imag(self):
        return GroupProfiles([tmp.imag for tmp in self.list], keys=self.keys)

    def __call__(self, t):
        """
        Evaluate the group of profile at various instants
        using the compiled functions :func:`(d|z)group_profile_eval`.

        Parameters
        ----------
        t: array-like
          Sequence of instants where to evaluate the profile.

        Returns
        -------
        out: list of arrays
          The values of the profiles at those instants. Returned with the same
          shape as the input. Each element of the list is associated to one of
          the Profiles
        """
        tmp = np.asarray(np.atleast_1d(t), dtype=float).flatten('C')
        val = np.empty((len(self), tmp.shape[0]), dtype=self.dtype)

        if self.dtype == 'd':
            val = sf.dgroup_profile_eval_vectorized(tmp, self.sizes_array, \
                self.instants_array, self.coefs_array, val)
        elif self.dtype == 'D':
            val = sf.zgroup_profile_eval_vectorized(tmp, self.sizes_array, \
                self.instants_array, self.coefs_array, val)
        else:
            raise TypeError("Can not evaluate GroupProfile with type %s." \
                % self.dtype)

        if isinstance(t, np.ndarray):
            tmp = list(t.shape)
            tmp.insert(0, len(self))
            return val.reshape(tmp)
#        elif np.isscalar(t) and np.size(val) == 1:
#            return np.asscalar(val)
        else:
            return np.atleast_1d(np.squeeze(val))

    def __len__(self):
        " Retrieve number of children. "
        return len(self.list)

    def __getitem__(self, k):
        " Retrieve one of the children. "
        if isinstance(k, int):
            return self.list[k]
        else:
            return self.list[self.keys.index(k)]

    def __setitem__(self, k, obj):
        " Change one of the children. "
        self.list[k] = toProfile(obj)
        self.__set_arrays()

    def save(self, filename):
        """
        Saves instance to fileusing pickle [pick]_.
        See :func:`Profiles.load_groupprofiles` for loading.
        """
        utils._pickle(self, filename)


class Constant(Profile):
    """
    :class:`Profile` subclass to handle constant real values.

    Attributes
    ----------
    value : float
      The numeric value of the :class:`Profiles.Constant` instance
    """

    def __init__(self, value):
        Profile.__init__(self)
        # Check "numericality"
        self.value = np.asscalar(np.array(value))

    def __call__(self, t):
        if np.isscalar(t):
            return self.value
        tmp = np.atleast_1d(t)
        values = np.empty(tmp.shape, dtype=type(self.value))
        values[:] = self.value
        return values

    @property
    def instants(self):
        " Instants array. "
        return np.array([0.])

    @instants.setter
    def instants(self, val):
        tmp = self.coefs
        self.__class__ = Profile
        self.instants = np.asarray(val)
        self.coefs    = tmp * np.zeros(self.instants.shape, dtype=self.dtype)

    @property
    def coefs(self):
        " Coefficients array. "
        return np.array([self.value])

    @coefs.setter
    def coefs(self, val):
        if np.isscalar(val) or len(val) == 1:
            self.value = np.asscalar(val)
        else:
            tmp = self.instants
            self.__class__ = Profile
            self.instants = tmp
            self.coefs    = np.asarray(val)

    def __getstate__(self):
        d = Profile.__getstate__(self)
        d.pop('instants')
        d.pop('coefs')
        d['value'] = self.value
        return d

    def __eq__(self, other):
        if np.isscalar(other):
            return self.value == other
        return Profile.__eq__(self, other)

    def __neg__(self):
        return Constant(-self.value)

    def __pow__(self, power):
        return Constant(self.value ** power)

    def __copy__(self):
        retobj = Profile.__copy__(self)
        retobj.value = 1. * self.value
        return retobj


class Linear(Profile):
    " :class:`Profile` subclass to handle linearly varying values."

    def __init__(self, instants, values):
        Profile.__init__(self)
        N = len(instants)
        instants = np.asarray(instants)
        values = np.asanyarray(values)

        if np.any(np.argsort(instants) != np.arange(N)):
            raise ValueError('Use sorted instants sequence.')

        self.instants = np.zeros(4 * (N - 1), float)
        self.instants[0::4] = instants[:-1]
        self.instants[1::4] = (2. * instants[:-1] + instants[1:]) / 3.
        self.instants[2::4] = (instants[:-1] + 2. * instants[1:]) / 3.
        self.instants[3::4] = instants[1:]

        self.coefs = np.zeros(4 * (N - 1), dtype=values.dtype)
        self.coefs[0::4] = values[:-1]
        self.coefs[1::4] = (2. * values[:-1] + values[1:]) / 3.
        self.coefs[2::4] = (values[:-1] + 2. * values[1:]) / 3.
        self.coefs[3::4] = values[1:]

    def __copy__(self):
        retobj = Profile.__copy__(self)
        retobj.instants = 1. * self.instants
        retobj.coefs = 1. * self.coefs
        return retobj


class C1_Step(Profile):
    " :class:`Profile` subclass to handle regularized steps (C1 profile)."

    def __init__(self, instants, step_value, out_of_step_value):
        Profile.__init__(self)
        assert len(instants) == 4
        instants = np.asarray(instants)

        if np.any(np.argsort(instants) != range(4)):
            raise ValueError('Use sorted instants sequence.')

        self.instants = np.empty(12, float)
        self.instants[0::4] = instants[:-1]
        self.instants[1::4] = (2. * instants[:-1] + instants[1:]) / 3.
        self.instants[2::4] = (instants[:-1] + 2. * instants[1:]) / 3.
        self.instants[3::4] = instants[1:]

        self.coefs = np.empty(12, dtype=type(step_value))
        self.coefs[:2] = self.coefs[-2:] = out_of_step_value
        self.coefs[2:-2] = step_value
SmoothStep = C1_Step

class C2_Step(Profile):
    """
    :class:`Profile` subclass to handle regularized steps (C2 profile).
    
    The rise and fall are built of three Bezier curves each ensuring the
    C2 continuity of the profile.
    """

    def __init__(self, instants, step_value, out_of_step_value):
        Profile.__init__(self)
        assert len(instants) == 4
        instants = np.asarray(instants)

        if np.any(np.argsort(instants) != range(4)):
            raise ValueError('Use sorted instants sequence.')
        tmp = np.linspace(0., 1., 4)
        tmp3 = np.r_[tmp, tmp + 1., tmp + 2.] / 3.
        
        dt2 = instants[3] - instants[2]
        dt1 = instants[2] - instants[1]
        dt0 = instants[1] - instants[0]
        
        self.instants = np.r_[
            instants[0] + dt0 * tmp3,   # 3 Bezier curves for the rise
            instants[1] + dt1 * tmp,    # 1 Bezier curve for the sustain
            instants[2] + dt2 * tmp3    # 3 Bezier curves for the decay
        ]
        
        # Sequence of coefficients ensuring the C2 continuity.
        tmp3 = np.r_[0., 0., 0., 1., 1., 2., 4., 5., 5., 6., 6., 6.] / 6.
        dv = step_value - out_of_step_value

        self.coefs = np.r_[
            out_of_step_value + dv * tmp3,
            step_value, step_value, step_value, step_value,
            step_value - dv * tmp3
        ]


class Spline(Profile):
    """
    :class:`Spline` represents a parametrization of a time-varying
    quantity. It is modelled with BSpline of degree 3, leading (under normal
    conditions) to a :math:`\mathcal{C}^2` continuous.

    Attributes
    ----------
    t,c,k :
      Tuple :math:`(t,c,k)` as used in the scipy spline library [spl]_
    """

    k = 3
    _c = None
    _t = None

    def __init__(self, tck):
        """
        Constructor for SplineProfile class.
        :param tck: a tuple with the tck coefficients of spline.
        """
        Profile.__init__(self)
        self.t = np.array(tck[0], copy=True)
        c = tck[1]
        if isinstance(c, list) and isinstance(c[0], np.ndarray):
            self.c = [np.array(tmp, copy=True) for tmp in c]
        else:
            raise ValueError('Invalid definition of Spline:\n%s' % c)
        # TODO: how to make a true copy of int (not a new reference ?)
        self.k = int(tck[2])
        self.debug("tck loaded.")

    def __eq__(self, other):
        return Profile.__eq__(self, other) \
            and len(self.c) == len(other.c) \
            and all([np.all(a==b) for a,b in zip(self.c, other.c)]) \
            and (self.t == other.t).all() \
            and self.k == other.k

    def call_as_tck(self, t):
        " Evaluate Spline using Dierckx fitpack. "
        tmp = np.asanyarray(ii.splev(t, (self.t, self.c, self.k)))
        if self.dtype == 'd':
            tmp = tmp[0]
            tmp[t < self.t[0]] = self.c[0][0]
            tmp[t > self.t[-1]] = self.c[0][-1]
        elif self.dtype == 'D':
            tmp = tmp[0] + 1.j * tmp[1]
            tmp[t < self.t[0]] = self.c[0][0] + 1.j* self.c[1][0]
            tmp[t > self.t[-1]] = self.c[0][-1] + 1.j* self.c[1][-1]
        else:
            raise TypeError("Can not evaluate Profile with type %s." \
                % self.dtype)
        return tmp

    def integrate_as_tck(self, t1, t2):
        " Integrate Spline using Dierckx fitpack. "
        tmp1, tmp2 = 0., 0.
        instants, coefs = self.instants, self.coefs
        if t1 < instants[0]:
            tmp1 = coefs[0] * (instants[0] - t1)
            t1 = instants[0]
        if t2 > instants[-1]:
            tmp2 = coefs[-1] * (t2 - instants[-1])
            t2 = instants[-1]
            
        tmp = np.asanyarray(ii.splint(t1, t2, (self.t, self.c, self.k)))
        if self.dtype == 'd':
            tmp = tmp[0]
        elif self.dtype == 'D':
            tmp = tmp[0] + 1.j * tmp[1]
        else:
            raise TypeError("Can not evaluate Profile with type %s." \
                % self.dtype)
        return np.asscalar(tmp) + tmp1 + tmp2

    def toBezier(self):
        """
        Based on Zachary Pincus script:
        http://mail.scipy.org/pipermail/scipy-dev/2007-February/006651.html ,
        transform the BSpline into a Bezier curves concatenation. It allows
        a faster evaluation (necessary for ODE solver).
        """
        if np.all(np.asarray(self._c) == np.asarray(self.c)) \
            and np.all(self._t == self.t):
            # tck has not changed, no reevaluation
            return self._instants, self._coefs

        knots_to_consider = np.unique(self.t[self.k + 1:-self.k - 1])
        new_tck = (self.t, self.c, self.k)
        # For each unique knot, bring it's multiplicity up to the next
        # multiple of k+1. This removes all continuity constraints between each
        # of the original knots, creating a set of independent Bezier curves.
        desired_multiplicity = self.k + 1
        for x in knots_to_consider:
            current_multiplicity = np.sum(np.abs(new_tck[0] - x) < 1e-6)
            remainder = current_multiplicity % desired_multiplicity
            if remainder != 0:
                # add enough knots to bring the current multiplicity
                # up to the desired multiplicity
                number_to_insert = desired_multiplicity - remainder
                new_tck = ii.insert(x, new_tck, number_to_insert, 0)
        tt, cc, kk = new_tck
        # strip off the last k+1 knots,
        # as they are redundant after knot insertion
        bezier_points = np.transpose(cc).T
        if len(self.c) == 1:  # float array
            values = np.squeeze(bezier_points)
        elif len(self.c) == 2:  # complex array
            values = np.squeeze(bezier_points[0] + 1.j * bezier_points[1])
        if values.shape[-1] > 2 * self.k:
            values = values[:-desired_multiplicity]
        instants = np.zeros(values.shape[-1], dtype=float)
        instants[0::4] = self.t[self.k:-self.k - 1]
        instants[3::4] = self.t[self.k + 1:-self.k]
        instants[1::4] = (2. * instants[0::4] + instants[3::4]) / 3.
        instants[2::4] = (instants[0::4] + 2. * instants[3::4]) / 3.
        self._t, self._c = self.t, self.c
        self._instants, self._coefs = instants, values
        return instants, values

    def summarize_seq(self):
        " Transform a raw sequence into a value-multiplicity representation. "
        seq = self.t
        val, rep = [], []
        for tmp in seq:
            idx = [abs(v - tmp) < eps for v in val]
            if sum(idx) < 1:
                val.append(tmp)
                rep.append(1)
            else:
                rep = [v[1] + v[0] for v in zip(idx, rep)]
        return np.array(val, dtype=type(seq[0])), np.array(rep, dtype=int)

    @property
    def instants(self):
        " Instants array. "
        instants, coefs = self.toBezier()
        return instants

    @instants.setter
    def instants(self, val):
        instants, coefs = self.toBezier()
        self.__class__ = Profile
        self.instants = val
        self.coefs    = coefs

    @property
    def coefs(self):
        " Coefficients array. "
        instants, coefs = self.toBezier()
        return coefs

    @coefs.setter
    def coefs(self, val):
        instants, coefs = self.toBezier()
        self.__class__ = Profile
        self.instants = instants
        self.coefs    = val

    @property
    def dtype(self):
        " Datatype of Coefficients array. "
        return self.coefs.dtype

    def __getstate__(self):
        d = Profile.__getstate__(self)
        d.pop('instants')
        d.pop('coefs')
        for att in ('t', 'c', 'k'):
            d[att] = getattr(self, att)
        return d

    def __copy__(self):
        retobj = Profile.__copy__(self)
        retobj.t = self.t.__copy__()
        retobj.c = [tmp.__copy__() for tmp in self.c]
        retobj.k = 1*int(self.k)
        return retobj

    def editor(self):
        """
        Raise a graphical user interface to manipulate the control points
        of the spline. If called from a :class:`Signal` instance, it allows
        to change the tolerance of fitting procedure.
        """
        try:
            import gtk
        except ImportError:
            raise ImportError('Need PyGTK to use editor')
        import sys, os
        sys.path.append(os.path.join(os.path.dirname(__file__), '../gui'))
        import spline_editor as se
        m = se.ModelSpline(self)
        v = se.ViewSpline()
        c = se.ControllerSpline(m, v)
        gtk.main()
        self.t = c.model.spline.t
        self.c = c.model.spline.c
        if c.model.is_signal:
            self.s = c.model.spline.s
        return c


class Signal(Spline):
    """
    :class:`Spline` represents a parametrization of a signal, for example
    a measured mouth pressure.

    Attributes
    ----------
    time, signal : arrays
      The samples of the signal to be parametrized.
    s : float
      The tolerance of fitting (also called smoothing condition in [spl]_).
    w : float or array
      Weighting coefficients (see [spl]_).
    """

    def __init__(self, time=None, signal=None, smoothness=1.):
        """
        Constructor for :class:`Signal` class.
        :param signal: the samples of the signal to be parametrized,
        :param time: the sampling instants.
        The spline curve fitting is applied in this constructor with defaut
        values of tolerance (:attr:`s`) and ponderation (:attr:`w`) but it
        can be refined by calling :meth:`fit_spline` method.
        """
        Profile.__init__(self)
        self.time = np.asanyarray(time, dtype=float)
        self.signal = np.asanyarray(signal)
        N = len(self.signal)
        self.sref = (N - np.sqrt(2 * N)) / 1000.
        self.s = float(smoothness) * self.sref
        self.w = np.ones(N) / np.std(self.signal)
        self.debug("Signal loaded.")
        self.fit_spline(s=self.s)

    def __eq__(self, other):
        return Spline.__eq__(self, other) \
            and (self.time == other.time).all() \
            and (self.signal == other.signal).all() \
            and (self.w == other.w).all() \
            and self.s == other.s \
            and self.sref == other.sref

    def fit_spline(self, s=None, w=None):
        """
        Fit a B-spline representation of the provided signal.
        For sake of simplicity in the editor, the B-spline is N-D (with N=1)
        and the parameter values are the samples instants.
        See scipy.interpolate.splprep for details on the smoothing factor s
        and the weights w.
        """
        if s != None:
            self.s = float(s)
        if w != None:
            if np.isscalar(w):
                self.w = np.ones(len(self.signal)) * float(w)
            elif len(w) == len(self.signal):
                self.w = np.array(w, dtype=float)
            else:
                raise TypeError("w argument unusable.")
        if np.iscomplexobj(self.signal):
            src = np.array(
                [self.signal.real, self.signal.imag], dtype=float)
        else:
            src = np.atleast_2d(np.asanyarray(self.signal, dtype=float))
        tmp = ii.splprep(src, u=self.time,
            k=self.k, w=self.w, s=self.s, full_output=True)
        tck, u, info = tmp[0][0], tmp[0][1], tmp[1:]
        self.t, self.c, self.k = tck
        self.debug("Curve fitted.")
        return tck, info[1:]

    def __getstate__(self):
        d = Spline.__getstate__(self)
        for att in ('s', 'w', 'time', 'signal', 'sref'):
            d[att] = getattr(self, att)
        return d


def expand_seq(values, repeats):
    " Expand a value-multiplicity representation of a sequence to a raw list."
    seq = []
    for ind in range(len(values)):
        seq.extend([values[ind]] * repeats[ind])
    return seq

load_profile = lambda s: utils._unpickle(s)
load_profile.__doc__ = utils.__pickle_common_doc \
    % {'class': 'Profile', 'output': 'Profile or subclass'}
load_profile.__name__ = 'load_profile'

load_groupprofiles = lambda s: utils._unpickle(s)
load_groupprofiles.__doc__ = utils.__pickle_common_doc \
     % {'class': 'GroupProfiles', 'output': 'GroupProfiles'}
load_groupprofiles.__name__ = 'load_groupprofiles'


def toProfile(obj):
    " Return a Profile from a scalar or a Profile "
    if isinstance(obj, Profile):
        return obj
    if np.isscalar(obj):
        return Constant(obj)
    raise TypeError("Can't convert to Profile object.")


def toScalar(obj, t=0.):
    " Return a scalar from a scalar or a Profile "
    if isinstance(obj, Profile):
        return obj(t)
    try:
        if np.isfinite(obj):
            return obj
        else:
            raise TypeError("Can not convert to scalar.")
    except TypeError:
        raise TypeError("Can not convert to scalar.")


def isEmpty(obj):
    " Check whether obj is an empty Profile. "
    return isinstance(obj, Profile) and obj.is_empty()


def isInversible(obj, values):
    " Check whether obj is an inversible Profile based on provided values. "
    if isinstance(obj, Profile):
        if obj.dtype == 'd':
            return (np.all(values > 0.) or np.all(values < 0.)) \
               and (np.all(obj.coefs > 0.) or np.all(obj.coefs < 0.))
        elif obj.dtype == 'D':
            return np.all(np.abs(values) > eps) \
               and np.all(np.abs(obj.coefs) > eps)
    else:
        return np.isfinite(obj) and obj != 0.
