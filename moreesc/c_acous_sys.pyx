#! /usr/bin/python
# -*- coding: utf-8 -*-
# Copyright Fabrice Silva, 2012.
# cython: profile=True

"""
"""

import cython
import numpy as np
cimport numpy as np
from moreesc.c_profiles cimport zgroup_profile_eval


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.complex128_t, ndim=1] acous_sys(
        np.float_t t,
        np.ndarray[np.complex128_t, ndim=1] X,
        np.ndarray[np.complex128_t, ndim=1] dX,
        np.float_t u,
        np.ndarray[np.complex128_t, ndim=1] sn,
        np.ndarray[np.complex128_t, ndim=1] cn):
    """
    Transfer function for the acoustic resonator
        P(s) = sum_n=0^(N-1) ( Cn/(s-sn) + Cn*/(s-sn*)) U(s)
    where * denotes complex conjugate

    Differential system may be written considering the complex-typed
    components Pn(s) of the pressure defined by
        Pn(s) = Cn/(s-sn) U(s)  i.e. dp_n(t) = Cn*u(t)+sn*p_n(t)
    and p(t) = 2*sum_{Im(s_n)>0} (Re p_n)(t)+sum_{Im(s_n)=0} p_n(t)

    For poles sn and residues Cn, one must specify them as arrays.
    """
    cdef np.int_t indn, nm

    nm = X.shape[0]
    assert nm == dX.shape[0]
    for indn in xrange(nm):
        dX[indn] = cn[indn] * u + sn[indn] * X[indn]
    return dX


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.complex128_t, ndim=1] acous_sys_profiles(
        np.float_t t,
        np.ndarray[np.complex128_t, ndim=1] X,
        np.ndarray[np.complex128_t, ndim=1] dX,
        np.float_t u,
        np.ndarray[np.int_t, ndim=1] ssn,
        np.ndarray[np.float_t, ndim=1] isn,
        np.ndarray[np.complex128_t, ndim=1] psn,
        np.ndarray[np.int_t, ndim=1] scn,
        np.ndarray[np.float_t, ndim=1] icn,
        np.ndarray[np.complex128_t, ndim=1] pcn):
    """
    See acous_sys docstring for documentation of the modal expansion
    implementation of the acoustic resonator.

    For poles sn and residues Cn, one must specify:
     - the GroupProfiles sizes_array:    (ssn and scn)
     - the GroupProfiles instants_array: (isn and icn)
     - the GroupProfiles coefs_array:    (psn and pcn)
    """
    cdef np.int_t indn, nm
    cdef np.ndarray[np.complex128_t, ndim = 1] sn, cn

    sn = np.empty(ssn.shape[0], dtype=np.complex128)
    cn = np.empty(scn.shape[0], dtype=np.complex128)
    zgroup_profile_eval(t, ssn, isn, psn, sn)
    zgroup_profile_eval(t, scn, icn, pcn, cn)
    return acous_sys(t, X, dX, u, sn, cn)

def get_pressure(X):
    " Extract the pressure value from the state vector. "
    return 2. * X.real.sum(axis=0)

cdef np.float_t c_get_pressure(
        np.ndarray[np.complex128_t, ndim=1] X):
    " Extract the pressure value from the state vector. "
    return 2. * X.real.sum()

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.complex128_t, ndim = 1] modal_impedance(
    np.ndarray[np.complex128_t, ndim = 1] s,
    np.ndarray[np.complex128_t, ndim = 1] sn,
    np.ndarray[np.complex128_t, ndim = 1] cn,
    np.ndarray[np.complex128_t, ndim = 1] Z):
    """
    evaluate the modal impedance with complex modes (sn,Cn) at given
    complex frequencies s=alpha+j*omega (Laplace variable).
    """
    cdef np.int_t nm, indn
    cdef np.complex128_t vsn, vcn

    nm = sn.shape[0]
    assert nm == cn.shape[0]
    assert s.shape[0] == Z.shape[0]

    for indn in xrange(s.shape[0]):
        Z[indn] = 0.j

    for indn in xrange(nm):
        vsn = sn[indn]
        vcn = cn[indn]
        Z += vcn / (s - vsn)
        Z += np.conj(vcn) / (s - np.conj(vsn))
    return Z

