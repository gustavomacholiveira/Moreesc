#! /usr/bin/python
# -*- coding: utf-8 -*-
# Copyright Fabrice Silva, 2012.
# cython: profile=True

"""
"""

import cython
import numpy as np
cimport numpy as np
from moreesc.c_profiles cimport dprofile_eval
from moreesc.c_profiles cimport dgroup_profile_eval


DTYPE = np.float64
ZTYPE = np.complex128
ctypedef np.float64_t DTYPE_t
ctypedef np.complex128_t ZTYPE_t

# Coefficient controlling the additional force due to contact between
# the mobile and the fixed parts of the valve (dimensionless factor to be 
# multipied by the mobile part's stiffness).
cdef np.float64_t beating_factor

cpdef np.float64_t get_beating_factor():
    return beating_factor

cpdef set_beating_factor(np.float64_t value):
    global beating_factor
    beating_factor = value

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[DTYPE_t, ndim = 1] mech_sys(
        np.float_t t,
        np.ndarray[DTYPE_t, ndim=1] X,
        np.ndarray[DTYPE_t, ndim=1] dX,
        np.float_t p, np.float_t h0,
        np.ndarray[DTYPE_t, ndim=1] bn,
        np.ndarray[DTYPE_t, ndim=1] an):
    """
    Transfer function for the valve
           sum_m=0^M num_m s^(M-m)
    H(s)= ------------------------- P(s) with den_0=1
           sum_n=0^N den_n s^(N-n)
    
    Differential system may be written as DX=AX+Bp et h=CX with
        X_n (n in [0, N-1]) is the n-th derivative of h(t),
        A NxN matrix: 1° overdiag with '1' and -den(1:N) as last row
        B Nx1 column: 1 at B(-1), 0 elsewhere
        C 1xN row: num(M:0:-1) in B(0:M), 0 elsewhere
        Controller canonical form of the state-space representation.

    The content of the state vector is thus the successive derivatives of
    the response for M=0

    One must specify the numerator and denominator polynomials coefficients.
    """
    cdef np.int_t indn, N, M
    cdef DTYPE_t h

    N = an.shape[0] - 1
    M = bn.shape[0] - 1

    assert N == X.shape[0]
    assert N == dX.shape[0]

    for indn in xrange(N-1):
        dX[indn] = X[indn+1]
    dX[N-1] = -(an[-1:0:-1] * X[:N-1]).sum() + p

    # Beating valve: preventing negative aperture
    h = c_get_opening(X, bn)
    if h < 0.:
        dX[N-1] -= an[N] * beating_factor * h
    return dX

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[DTYPE_t, ndim = 1] mech_sys_profiles(
        np.float_t t,
        np.ndarray[DTYPE_t, ndim=1] X,
        np.ndarray[DTYPE_t, ndim=1] dX,
        np.float_t p,
        np.ndarray[np.int_t, ndim=1] sbn,
        np.ndarray[DTYPE_t, ndim=1] ibn,
        np.ndarray[DTYPE_t, ndim=1] pbn,
        np.ndarray[np.int_t, ndim=1] san,
        np.ndarray[DTYPE_t, ndim=1] ian,
        np.ndarray[DTYPE_t, ndim=1] pan,
        np.ndarray[DTYPE_t, ndim=1] ih0,
        np.ndarray[DTYPE_t, ndim=1] ph0):
    """
    See mech_sys docstring for documentation of the transfer function 
    implementation.

    For the numerator and denominator polynomials, one must specify:
     - the GroupProfiles sizes_array:    (sbn and san)
     - the GroupProfiles instants_array: (ibn and ian)
     - the GroupProfiles coefs_array:    (pbn and pan)
    """
    cdef DTYPE_t h0
    cdef np.ndarray[DTYPE_t, ndim = 1] bn, an

    h0 = dprofile_eval(t, ih0, ph0)
    bn = np.empty(sbn.shape[0], dtype=DTYPE)
    an = np.empty(san.shape[0], dtype=DTYPE)
    dgroup_profile_eval(t, sbn, ibn, pbn, bn)
    dgroup_profile_eval(t, san, ian, pan, an)

    return mech_sys(t, X, dX, p, h0, bn, an)

# Convenience function available in order not to require the user to know
# the constitution of the state vector.
def get_opening(X, bn):
    " Extract the opening value from the state vector. "
    M = bn.shape[-1]
    return np.sum(X[:, :M] * bn[:, ::-1])

cdef np.float_t c_get_opening(np.ndarray[np.float_t, ndim=1] X,
        np.ndarray[DTYPE_t, ndim=1] bn):
    " Extract the opening value from the state vector. "
    cdef int M
    M = len(bn)
    return np.sum(X[:M] * bn[::-1])

def get_velocity(X, bn):
    " Extract the velocity value from the state vector. "
    M = bn.shape[-1]
    return np.sum(X[:, 1:M + 1] * bn[:, ::-1])

cdef np.float_t c_get_velocity(np.ndarray[np.float_t, ndim=1] X,
                               np.ndarray[DTYPE_t, ndim=1] bn):
    " Extract the velocity value from the state vector. "
    cdef int M
    M = len(bn)
    return np.sum(X[1:M + 1] * bn[::-1])

cdef np.float_t c_get_opening_profiles(
        np.ndarray[np.float_t, ndim=1] X,
        np.float_t t,
        np.ndarray[np.int_t, ndim=1] sbn,
        np.ndarray[DTYPE_t, ndim=1] ibn,
        np.ndarray[DTYPE_t, ndim=1] pbn):
    " Extract the velocity value from the state vector. "
    cdef int M
    cdef np.ndarray[DTYPE_t, ndim = 1] bn
    cdef np.ndarray[DTYPE_t, ndim = 1] v
    bn = np.empty(sbn.shape[0], dtype=DTYPE)
    dgroup_profile_eval(t, sbn, ibn, pbn, bn)
    M = len(bn)
    v = np.sum(X[:, :M] * bn[:, ::-1])

cdef np.float_t c_get_velocity_profiles(
        np.ndarray[np.float_t, ndim=1] X,
        np.float_t t,
        np.ndarray[np.int_t, ndim=1] sbn,
        np.ndarray[DTYPE_t, ndim=1] ibn,
        np.ndarray[DTYPE_t, ndim=1] pbn):
    " Extract the velocity value from the state vector. "
    cdef int M
    cdef np.ndarray[DTYPE_t, ndim = 1] bn
    cdef np.ndarray[DTYPE_t, ndim = 1] v
    bn = np.empty(sbn.shape[0], dtype=DTYPE)
    dgroup_profile_eval(t, sbn, ibn, pbn, bn)
    M = len(bn)
    v = np.sum(X[:, 1:M + 1] * bn[:, ::-1])
