#! /usr/bin/python
# -*- coding: utf-8 -*-
# Copyright Fabrice Silva, 2012.
# cython: profile=True

"""
"""

import cython
import numpy as np
cimport numpy as np
cimport moreesc.c_profiles as cp
cimport moreesc.c_acous_sys as ca
cimport moreesc.c_mech_sys as cm

cdef extern from "math.h":
     np.float64_t sqrt(np.float64_t)

cdef np.float64_t twodivbyrho, reed_motion_coef

cpdef np.float64_t get_twodivbyrho():
    return twodivbyrho

cpdef set_twodivbyrho(np.float_t rho):
    global twodivbyrho
    twodivbyrho = 2. / rho

cpdef get_reed_motion_coef():
    return reed_motion_coef

cpdef set_reed_motion_coef(np.float_t value):
    global reed_motion_coef
    reed_motion_coef = value

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim = 1] global_sys(
        np.float64_t t,
        np.ndarray[np.float64_t, ndim=1] X,
        np.ndarray[np.float64_t, ndim=1] dX,
        # Poles and residues of the acoustic resonator
        np.ndarray[np.complex128_t, ndim=1] sn,
        np.ndarray[np.complex128_t, ndim=1] cn,
        # Numerator and denominator of the valve
        np.ndarray[np.float64_t, ndim=1] bn,
        np.ndarray[np.float64_t, ndim=1] an,
        # Other parameters: (pm, h0, w)
        np.ndarray[np.float64_t, ndim=1] params
        ):
    cdef int nm
    cdef np.float64_t u, p, h, pm, h0
    cdef np.ndarray[np.float64_t, ndim=1] Xm, dXm
    cdef np.ndarray[np.complex128_t, ndim=1] Xa, dXa
    nm = sn.shape[0]
    # The following partial state vector (and derivatives) are views
    Xa = c_get_acoustic_state_vector(nm, X)
    dXa = c_get_acoustic_state_vector(nm, dX)
    Xm = c_get_mechanic_state_vector(nm, X)
    dXm = c_get_mechanic_state_vector(nm, dX)

    # Evaluating parameters
    pm = params[0]
    h0 = params[1]
    
    p = ca.c_get_pressure(Xa)
    h = cm.c_get_opening(Xm)
    v = cm.c_get_velocity(Xm, an)
    #h = cm.c_get_opening(Xm, bn)
    #v = cm.c_get_velocity(Xm, bn)
    u = c_nl_coupling(pm-p, h, v)

    ca.acous_sys(t, Xa, dXa, u, sn, cn)
    cm.mech_sys(t, Xm, dXm, p-pm, h0, bn, an)
    # No copies to dX are needed, as dXa and dXm are views
    return dX

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim = 1] global_sys_profiles(
        np.float64_t t,
        np.ndarray[np.float64_t, ndim=1] X,
        np.ndarray[np.float64_t, ndim=1] dX,
        # Poles of the acoustic resonator
        np.ndarray[np.int_t, ndim=1] ssn,
        np.ndarray[np.float64_t, ndim=1] isn,
        np.ndarray[np.complex128_t, ndim=1] psn,
        # Residues of the acoustic resonator
        np.ndarray[np.int_t, ndim=1] scn,
        np.ndarray[np.float64_t, ndim=1] icn,
        np.ndarray[np.complex128_t, ndim=1] pcn,
        # Numerator of the valve
        np.ndarray[np.int_t, ndim=1] sbn,
        np.ndarray[np.float64_t, ndim=1] ibn,
        np.ndarray[np.float64_t, ndim=1] pbn,
        # Denominator of the valve
        np.ndarray[np.int_t, ndim=1] san,
        np.ndarray[np.float64_t, ndim=1] ian,
        np.ndarray[np.float64_t, ndim=1] pan,
        # Other parameters: (pm, h0, w)
        np.ndarray[np.int_t, ndim=1] son,
        np.ndarray[np.float64_t, ndim=1] ion,
        np.ndarray[np.float64_t, ndim=1] pon
        ):
    cdef int nm,np0, np1
    cdef np.float64_t u, p, h, pm
    cdef np.ndarray[np.float64_t, ndim=1] params, ih0, ph0
    cdef np.ndarray[np.float64_t, ndim=1] Xm, dXm
    cdef np.ndarray[np.complex128_t, ndim=1] Xa, dXa
    nm = ssn.shape[0]
    # The following partial state vector (and derivatives) are views
    Xa = c_get_acoustic_state_vector(nm, X)
    dXa = c_get_acoustic_state_vector(nm, dX)
    Xm = c_get_mechanic_state_vector(nm, X)
    dXm = c_get_mechanic_state_vector(nm, dX)

    # Evaluating parameters
    # TODO : how to guarantee the order of the parameters?
    params = np.empty(son.shape[0], dtype=np.float64)
    cp.dgroup_profile_eval(t, son, ion, pon, params)
    pm = params[0]
    
    p = ca.c_get_pressure(Xa)
    h = cm.c_get_opening(Xm)
    v = cm.c_get_velocity_profiles(Xm, t, san, ian, pan)
    #h = cm.c_get_opening_profiles(Xm, t, sbn, ibn, pbn)
    #v = cm.c_get_velocity_profiles(Xm, t, sbn, ibn, pbn)
    u = c_nl_coupling(pm-p, h, v)
    
    # Extract h0 profile
    np0 = son[0]
    np1 = son[1] + np0
    ih0 = ion[np0: np1]
    ph0 = pon[np0: np1]

    ca.acous_sys_profiles(t, Xa, dXa, u, ssn, isn, psn, scn, icn, pcn)
    cm.mech_sys_profiles(t, Xm, dXm, p-pm, sbn, ibn, pbn, san, ian, pan, ih0, ph0)
    # No copies to dX are needed, as dXa and dXm are views
    return dX

# According to vode manual, only an approximation of the jacobian is required.
# Only the contribution of the acoustic resonator poles and the proper
# dynamics of the valve is set.
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim = 2] global_jac(
        np.float64_t t,
        np.ndarray[np.float64_t, ndim=1] X,
        np.ndarray[np.float64_t, ndim=2] jX,
        # Poles and residues of the acoustic resonator
        np.ndarray[np.complex128_t, ndim=1] sn,
        np.ndarray[np.complex128_t, ndim=1] cn,
        # Numerator and denominator of the valve
        np.ndarray[np.float64_t, ndim=1] bn,
        np.ndarray[np.float64_t, ndim=1] an
        ):
    cdef int nm, na, n1, indn1
    cdef np.complex128_t pole, residue
    nm = sn.shape[0]

    # The supplied array is preset to zero.
    # Acoustic part of the jacobian
    for n1 in xrange(0, nm):
        # acoustic - acoustic
        indn1 = 2 * n1
        pole = sn[n1]
        residue = cn[n1]
        jX[indn1, indn1] = pole.real
        jX[indn1, indn1 + 1] = -pole.imag
        jX[indn1 + 1, indn1] = pole.imag
        jX[indn1 + 1, indn1 + 1] = pole.real
        
    # Mechanic part of the jacobian
    nb = bn.shape[0]
    na = an.shape[0] - 1
    # mechanic - mechanic
    for n1 in xrange(na):
        indn1 = 2 * nm + n1
        jX[indn1, 2 * nm] = -an[n1 + 1]
        if n1 != na - 1:
            # for all components except the last one
            jX[indn1, indn1 + 1] = 1.
    return jX

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim = 2] global_jac_profiles(
        np.float64_t t,
        np.ndarray[np.float64_t, ndim=1] X,
        np.ndarray[np.float64_t, ndim=2] jX,
        # Poles of the acoustic resonator
        np.ndarray[np.int_t, ndim=1] ssn,
        np.ndarray[np.float64_t, ndim=1] isn,
        np.ndarray[np.complex128_t, ndim=1] psn,
        # Residues of the acoustic resonator
        np.ndarray[np.int_t, ndim=1] scn,
        np.ndarray[np.float64_t, ndim=1] icn,
        np.ndarray[np.complex128_t, ndim=1] pcn,
        # Numerator of the valve
        np.ndarray[np.int_t, ndim=1] sbn,
        np.ndarray[np.float64_t, ndim=1] ibn,
        np.ndarray[np.float64_t, ndim=1] pbn,
        # Denominator of the valve
        np.ndarray[np.int_t, ndim=1] san,
        np.ndarray[np.float64_t, ndim=1] ian,
        np.ndarray[np.float64_t, ndim=1] pan,
        ):
    cdef int nm,np0, np1
    
    cdef np.ndarray[np.complex128_t, ndim = 1] sn, cn
    cdef np.ndarray[np.float64_t, ndim = 1] bn, an

    bn = np.empty(sbn.shape[0], dtype=np.float64)
    an = np.empty(san.shape[0], dtype=np.float64)
    sn = np.empty(ssn.shape[0], dtype=np.complex128)
    cn = np.empty(scn.shape[0], dtype=np.complex128)

    cp.dgroup_profile_eval(t, sbn, ibn, pbn, bn)
    cp.dgroup_profile_eval(t, san, ian, pan, an)
    cp.zgroup_profile_eval(t, ssn, isn, psn, sn)
    cp.zgroup_profile_eval(t, scn, icn, pcn, cn)
    return global_jac(t, X, jX, sn, cn, bn, an)

# Non linear coupling equation: flow = function(X)
def nl_coupling(dp, h, v):
    " Bernoulli based flow, plus reed motion induced flow "
    global twodivbyrho, reed_motion_coef
    type_flag = type(dp)
    dp = np.atleast_1d(np.asarray(dp, dtype=float).ravel())
    h = np.atleast_1d(np.asarray(h, dtype=float).ravel())

    result = np.nan + np.empty_like(dp)
    idx = h >= 0.
    result[h < 0.] = 0.

    arg = dp * twodivbyrho
    idxdp = np.logical_and(idx, dp >= 0.)
    result[idxdp] = h[idxdp] * np.sqrt(arg[idxdp])
    idxdp = np.logical_and(idx, dp < 0.)
    result[idxdp] = -h[idxdp] * np.sqrt(-arg[idxdp])

    # Reed motion induced flow
    result += reed_motion_coef * v
    if type_flag is np.ndarray:
        return result
    return np.asscalar(result)

cdef np.float64_t c_nl_coupling(
        np.float64_t dp,
        np.float64_t h,
        np.float64_t v):
    global twodivbyrho
    cdef np.float64_t arg, retval

    if h < 0.:
        return 0.

    arg = dp * twodivbyrho
    if dp > 0.:
        retval = h * sqrt(arg)
    else:
        retval = -h * sqrt(-arg)
    retval += reed_motion_coef * v
    return retval

## C versions work on 1D arrays, Python ones can operate on ND arrays

def get_acoustic_state_vector(int nm, X):
    """
    Retrieve the acoustic state vector (may be a copy, and not a view).
    """
    Xa = np.require(X[: 2 * nm, ...], requirements=['F',])
    return Xa.view(dtype=np.complex128)

@cython.boundscheck(False)
cdef np.ndarray[np.complex128_t, ndim = 1] c_get_acoustic_state_vector(
        int nm, np.ndarray[np.float64_t, ndim=1] X):
    return X[: 2 * nm].view(dtype=np.complex128)

def set_acoustic_state_vector(int nm, X, Xa):
    " Affect a complex valued vector to the acoustic state vector. "
    X[: 2 * nm, ...] = Xa.view(dtype=np.float64)
    return X

@cython.boundscheck(False)
cdef np.ndarray[np.float64_t, ndim = 1] c_set_acoustic_state_vector(int nm,
        np.ndarray[np.float64_t, ndim=1] X,
        np.ndarray[np.complex128_t, ndim=1] Xa):
    X[: 2 * nm] = Xa.view(dtype=np.float64)
    return X

def get_mechanic_state_vector(int nm, X):
    " Retrieve the mechanic state vector. "
    return X[2 * nm:, ...]

@cython.boundscheck(False)
cdef np.ndarray[np.float64_t, ndim = 1] c_get_mechanic_state_vector(
        int nm, np.ndarray[np.float64_t, ndim=1] X):
    return X[2 * nm:]

def set_mechanic_state_vector(int nm, X, Xm):
    " Affect a real valued vector to the mechanic state vector. "
    X[2 * nm:, ...] = Xm
    return X

@cython.boundscheck(False)
cdef np.ndarray[np.float64_t, ndim = 1] c_set_mechanic_state_vector(
        int nm,
        np.ndarray[np.float64_t, ndim=1] X,
        np.ndarray[np.float64_t, ndim=1] Xm):
    X[2 * nm:] = Xm
    return X

