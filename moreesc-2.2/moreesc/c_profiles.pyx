#! /usr/bin/python
# -*- coding: utf-8 -*-
# Copyright Fabrice Silva, 2012.
# cython: profile=True

"""
"""

cimport cython
import numpy as np
cimport numpy as np


DTYPE = np.float64
ZTYPE = np.complex128
ctypedef np.float64_t DTYPE_t
ctypedef np.complex128_t ZTYPE_t

cdef extern from "math.h":
    bint isnan(DTYPE_t x)
cdef inline bint zisnan(ZTYPE_t x):
    return isnan(x.real) or isnan(x.imag)

# Double precision Profiles functions
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef DTYPE_t dprofile_eval(DTYPE_t t,
        np.ndarray[DTYPE_t, ndim=1] inst,
        np.ndarray[DTYPE_t, ndim=1] pts) except -1:
    """
    evaluates the real value of a slowly varying quantity
    according to a Bezier curve parametrization.

    Parameters
    ----------
    t: scalar
      Time instant where the profile is to be evaluated
    inst, pts: arrays
      Instants and coefficients arrays defining the profile.
    """
    cdef np.int_t nbpts, i4
    cdef DTYPE_t w, tmp

    nbpts = inst.shape[0]
    assert nbpts == pts.shape[0]

    if nbpts == 0:
        return 0.
    elif nbpts == 1:
        return pts[0]

    assert nbpts % 4 == 0
    if t <= inst[0]:
        return pts[0]
    elif t >= inst[nbpts - 1]:
        return pts[nbpts - 1]
    for i4 in xrange(0, nbpts, 4):
        if t >= inst[i4 + 3]:
            continue
        w = (t - inst[i4]) / (inst[i4 + 3] - inst[i4])
        tmp = pts[i4 + 3] - pts[i4] - 3. * (pts[i4 + 2] - pts[i4 + 1])
        tmp = w * tmp + 3. * (pts[i4 + 2] - 2. * pts[i4 + 1] + pts[i4])
        tmp = w * tmp + 3. * (pts[i4 + 1] - pts[i4])
        tmp = w * tmp + pts[i4]
        return tmp

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[DTYPE_t, ndim=1] dprofile_eval_vectorized(
        np.ndarray[DTYPE_t, ndim=1] t,
        np.ndarray[DTYPE_t, ndim=1] inst,
        np.ndarray[DTYPE_t, ndim=1] pts,
        np.ndarray[DTYPE_t, ndim=1] result):
    """
    evaluates the real values of a slowly varying quantity
    according to a Bezier curve parametrization.

    Parameters
    ----------
    t: array-like
      Time instants where the profile is to be evaluated
    inst, pts: arrays
      Instants and coefficients arrays defining the profile.
    """
    cdef np.int_t nt, indt

    nt = t.shape[0]
    assert nt == result.shape[0]

    for indt in xrange(nt):
        result[indt] = dprofile_eval(t[indt], inst, pts)
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef DTYPE_t dprofile_integrate(
        DTYPE_t t,
        np.ndarray[DTYPE_t, ndim=1] inst,
        np.ndarray[DTYPE_t, ndim=1] pts):
    """
    integrate the real valued slowly varying profile
    from its initial instant to the time instant specified in argument.

    Parameters
    ----------
    t: scalar
      Upper bound of the time integral.
    inst, pts: arrays
      Instants and coefficients arrays defining the profile.
    """
    cdef np.int_t nbpts, i4
    cdef DTYPE_t w, tmp, result

    nbpts = inst.shape[0]
    assert nbpts == pts.shape[0]

    if nbpts == 0:
        return 0.
    elif nbpts == 1:
        return (t - inst[0]) * pts[0]
    elif t <= inst[0]:
        return pts[0] * (t - inst[0])

    assert nbpts % 4 == 0

    result = 0.
    for i4 in xrange(0, nbpts, 4):
        if t >= inst[i4 + 3]:
            # Full interval
            tmp = pts[i4] + pts[i4 + 1] + pts[i4 + 2] + pts[i4 + 3]
            result += .25 * tmp * (inst[i4 + 3] - inst[i4])
        else:
            # Partial interval
            w = (t - inst[i4]) / (inst[i4 + 3] - inst[i4])
            tmp = pts[i4 + 3]
            tmp = w * tmp + pts[i4 + 2] * (4. - 3. * w)
            tmp = w * tmp + pts[i4 + 1] * (6. + w * (-8. + 3. * w))
            tmp = w * tmp + pts[i4] * (4. + w * (-6. + w * (4. - w)))
            result += .25 * tmp * (t - inst[i4])
            break
    else:
        # Looped over all the intervals without breaking the loop
        result += pts[nbpts - 1] * (t - inst[nbpts - 1])
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[DTYPE_t, ndim=1] dprofile_integrate_vectorized(
        np.ndarray[DTYPE_t, ndim=1] t,
        np.ndarray[DTYPE_t, ndim=1] inst,
        np.ndarray[DTYPE_t, ndim=1] pts,
        np.ndarray[DTYPE_t, ndim=1] result):
    """
    integrate the real valued slowly varying profile
    from its initial instant to the time instant specified in argument.

    Parameters
    ----------
    t: scalar or array-like
      Upper bound of the time integral.
    inst, pts: arrays
      Instants and coefficients arrays defining the profile.
    """
    cdef np.int_t nt, nbpts, indt, i, i4
    cdef DTYPE_t ti, w, tmp
    cdef np.ndarray[DTYPE_t, ndim = 1] store

    nt = t.shape[0]
    nbpts = inst.shape[0]
    assert nbpts == pts.shape[0]
    assert nt == result.shape[0]

    if nbpts == 0:
        result[:] = 0.
    elif nbpts ==1:
        for indt in xrange(nt):
            result[indt] = (t[indt] - inst[0]) * pts[0]
    if nbpts <= 1:
        return result

    assert nbpts % 4 == 0
    store = np.empty(nbpts/4, dtype=DTYPE) + np.nan

    for indt in xrange(nt):
        ti = t[indt]
        if ti <= inst[0]:
            result[indt] = pts[0] * (ti - inst[0])
            continue

        result[indt] = 0.
        i = 0
        for i4 in xrange(0, nbpts, 4):
            if ti >= inst[i4 + 3]:
                # Full interval
                if isnan(store[i]):
                    tmp = pts[i4] + pts[i4 + 1] + pts[i4 + 2] + pts[i4 + 3]
                    store[i] = .25 * tmp * (inst[i4 + 3] - inst[i4])
                result[indt] += + store[i]
            else:
                # Partial interval
                w = (ti - inst[i4]) / (inst[i4 + 3] - inst[i4])
                tmp = pts[i4 + 3]
                tmp = w * tmp + pts[i4 + 2] * (4. - 3. * w)
                tmp = w * tmp + pts[i4 + 1] * (6. + w * (-8. + 3. * w))
                tmp = w * tmp + pts[i4] * (4. + w * (-6. + w * (4. - w)))
                result[indt] += .25 * tmp * (ti - inst[i4])
                break
            i += 1
        else:
            # Looped over all the intervals without breaking the loop
            result[indt] += pts[nbpts - 1] * (ti - inst[nbpts - 1])
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[DTYPE_t, ndim = 1] dgroup_profile_eval(
        DTYPE_t t,
        np.ndarray[np.int_t, ndim=1] sizes,
        np.ndarray[DTYPE_t, ndim=1] inst,
        np.ndarray[DTYPE_t, ndim=1] pts,
        np.ndarray[DTYPE_t, ndim=1] result):
    """
    evaluates the real values of a group of profiles.

    Parameters
    ----------
    t: scalar
      Time instants where the profile is to be evaluated
    sizes: intergers-array
      The array of the sizes of each profiles of the group.
    inst, pts: arrays
      Instants and coefficients arrays defining the group profile.
    """
    cdef np.int_t nt, nprof, npts, indp, idx

    nprof = sizes.shape[0]
    npts = inst.shape[0]
    assert npts == pts.shape[0]
    assert nprof == result.shape[0]

    idx = 0
    for indp in xrange(nprof):
        npts = sizes[indp]
        result[indp] = dprofile_eval(t, inst[idx: idx + npts], 
                                        pts[idx: idx + npts])
        idx += npts
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[DTYPE_t, ndim = 2] dgroup_profile_eval_vectorized(
        np.ndarray[DTYPE_t, ndim=1] t,
        np.ndarray[np.int_t, ndim=1] sizes,
        np.ndarray[DTYPE_t, ndim=1] inst,
        np.ndarray[DTYPE_t, ndim=1] pts,
        np.ndarray[DTYPE_t, ndim = 2] result):
    """
    evaluates the real value(s) of a group of profiles.

    Parameters
    ----------
    t: scalar or array-like
      Time instants where the profile is to be evaluated
    sizes: intergers-array
      The array of the sizes of each profiles of the group.
    inst, pts: arrays
      Instants and coefficients arrays defining the group profile.
    """
    cdef np.int_t nt, nprof, npts, indp, idx

    nprof = sizes.shape[0]
    npts = inst.shape[0]
    assert npts == pts.shape[0]
    assert nprof == result.shape[0]

    idx = 0
    for indp in xrange(nprof):
        npts = sizes[indp]
        dprofile_eval_vectorized(t, inst[idx: idx + npts], pts[idx: idx + npts],
                      result[indp, :])
        idx += npts
    return result


# Double complex Profiles functions

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef ZTYPE_t zprofile_eval(DTYPE_t t,
        np.ndarray[DTYPE_t, ndim=1] inst,
        np.ndarray[ZTYPE_t, ndim=1] pts):
    """
    evaluates the real value of a slowly varying quantity
    according to a Bezier curve parametrization.

    Parameters
    ----------
    t: scalar
      Time instant where the profile is to be evaluated
    inst, pts: arrays
      Instants and coefficients arrays defining the profile.
    """
    cdef np.int_t nbpts, i4
    cdef DTYPE_t w
    cdef ZTYPE_t tmp

    nbpts = inst.shape[0]
    assert nbpts == pts.shape[0]

    if nbpts == 0:
        return 0.j
    elif nbpts == 1:
        return pts[0]

    assert nbpts % 4 == 0
    if t <= inst[0]:
        return pts[0]
    elif t >= inst[nbpts - 1]:
        return pts[nbpts - 1]
    for i4 in xrange(0, nbpts, 4):
        if t >= inst[i4 + 3]:
            continue
        w = (t - inst[i4]) / (inst[i4 + 3] - inst[i4])
        tmp = pts[i4 + 3] - pts[i4] - 3. * (pts[i4 + 2] - pts[i4 + 1])
        tmp = w * tmp + 3. * (pts[i4 + 2] - 2. * pts[i4 + 1] + pts[i4])
        tmp = w * tmp + 3. * (pts[i4 + 1] - pts[i4])
        tmp = w * tmp + pts[i4]
        return tmp


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[ZTYPE_t, ndim=1] zprofile_eval_vectorized(
        np.ndarray[DTYPE_t, ndim=1] t,
        np.ndarray[DTYPE_t, ndim=1] inst,
        np.ndarray[ZTYPE_t, ndim=1] pts,
        np.ndarray[ZTYPE_t, ndim=1] result):
    """
    evaluates the complex value(s) of a slowly varying quantity
    according to a Bezier curve parametrization.

    Parameters
    ----------
    t: scalar or array-like
      Time instants where the profile is to be evaluated
    inst, pts: arrays
      Instants and coefficients arrays defining the profile.
    """
    cdef np.int_t nt, indt

    nt = t.shape[0]
    assert nt == result.shape[0]

    for indt in xrange(nt):
        result[indt] = zprofile_eval(t[indt], inst, pts)
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef ZTYPE_t zprofile_integrate(
        DTYPE_t t,
        np.ndarray[DTYPE_t, ndim=1] inst,
        np.ndarray[ZTYPE_t, ndim=1] pts):
    """
    integrate the complex valued slowly varying profile
    from its initial instant to the time instant specified in argument.

    Parameters
    ----------
    t: scalar
      Upper bound of the time integral.
    inst, pts: arrays
      Instants and coefficients arrays defining the profile.
    """
    cdef np.int_t nbpts, i4
    cdef DTYPE_t ti, w
    cdef ZTYPE_t tmp, result

    nbpts = inst.shape[0]
    assert nbpts == pts.shape[0]

    if nbpts == 0:
        return 0.
    elif nbpts == 1:
        return pts[0] * (t - inst[0])
    elif t <= inst[0]:
        return pts[0] * (t - inst[0])

    assert nbpts % 4 == 0
    result = 0.j

    for i4 in xrange(0, nbpts, 4):
        if t >= inst[i4 + 3]:
            # Full interval
            tmp = pts[i4] + pts[i4 + 1] + pts[i4 + 2] + pts[i4 + 3]
            result = result + .25 * tmp * (inst[i4 + 3] - inst[i4])
        else:
            # Partial interval
            w = (t - inst[i4]) / (inst[i4 + 3] - inst[i4])
            tmp = pts[i4 + 3]
            tmp = w * tmp + pts[i4 + 2] * (4. - 3. * w)
            tmp = w * tmp + pts[i4 + 1] * (6. + w * (-8. + 3. * w))
            tmp = w * tmp + pts[i4] * (4. + w * (-6. + w * (4. - w)))
            return result + tmp * .25 * (t - inst[i4])
    else:
        # Looped over all the intervals without breaking the loop
        return result + pts[nbpts - 1] * (t - inst[nbpts - 1])

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[ZTYPE_t, ndim = 1] zprofile_integrate_vectorized(
        np.ndarray[DTYPE_t, ndim=1] t,
        np.ndarray[DTYPE_t, ndim=1] inst,
        np.ndarray[ZTYPE_t, ndim=1] pts,
        np.ndarray[ZTYPE_t, ndim = 1] result):
    """
    integrate the complex valued slowly varying profile
    from its initial instant to the time instant specified in argument.

    Parameters
    ----------
    t: scalar or array-like
      Upper bound of the time integral.
    inst, pts: arrays
      Instants and coefficients arrays defining the profile.
    """
    cdef np.int_t nt, nbpts, indt, i, i4
    cdef DTYPE_t ti, w
    cdef ZTYPE_t tmp
    cdef np.ndarray[ZTYPE_t, ndim = 1] store

    nt = t.shape[0]
    nbpts = inst.shape[0]
    assert nbpts == pts.shape[0]
    assert nt == result.shape[0]

    if nbpts == 0:
        result[:] = 0.
    elif nbpts ==1:
        for indt in xrange(nt):
            result[indt] = (t[indt] - inst[0]) * pts[0]
    if nbpts <= 1:
        return result

    assert nbpts % 4 == 0
    store = np.empty(nbpts/4, dtype=np.complex128) + np.nan

    for indt in xrange(nt):
        ti = t[indt]
        if ti <= inst[0]:
            result[indt] = pts[0] * (ti - inst[0])
            continue

        result[indt] = 0.j
        i = 0
        for i4 in xrange(0, nbpts, 4):
            if ti >= inst[i4 + 3]:
                # Full interval
                if zisnan(store[i]):
                    tmp = pts[i4] + pts[i4 + 1] + pts[i4 + 2] + pts[i4 + 3]
                    store[i] = .25 * tmp * (inst[i4 + 3] - inst[i4])
                result[indt] = result[indt] + store[i]
            else:
                # Partial interval
                w = (ti - inst[i4]) / (inst[i4 + 3] - inst[i4])
                tmp = pts[i4 + 3]
                tmp = w * tmp + pts[i4 + 2] * (4. - 3. * w)
                tmp = w * tmp + pts[i4 + 1] * (6. + w * (-8. + 3. * w))
                tmp = w * tmp + pts[i4] * (4. + w * (-6. + w * (4. - w)))
                tmp *= .25 * (ti - inst[i4])
                result[indt] = result[indt] + tmp
                break
            i += 1
        else:
            # Looped over all the intervals without breaking the loop
            tmp = pts[nbpts - 1] * (ti - inst[nbpts - 1])
            result[indt] = result[indt] + tmp
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[ZTYPE_t, ndim = 1] zgroup_profile_eval(
        DTYPE_t t,
        np.ndarray[np.int_t, ndim=1] sizes,
        np.ndarray[DTYPE_t, ndim=1] inst,
        np.ndarray[ZTYPE_t, ndim=1] pts,
        np.ndarray[ZTYPE_t, ndim=1] result):
    """
    evaluates the complex value(s) of a group of profiles.

    Parameters
    ----------
    t: array-like
      Time instants where the profile is to be evaluated
    sizes: intergers-array
      The array of the sizes of each profiles of the group.
    inst, pts: arrays
      Instants and coefficients arrays defining the group profile.
    """
    cdef np.int_t n, nprof, npts, indp, idx

    nprof = sizes.shape[0]
    npts = inst.shape[0]
    assert npts == pts.shape[0]
    assert nprof == result.shape[0]

    idx = 0
    for indp in xrange(nprof):
        npts = sizes[indp]
        result[indp] = zprofile_eval(t, inst[idx: idx + npts],
                                        pts[idx: idx + npts])
        idx += npts
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[ZTYPE_t, ndim = 2] zgroup_profile_eval_vectorized(
        np.ndarray[DTYPE_t, ndim=1] t,
        np.ndarray[np.int_t, ndim=1] sizes,
        np.ndarray[DTYPE_t, ndim=1] inst,
        np.ndarray[ZTYPE_t, ndim=1] pts,
        np.ndarray[ZTYPE_t, ndim = 2] result):
    """
    evaluates the complex value(s) of a group of profiles.

    Parameters
    ----------
    t: array-like
      Time instants where the profile is to be evaluated
    sizes: intergers-array
      The array of the sizes of each profiles of the group.
    inst, pts: arrays
      Instants and coefficients arrays defining the group profile.
    """
    cdef np.int_t n, nprof, npts, indp, idx

    nprof = sizes.shape[0]
    npts = inst.shape[0]
    assert npts == pts.shape[0]
    assert nprof == result.shape[0]

    idx = 0
    for indp in xrange(nprof):
        npts = sizes[indp]
        zprofile_eval_vectorized(t, inst[idx: idx + npts], pts[idx: idx + npts],
                      result[indp])
        idx += npts
    return result

