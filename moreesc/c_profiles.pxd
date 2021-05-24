#! /usr/bin/python
# -*- coding: utf-8 -*-
# Copyright Fabrice Silva, 2012.

cimport numpy as np

# Double precision Profiles functions
cpdef np.float64_t dprofile_eval(
        np.float64_t t,
        np.ndarray inst,
        np.ndarray pts) except -1
cpdef np.float64_t dprofile_integrate(
        np.float64_t t,
        np.ndarray inst,
        np.ndarray pts)
cpdef np.ndarray dgroup_profile_eval(
        np.float64_t t,
        np.ndarray sizes,
        np.ndarray inst,
        np.ndarray pts,
        np.ndarray result)

# Double precision Profiles functions (vectorized wrt time)
cpdef np.ndarray   dprofile_eval_vectorized(
        np.ndarray t,
        np.ndarray inst,
        np.ndarray pts,
        np.ndarray result)
cpdef np.ndarray dprofile_integrate_vectorized(
        np.ndarray t,
        np.ndarray inst,
        np.ndarray pts,
        np.ndarray result)
cpdef np.ndarray dgroup_profile_eval_vectorized(
        np.ndarray t,
        np.ndarray sizes,
        np.ndarray inst,
        np.ndarray pts,
        np.ndarray result)

# Double complex Profiles functions
cpdef np.complex128_t zprofile_eval(
        np.float64_t t,
        np.ndarray inst,
        np.ndarray pts)
cpdef np.complex128_t zprofile_integrate(
        np.float64_t t,
        np.ndarray inst,
        np.ndarray pts)
cpdef np.ndarray zgroup_profile_eval(
        np.float64_t t,
        np.ndarray sizes,
        np.ndarray inst,
        np.ndarray pts,
        np.ndarray result)

# Double complex Profiles functions (vectorized wrt time)
cpdef np.ndarray zprofile_eval_vectorized(
        np.ndarray t,
        np.ndarray inst,
        np.ndarray pts,
        np.ndarray result)
cpdef np.ndarray zprofile_integrate_vectorized(
        np.ndarray t,
        np.ndarray inst,
        np.ndarray pts,
        np.ndarray result)
cpdef np.ndarray zgroup_profile_eval_vectorized(
        np.ndarray t,
        np.ndarray sizes,
        np.ndarray inst,
        np.ndarray pts,
        np.ndarray result)

