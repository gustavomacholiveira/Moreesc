#! /usr/bin/python
# -*- coding: utf-8 -*-
# Copyright Fabrice Silva, 2012.

cimport numpy as np

cpdef set_beating_factor(np.float64_t value)
cpdef np.float64_t get_beating_factor()

cpdef np.ndarray mech_sys(np.float_t t, np.ndarray X, np.ndarray dX,
        np.float_t p, np.float_t h0, 
        np.ndarray bn, np.ndarray an)

cpdef np.ndarray mech_sys_profiles(np.float_t t, np.ndarray X, np.ndarray dX,
        np.float_t p,
        np.ndarray sbn, np.ndarray ibn, np.ndarray pbn,
        np.ndarray san, np.ndarray ian, np.ndarray pan,
        np.ndarray ih0, np.ndarray ph0)

cdef np.float_t c_get_opening(np.ndarray X, np.ndarray bn)

cdef np.float_t c_get_velocity(np.ndarray X, np.ndarray bn)

cdef np.float_t c_get_opening_profiles(np.ndarray X, np.float_t t,
        np.ndarray sbn, np.ndarray ibn, np.ndarray pbn)

cdef np.float_t c_get_velocity_profiles(np.ndarray X, np.float_t t,
        np.ndarray sbn, np.ndarray ibn, np.ndarray pbn)
