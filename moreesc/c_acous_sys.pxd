#! /usr/bin/python
# -*- coding: utf-8 -*-
# Copyright Fabrice Silva, 2012.

cimport numpy as np

cpdef np.ndarray acous_sys(np.float_t, np.ndarray, np.ndarray,
        np.float_t, np.ndarray, np.ndarray)

cpdef np.ndarray acous_sys_profiles(np.float_t, np.ndarray, np.ndarray, np.float_t,
        np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, np.ndarray)

cdef np.float_t c_get_pressure(np.ndarray X)


cpdef np.ndarray modal_impedance(np.ndarray, np.ndarray, np.ndarray, np.ndarray)
