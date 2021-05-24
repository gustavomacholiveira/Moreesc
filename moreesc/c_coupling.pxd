#! /usr/bin/python
# -*- coding: utf-8 -*-
# Copyright Fabrice Silva, 2012.

cimport numpy as np

cpdef set_twodivbyrho(np.float64_t rho)
cpdef np.float64_t get_twodivbyrho()

cpdef np.ndarray global_sys(np.float_t t, np.ndarray X, np.ndarray dX,
        # Acoustic resonator
        np.ndarray sn, np.ndarray cn,
        # Valve
        np.ndarray bn, np.ndarray an,
        # Other parameters
        np.ndarray params)

cpdef np.ndarray global_sys_profiles(np.float_t t, np.ndarray X, np.ndarray dX,
        # Poles of the acoustic resonator
        np.ndarray ssn, np.ndarray isn, np.ndarray psn,
        # Residues of the acoustic resonator
        np.ndarray scn, np.ndarray icn, np.ndarray pcn,
        # Numerator of the valve
        np.ndarray sbn, np.ndarray ibn, np.ndarray pbn,
        # Denominator of the valve
        np.ndarray san, np.ndarray ian, np.ndarray pan,
        # Other parameters
        np.ndarray son, np.ndarray ion, np.ndarray pon)

cpdef np.ndarray global_jac(np.float64_t t, np.ndarray X, np.ndarray jX,
        # Poles and residues of the acoustic resonator
        np.ndarray sn, np.ndarray cn,
        # Numerator and denominator of the valve
        np.ndarray bn, np.ndarray an)

cpdef np.ndarray global_jac_profiles(np.float64_t t, np.ndarray X, np.ndarray jX,
        # Poles of the acoustic resonator
        np.ndarray ssn, np.ndarray isn, np.ndarray psn,
        # Residues of the acoustic resonator
        np.ndarray scn, np.ndarray icn, np.ndarray pcn,
        # Numerator of the valve
        np.ndarray sbn, np.ndarray ibn, np.ndarray pbn,
        # Denominator of the valve
        np.ndarray san, np.ndarray ian, np.ndarray pan)

cdef np.ndarray c_get_acoustic_state_vector(int nm, np.ndarray X)
cdef np.ndarray c_set_acoustic_state_vector(int nm, np.ndarray X, np.ndarray Xa)
cdef np.ndarray c_get_mechanic_state_vector(int nm, np.ndarray X)
cdef np.ndarray c_set_mechanic_state_vector(int nm, np.ndarray X, np.ndarray Xm)

cdef np.float64_t c_nl_coupling(np.float64_t dp, np.float64_t h, np.float64_t v)
