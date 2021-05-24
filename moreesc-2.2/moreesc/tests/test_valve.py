    #!/usr/bin/python
# -*- coding: utf-8 -*-

#Copyright (C) 2010 Fabricio Silva

import numpy as np
import numpy.random as rd
import numpy.lib.polynomial as poly
import numpy.testing as tt
import matplotlib.pyplot as plt
import sys, warnings#, tempfile
#sys.path.append('../../')
from moreesc import Valve, Profiles
from moreesc.c_mech_sys import mech_sys_profiles
from . import test_profile as ttp

# Change to True it if Valve.py is modified
#__test__ = False

def setup():
    warnings.simplefilter("ignore")    
def teardown():
    warnings.resetwarnings()
    ttp.del_tmpfile()

def generate_valid_denominator(N):
    l = ttp.generate_real_profiles(N)
    l[0] = 1.
    return l

def generate_tf(M=None,N=None):
    if M is None:
        M = rd.randint(1,10)
    if N is None:
        N = rd.randint(1,10)
    M, N = np.sort([M,N])
    num = ttp.generate_real_profiles(M)
    den = generate_valid_denominator(N)
    return Valve.TransferFunction(num, den)

class TypesTestCase:
#    __test__ = False
    def test_TransferFunction(self):
        tmp = generate_tf()
        self.do(tmp)

    def test_OneDOFOscillator(self):
        tmp = ttp.generate_real_profiles(3)
        tmp = Valve.OneDOFOscillator(*tmp)
        self.do(tmp)

    def test_ReedDynamics(self):
        tmp = ttp.generate_real_profiles(4)
        tmp[2] = Profiles.toScalar(tmp[2])
        tmp[3] = Profiles.toScalar(tmp[3])
        tmp = Valve.ReedDynamics(*tmp)
        self.do(tmp)

    def test_LipDynamics(self):
        tmp = ttp.generate_real_profiles(4)
        tmp[2] = Profiles.toScalar(tmp[2])
        tmp[3] = Profiles.toScalar(tmp[3])
        tmp = Valve.LipDynamics(*tmp)
        self.do(tmp)

class TestSaveLoad(TypesTestCase):
    def do(self, v):
        f = ttp.tmpfile()
        v.save(f)
        represent = '%s' % v
        del represent
        c = Valve.load_transferfunction(f)
        tt.assert_equal(c, v)
        tt.assert_equal(c==[], False)

class TestSimpleEval(TypesTestCase):
    def do(self, v):
        f = np.linspace(0,5000,50)
        s = 2.j*np.pi*f
        ref = poly.polyval(v.num(0), s)/poly.polyval(v.den(0), s)
        tt.assert_array_almost_equal(v(f), ref)

class TestMultipleEval(TypesTestCase):
    def do(self, v):
        s = 2.j*np.pi*np.linspace(0,5000,50)
        t = np.linspace(0,10,10)
        val = v(s, t)
        nums = v.num(t)
        dens = v.den(t)
        for indt in range(len(t)):
            ref = poly.polyval(nums[:, indt], s)/poly.polyval(dens[:, indt], s)
            ttp.assert_allclose_appop(val[indt], ref, rtol=1e-4)

class TestMultipleTrace(TypesTestCase):
    def do(self, v):
        t = np.linspace(0,10,10)
        for f in (None, np.logspace(10,2000,1024)):
            for scale in ('lin', 'log'):
                v.trace(t=t, f=f, linlog=scale)
        plt.close('all')

class OneDOF_TypesCase:
#    __test__ = False
    wr = Profiles.Linear((0., 1.), (2.*np.pi * 1000., 2.*np.pi * 2000.))
    qr = .01#Profiles.Linear((0., 1.), (0.01, .1))
    kr = 1.e6
    def test_OneDOFOscillator(self):
        v = Valve.OneDOFOscillator(self.wr, self.qr, 1/self.kr)
        return self.do(v)
    def test_ReedDynamics(self):
        v = Valve.ReedDynamics(self.wr, self.qr, self.kr)
        return self.do(v)
    def test_LipDynamics(self):
        v = Valve.LipDynamics(self.wr, self.qr, self.kr)
        return self.do(v)

class TestStepResponseLTI(OneDOF_TypesCase):
#    __test__ = False
    # Only testing LTI
    def do(self, v):
        if not(isinstance(v, Valve.OneDOFOscillator)): return
        vect = np.linspace(0, 1e0, 2**14)
        wr, qr, kr = v.wr(0.), v.qr(0.), 1./v.H0(0.)

        # Using scipy.signal module
        num, den = (wr**2/kr,), (1, qr*wr, wr**2)
        from scipy.signal import impulse2
        t,y1 = impulse2((num, den), T=vect)
        
        h0 = Profiles.Profile()
        # Using moreesc fortran code
        from scipy.integrate import ode
        dX = np.zeros(2, dtype=np.float64) # Pre-allocated dX
        v2 = Valve.OneDOFOscillator(wr=wr, qr=qr, H0=v.H0(0.))
        
        args = (dX, 0., 
            np.asarray(v2.num.sizes_array),
            np.asarray(v2.num.instants_array),
            np.asarray(v2.num.coefs_array),
            np.asarray(v2.den.sizes_array),
            np.asarray(v2.den.instants_array),
            np.asarray(v2.den.coefs_array),
            np.asarray(h0.instants),
            np.asarray(h0.coefs))

        mor = np.zeros((2, vect.size), dtype=np.float64)
        # Initialisation based on previous result and continuity relation
        mor[0, 0] = y1[0]
        mor[1, 0] = (num[-1] - den[1] * y1[0])
        I = ode(mech_sys_profiles).set_integrator('vode')
        I.set_initial_value(mor[:, 0])
        I.set_f_params(*args)
        I.t = 0.
        indt = 1
        while I.successful() and indt < len(vect):
            I.integrate(vect[indt])
            mor[:, indt] = I.y
            indt += 1

        # Analytic solution
        Om = np.sqrt(1-qr**2/4)*wr
        y0 = np.exp(-.5 * qr * wr * vect) * wr**2 / (kr * Om) * np.sin(Om*vect)

        ttp.assert_allclose_appop(mor[0, :], y1, rtol=1e-3, atol=1e-4*np.max(np.abs(y1)))
        ttp.assert_allclose_appop(y0, y1, rtol=1e-3, atol=1e-4*np.max(np.abs(y1)))


# This test is known to fail: using the canonical observable form, 
# the mechanical system to study is sligthly modified for the coefficients
# of the ODE are time variable. For example
#       h"(t) + a1 * h'(t) + a2 * h(t) = delta(t)
# becomes
#       h"(t) + d( a1 * h(t) )/dt + a2 * h(t) = delta(t)
# It should not matter for very slowing varying coefficient, but no robust
# test can be well defined.
class TestStepResponse(OneDOF_TypesCase):
#    __test__ = False
    def do(self, v):
        if not(isinstance(v, Valve.OneDOFOscillator)): return
        vect = np.linspace(0, .1e0, 2**14)
        wr = v.wr.__copy__()
        qr = v.qr.__copy__()
        # Analytic solution
        reals0 = -.5 * qr * wr
        imags0 = wr * (1. - .25 * qr ** 2) ** .5
        s00 = reals0(0) + 1.j * imags0(0)
        reals0_int = reals0.integrate(vect)
        imags0_int = imags0.integrate(vect)
        b00 = v.num[-1](0.)
        env = np.exp(reals0_int) / s00.imag * b00
        y0 = env * np.sin(imags0_int)

        # Using moreesc fortran code
        from scipy.integrate import ode
        dX = np.zeros(2, dtype=np.float64) # Pre-allocated dX
        h0 = Profiles.Profile()
        args = (dX, 0.,
            v.num.sizes_array, v.num.instants_array, v.num.coefs_array,
            v.den.sizes_array, v.den.instants_array, v.den.coefs_array,
            h0.instants, h0.coefs)

        mor = np.zeros((2, vect.size), dtype=np.float64)
        # Initialisation based on previous result and continuity relation
        mor[0, 0] = 0.
        mor[1, 0] = b00
        I = ode(mech_sys_profiles).set_integrator('vode')
        I.set_initial_value(mor[:, 0])
        I.set_f_params(*args)
        I.t = 0.
        indt = 1
        while I.successful() and indt < len(vect):
            I.integrate(vect[indt])
            mor[:, indt] = I.y
            indt += 1
        ttp.assert_allclose_appop(mor[0, :], y0, rtol=4e-2, atol=1e-3*np.max(np.abs(y0)))


def test_InvalidTransferFunction():
    # Test construction of valid TransferFunction.
    tmp = rd.randint(1,10,size=2)
    num = ttp.generate_real_profiles(min(tmp))
    den = ttp.generate_real_profiles(max(tmp))
    if den[0]!=1.:
        tt.assert_raises(ValueError, Valve.TransferFunction, num,den)

if __name__=='__main__':
    tt.run_module_suite()
    ttp.del_tmpfile()
