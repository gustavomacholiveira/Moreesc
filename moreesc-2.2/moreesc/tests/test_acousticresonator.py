#!/usr/bin/python
# -*- coding: utf-8 -*-

#Copyright (C) 2010 Fabricio Silva

import numpy as np
import numpy.random as rd
import numpy.testing as tt
import matplotlib.pyplot as plt
import sys, warnings
sys.path.append('../../')
from moreesc import AcousticResonator, Profiles
from nose.tools import nottest
from . import test_profile as ttp

# Change to True it if AcousticResonator.py is modified
__test__ = False


def setup():
    warnings.simplefilter("ignore")


def teardown():
    warnings.resetwarnings()
    ttp.del_tmpfile()


def generate_tvi(N=1):
    sn = Profiles.GroupProfiles(ttp.generate_complex_profiles(N))
    Cn = Profiles.GroupProfiles(ttp.generate_complex_profiles(N))
    Zc = ttp.get_random_Linear()[0]
    for b in (True, False):
        yield AcousticResonator.TimeVariantImpedance(
            sn=sn, Cn=Cn, reduced=b, Zc=Zc)


def generate_tii(N=None):
    if N is None:
        N = rd.randint(1, 20)
    tmp = rd.rand(4, N)
    poles    = -(np.abs(tmp[0]) + 1.j * tmp[1]) * 1000
    residues = (tmp[2] + 1.j * tmp[3]) * 1000
    Zc = rd.rand(1) * np.exp(rd.rand(1))
    for b in (True, False):
        yield AcousticResonator.TimeInvariantImpedance(
            sn=poles, Cn=residues, reduced=b, Zc=Zc)


class TypesCase:
    #__test__ = False
    def test_tvi(self):
        for tmp in generate_tvi(3):
            self.do(tmp)

    def test_tii(self):
        for tmp in generate_tii():
            self.do(tmp)

    def test_cylinder(self):
        for radiate in (False, True):
            tmp = AcousticResonator.Cylinder(L=1., r=7e-3,
                radiates=radiate, nbmodes=rd.randint(2, 20))
            self.do(tmp)

    @nottest
    def test_measuredimpedance(self):
        tmp = AcousticResonator.MeasuredImpedance(
            filename='./data/Trombone.dat', storage='txt_realimag')
        tmp.estimate_modal_expansion()
        self.do(tmp)


class TestSaveLoad(TypesCase):
    def do(self, v):
        f = ttp.tmpfile()
        v.save(f)
        c = AcousticResonator.load_impedance(f)
        representation = "%s" % v
        del representation
        tt.assert_equal(c, v)
        tt.assert_equal(c is None, False)
        tt.assert_equal(c == [], False)


class TestMultipleTrace(TypesCase):
    def do(self, v):
        t = np.linspace(0, 10, 10)
        for scale in ('lin', 'log'):
            v.trace(t=t, linlog=scale)
        plt.close('all')


@tt.dec.slow
class TestImpulseResponse(TypesCase):
    #__test__ = False
    def do(self, z):
        vect = np.linspace(0, 10e0, 2 ** 14)[:2]
        dt = vect[1] - vect[0]
        Z = AcousticResonator.TimeVariantImpedance(
            sn=[z.poles[n] for n in range(1)],
            Cn=[z.residues[n] for n in range(1)],
            reduced=False, Zc=8.e6)
        # Analytical solution
        ana = np.zeros((Z.nbmodes, vect.size), dtype=np.complex128)
        # integrals of poles trajectories
        isn = np.zeros((Z.nbmodes, vect.size), dtype=np.complex128)
        for indn in range(Z.nbmodes):
            isn[indn, :] = Z.poles[indn].integrate(vect)
            ana[:, :] += Z.residues[indn](0.) * np.exp(isn)

        # Using compiled code
        from scipy.integrate import ode
        dX = np.zeros(Z.nbmodes, dtype=np.complex128) # Pre-allocated dX
        args = (dX, 0.,
            Z.poles.sizes_array, Z.poles.instants_array, Z.poles.coefs_array,
            Z.residues.sizes_array, Z.residues.instants_array, Z.residues.coefs_array)

        mor = np.zeros((Z.nbmodes, vect.size), dtype=np.complex128)
        mor[:, 0] = ana[:, 0]
        I = ode(AcousticResonator.acous_sys_profiles).set_integrator('zvode')
        I.set_initial_value(mor[:, 0])
        I.set_f_params(*args)
        I.t = 0.
        indt = 0
        while I.successful() and I.t < vect[-1]:
            I.integrate(I.t + dt)
            indt += 1
            mor[:, indt] = np.asscalar(I.y)
        ttp.assert_allclose_appop(mor, ana, rtol=1e-3, atol=1e-4*np.max(np.abs(ana)))


def test_Cylinder():
    tmp = rd.randn(2)
    L = np.abs(.5 + 0.1 * tmp[0])
    r = np.abs(7e-3 + 1e-3 * tmp[1])
    for radiate in (True, False):
        Z = AcousticResonator.Cylinder(L=L, r=r,
                radiates=radiate, nbmodes=1000)
        freq = np.linspace(30., 2000., 2048)
        s = 2.j * np.pi * freq
        ana = Z.eval_analytically(s)
        mor = Z(s)
        ttp.assert_allclose_appop(mor, ana, rtol=1e-3,
            atol=1e-3 * np.max(np.abs(ana)))
    plt.show()

if __name__ == '__main__':
    tt.run_module_suite()
    ttp.del_tmpfile()
