#!/usr/bin/python
# -*- coding: utf-8 -*-

#Copyright (C) 2010 Fabricio Silva

import numpy as np
import numpy.random as rd
import numpy.testing as tt
from numpy.testing.utils import assert_array_compare
import sys
sys.path.append('../../')
import tempfile
import warnings
from moreesc import Profiles
#reload(Profiles)  # for coverage to take into account definitions

# Change to True it if Profiles.py is modified
__test__ = False


def setup():
    warnings.simplefilter("ignore")


def teardown():
    warnings.resetwarnings()
    del_tmpfile()

rtol = 1e-6
def sym_allclose(a, b, rtol=1.e-8, atol=1.e-8):
    """
    Comparator with symmetric relative tolerance.
    This criterion implies that
    ( np.abs(x-y) <= atol + rtol * max(np.abs(x),np.abs(y)) )
    with no maximum searches.
    """
    x = np.array(a, copy=False, ndmin=1)
    y = np.array(b, copy=False, ndmin=1)
    xfin, yfin = np.isfinite(x), np.isfinite(y)
    if not(np.all(xfin) or np.all(yfin)):
        # Check that x and y have inf or nan's only in the same positions
        if not np.all(xfin == yfin):
            return False
        # Check that sign of inf's in x and y is the same
        if not np.all(x[np.isinf(x)] == y[np.isinf(y)]):
            return False
        x, y = x[xfin], y[yfin]
    if atol < 0.:
        # Do not catch differences near zero crossing
        atol = 1.e-4 * np.max(np.abs(y))
    tmp = np.abs(x - y) <= atol + rtol * .5 * (np.abs(x) + np.abs(y))
    return tmp


def assert_allclose_appop(actual, desired, rtol=5.e-3, atol=-1., \
    err_msg='', verbose=True):
    if atol == -1 and np.sign(desired.max()) != np.sign(desired.min()):
        atol = 1e-3 * np.abs(desired).max()
    assert_array_compare(
        lambda x, y: sym_allclose(x, y, rtol=rtol, atol=atol),
        np.asanyarray(actual), np.asanyarray(desired),
        err_msg=str(err_msg), verbose=verbose,
        header='Not equal to tolerance rtol=%g, atol=%g' % (rtol, atol))


# Generators
def get_random_Linear(Npt=5):
    tmp = rd.randn(2, Npt)
    tmp[0, :] = np.sort(tmp[0, :])
    return Profiles.Linear(instants=tmp[0, :], values=tmp[1, :]), tmp


def get_random_Spline(Nknots=5, k=3, maxi=10.):
    x, y = np.sort(rd.rand(Nknots) * maxi), rd.randn(Nknots)
    tck = Profiles.ii.splrep(x, y, s=0)
    t, c, k = tck
    c = [c[:-2 * k + 2], ]
    return Profiles.Spline((t, c, k)), (t, c, k)


def get_smooth_Signal(N=512, tmax=10., Nf=5):
    t = np.linspace(0, tmax, N)
    f = rd.rand(2, Nf) + 1
    Cn = rd.randn(2, Nf)
    sig = np.zeros(N)
    for ind in range(Nf):
        sig += np.cos(f[0, ind] * t * ind) * Cn[0, ind]
        sig += np.sin(f[1, ind] * t * ind) * Cn[1, ind]
    return Profiles.Signal(t, sig, smoothness=1.e-4), t, sig


def get_C1_step(dtype=None):
    t = 10. * np.sort(rd.rand(4))
    if dtype is complex:
        tmp2 = rd.randn(4)
        values = 10. * tmp2[:2] + 5.j * tmp2[2:]
        del tmp2
    else:
        values = 10. * rd.randn(2)
    return Profiles.C1_Step(t, *values)


def get_C2_step(dtype=None):
    t = 10. * np.sort(rd.rand(4))
    if dtype is complex:
        tmp2 = rd.randn(4)
        values = 10. * tmp2[:2] + 5.j * tmp2[2:]
        del tmp2
    else:
        values = 10. * rd.randn(2)
    return Profiles.C2_Step(t, *values)


def get_cyclic_complex_signal(N=32):
    t = np.linspace(0, rd.rand(1), N)
    ampl = 2. + rd.rand(1)
    sig = ampl * np.exp(2.j * np.pi * t)
    return Profiles.Signal(t, sig, smoothness=1.e-4), ampl


def get_cyclic_complex_spline(N=32):
    a, ampl = get_cyclic_complex_signal()
    return Profiles.Spline((a.t, a.c, a.k)), ampl


def list_profiles(with_empty=True):
    tmp = [\
        rd.randn(1)[0],
        Profiles.Profile(),
        Profiles.Constant(rd.randn(1)),
        Profiles.Constant(1.j),
        get_random_Linear()[0],
        get_C1_step(float),
        get_C1_step(complex),
        get_C2_step(float),
        get_C2_step(complex),
        get_random_Spline()[0],
        get_smooth_Signal()[0],
        get_cyclic_complex_spline()[0],
        get_cyclic_complex_signal()[0],
        ]
    if not(with_empty):
        tmp.pop(1)
    return tmp

def generate_real_profiles(N=1):
    l = []
    generators = [
        lambda: rd.randn(1)[0],
        lambda: Profiles.Constant(rd.randn(1)),
        lambda: get_random_Linear()[0],
        lambda: get_C1_step(),
        lambda: get_C2_step(),
        lambda: get_random_Spline()[0],
        lambda: get_smooth_Signal()[0]        
    ]
    tmp = rd.randint(0, 5, N)
    for ind in range(N):
        l.append(generators[tmp[ind]]())
    return l

def generate_complex_profiles(N=1):
    l = []
    generators = [
        lambda: rd.randn(1)[0],
        lambda: Profiles.Constant(rd.randn(1)),
        lambda: get_random_Linear()[0],
        lambda: get_C1_step(complex),
        lambda: get_random_Spline()[0],
        lambda: get_smooth_Signal()[0]        
    ]
    tmp = rd.randint(0, 5, 2 * N)
    for ind in range(0, 2 * N, 2):
        l.append(generators[tmp[ind]]() + 1.j * generators[tmp[ind+1]]())
    return l

# Temporary files
def tmpfile():
    idf, f = tempfile.mkstemp(prefix='moreesctest', suffix='.npz')
    return f


def del_tmpfile():
    import os
    l = [os.path.join('/tmp', s) for s in os.listdir('/tmp/')
         if s.startswith('moreesctest') and s.endswith('.npz')]
    for f in l:
        os.remove(f)


def test_empty(N=10):
    """
    test_empty
    Calling Profile base class should give zero as it has no coefficient.
    """
    a = Profiles.Profile()
    val = a(rd.randn(N))
    tt.assert_allclose(val, np.zeros(N, dtype=float), rtol=rtol)


def test_Constant(N=10, val=100):
    """
    test_Constant
    Calling Constant class should give identical values.
    """
    a = Profiles.Constant(val)
    values = a(rd.randn(N))
    tt.assert_allclose(values, np.ones(N, 'float') * val, rtol=rtol)


def test_Linear(N=2048):
    """
    test_Linear
    Calling Linear class should give identical values.
    """
    a, tmp = get_random_Linear(Npt=3)
    time, vals = tmp
    t2 = np.sort(rd.randn(N))
    ref = np.interp(t2, time, vals, left=vals[0], right=vals[-1])
    val = a(t2)
    tt.assert_allclose(val, ref, rtol=rtol)


def test_Spline(N=512, maxi=10, plot=False):
    """
    test_Spline
    Calling Spline class should return same values as low-level splev call.
    """
    a, tck = get_random_Spline(maxi=maxi)
    t2 = np.sort(maxi * (.5 + rd.randn(N)))
    values = a(t2)
    values2 = a.call_as_tck(t2)
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(t2, values, 'r-', label='bezier')
        plt.plot(t2, values2, 'bx-', label='spline')
        plt.axvline(a.t.min(), c='k')
        plt.axvline(a.t.max(), c='k')
        plt.plot(a.t[3:-3], a.c[0], 'ko')
        plt.plot(a.coefs[0], a.coefs[1], 'ro')
        plt.legend()
        plt.figure()
        plt.plot(t2, values - values2)
        plt.show()
    tt.assert_allclose(values, values2, rtol=rtol)
    return a


def test_Signal(N=2048, maxi=10, plot=False):
    """
    test_Signal
    Calling Signal class should return values close to original values
    """
    a, t, sig = get_smooth_Signal(N=N, tmax=maxi)
    val1 = a(t).__copy__()
    a.fit_spline(s=a.sref * 1e-6)
    val2 = a(t)
    if plot:
        import matplotlib.pyplot as plt
        plt.subplot(211)
        plt.plot(t, sig, 'b-', label='signal')
        plt.plot(t, val1, 'r-', label='appref')
        plt.plot(t, val2, 'g-', label='app2')
        plt.legend()
        plt.subplot(212)
        plt.plot(t, val1 - sig, 'r')
        plt.plot(t, val2 - sig, 'g')
        plt.show()
    assert_allclose_appop(val2, sig, rtol=.02)


def test_realworldsignal(plot=False):
    filename = '/tmp/FilteredMeasurement.npz'
    try:
        d = np.load(filename)
        time, raw, filt = d['time'], d['raw'], d['filt']
    except:
        import scipy.io.wavfile as wave
        import os
        path = os.path.abspath(os.path.dirname(Profiles.__file__))
        path = os.path.join(path, '../data/Profil_Pbouche.wav')
        tmp = wave.read(path)
        Fe, raw = float(tmp[0]), np.array(tmp[1], dtype=float)
        time = np.arange(len(raw)) / Fe

        # Spectre
        Np = int(2 ** np.ceil(np.log2(len(raw))))
        data_fft = np.fft.rfft(raw, n=Np)
        freq = np.linspace(0, int(Fe / 2), len(data_fft) + 1)[:-1]
        idx = freq > 50
        f0 = freq[idx][np.argmax(np.abs(data_fft[idx]))]

        # Filtrage harmonique pour supprimer les oscillations
        # (peigne de zéros)
        import scipy.signal as sig
        Dp1 = int(2 * np.pi * Fe / f0)
        tmp = np.concatenate([raw, raw[-1: -2 * Dp1:-1]])
        tmp = sig.lfilter(np.ones(Dp1), Dp1, tmp)[::-1]
        filt = sig.lfilter(np.ones(Dp1), Dp1, tmp)[::-1][:len(raw)]
        np.savez(filename, time=time, raw=raw, filt=filt)

    p = Profiles.Signal(time, filt)
    val = p(time)
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(time, val, label='Spline')
        plt.plot(time, filt, label='Signal')
        plt.legend()
        plt.show()
    # 5% of max value of signal
    tt.assert_array_less(np.abs(val - filt), .05 * np.abs(filt).max())
    return p


def test_complex_spline(N=2048):
    a, ampl = get_cyclic_complex_spline()
    t = np.linspace(a.t[0], a.t[-1], 257)
    ref = ampl * np.exp(2.j * np.pi * t)
    assert_allclose_appop(a(t), ref)


def test_groupprofiles(plot=False):
    tmp = list_profiles()
    gp = Profiles.GroupProfiles(tmp)
    represent = '%s' % gp
    del represent
    t = np.sort(rd.rand(128))
    vals = gp(t)
    for ind, val in enumerate(vals):
        tmp = gp[ind](t)
        tt.assert_allclose(val, tmp)


# Base Classes for testing the various Profiles types (even scalars)
class UnaryTypes(object):
#    __test__ = False
    def test(self):
        for f in [w.__copy__() for w in list_profiles()[1:]
            if isinstance(w, Profiles.Profile)]:
            fun = lambda x: self.do(f)
            fun.description = '%s.test_%s%s' % \
                (self.__class__.__name__, type(f).__name__, f.dtype.kind)
            yield fun, f

    def do(self, obj):
        pass


class BinaryTypes(object):
#    __test__ = False
    def test(self):
        for f, g in [(w.__copy__(), z.__copy__())
                    for w in list_profiles() for z in list_profiles()]:
            if isinstance(f, Profiles.Profile):
                if isinstance(g, Profiles.Profile):
                    tmin = np.min(np.r_[f.instants, g.instants, 0.])
                    tmax = np.max(np.r_[f.instants, g.instants, 1.])
                    t = np.sort((tmin + tmax) / 2.
                              + (tmax - tmin) / 2. * rd.randn(100))
                    valf, valg = f(t), g(t)
                else:
                    tmin = np.min(np.r_[f.instants, 0.])
                    tmax = np.max(np.r_[f.instants, 1.])
                    t = np.sort((tmin + tmax) / 2.
                              + (tmax - tmin) / 2. * rd.randn(100))
                    valf, valg = f(t), g
            else:
                if isinstance(g, Profiles.Profile):
                    tmin = np.min(np.r_[g.instants, 0.])
                    tmax = np.max(np.r_[g.instants, 1.])
                    t = np.sort((tmin + tmax) / 2.
                              + (tmax - tmin) / 2. * rd.randn(100))
                    valf, valg = f, g(t)
                else:
                    continue
            fun = lambda: self.do(t, f, g, valf, valg)
            fun.description = '%s.test_%s%s_%s%s' % \
                (self.__class__.__name__,
                 type(f).__name__, f.dtype.kind,
                 type(g).__name__, g.dtype.kind)
            yield fun

    def do(self, *args):
        pass


# Derived Test classes
class TestSaveLoad(UnaryTypes):
    def do(self, x):
        if np.isscalar(x):
            return
        f = tmpfile()
        x.save(f)
        represent = '%s' % x
        del represent
        y = Profiles.load_profile(f)
        tt.assert_equal(x, y)


class TestCopy(UnaryTypes):
    def do(self, x):
        y = x.__copy__()
        tt.assert_equal(x, y)


class TestOpposite(UnaryTypes):
    def do(self, x):
        t = rd.randn(100) + rd.randn(1) * 10.
        if Profiles.isEmpty(x):
            tt.assert_raises(ValueError, lambda: -x)
        else:
            tt.assert_allclose((-x)(t), -(x(t)))


class TestSquare(UnaryTypes):
    def do(self, x):
        t = rd.randn(100) + rd.randn(1) * 10.
        valx = x(t)
        if np.isscalar(x) or Profiles.isEmpty(x):
            return
        assert_allclose_appop((x ** 2)(t), valx ** 2)


class TestAdd(BinaryTypes):
    def do(self, t, x, y, valx, valy):
        if Profiles.isEmpty(x) or Profiles.isEmpty(y):
            tt.assert_raises(ValueError, lambda: x + y)
        else:
            tt.assert_allclose((x + y)(t), valx + valy)


class TestiAdd(BinaryTypes):
    def do(self, t, x, y, valx, valy):
        if not(isinstance(x, Profiles.Profile)):
            return
        if Profiles.isEmpty(x) or Profiles.isEmpty(y):
            def fun(a, b):
                a += b
            tt.assert_raises(ValueError, fun, x, y)
        else:
            x += y
            tt.assert_allclose(x(t), valx + valy)


class TestSub(BinaryTypes):
    def do(self, t, x, y, valx, valy):
        if Profiles.isEmpty(x) or Profiles.isEmpty(y):
            tt.assert_raises(ValueError, lambda: x - y)
        else:
            tt.assert_allclose((x - y)(t), valx - valy)


class TestiSub(BinaryTypes):
    def do(self, t, x, y, valx, valy):
        if not(isinstance(x, Profiles.Profile)):
            return
        if Profiles.isEmpty(x) or Profiles.isEmpty(y):
            def fun(a, b):
                a -= b
            tt.assert_raises(ValueError, fun, x, y)
        else:
            x -= y
            tt.assert_allclose(x(t), valx - valy)


class TestMul(BinaryTypes):
    def do(self, t, x, y, valx, valy):
        if Profiles.isEmpty(x) or Profiles.isEmpty(y):
            tt.assert_raises(ValueError, lambda: x * y)
        else:
            assert_allclose_appop((x * y)(t), valx * valy)


class TestiMul(BinaryTypes):
    def do(self, t, x, y, valx, valy):
        if not(isinstance(x, Profiles.Profile)):
            return
        if Profiles.isEmpty(x) or Profiles.isEmpty(y):
            def fun(a, b):
                a *= b
            tt.assert_raises(ValueError, fun, x, y)
        else:
            x *= y
            assert_allclose_appop(x(t), valx * valy)


class TestDiv(BinaryTypes):
    def do(self, t, x, y, valx, valy):
        if Profiles.isEmpty(x) or Profiles.isEmpty(y):
            tt.assert_raises(ValueError, lambda: x / y)
            return
        if Profiles.isInversible(y, valy):
            assert_allclose_appop((x / y)(t), valx / valy)
        else:
            tmp = lambda: x / y
            tt.assert_raises(ValueError, tmp)


class TestiDiv(BinaryTypes):
    def do(self, t, x, y, valx, valy):
        if not(isinstance(x, Profiles.Profile)):
            return
        if Profiles.isEmpty(x) or Profiles.isEmpty(y) \
            or not(Profiles.isInversible(y, valy)):
            def fun(a, b):
                a /= b
            tt.assert_raises(ValueError, fun, x, y)
        else:
            x /= y
            assert_allclose_appop(x(t), valx / valy)


class TestIntegrate(UnaryTypes):
#    __test__ = False
    def do(self, x):
        if not(isinstance(x, Profiles.Profile)) or Profiles.isEmpty(x):
            return
        inst = x.instants
        coefs = x.coefs
        t = rd.randn(1) * (inst[-1] - inst[0] + .1) + .5 * (inst[-1] + inst[0]) + 50
        val = x.integrate(t)
        if isinstance(x, Profiles.Constant):
            ref = x.value * t
        elif isinstance(x, Profiles.Linear):
            if t < inst[0]:
                ref = coefs[0] * (t - inst[0])
            else:
                ref = 0.
                for i4 in range(0, len(inst), 4):
                    if t > inst[i4 + 3]:
                        # Full interval integration
                        ref += .5 * (coefs[i4 + 3] + coefs[i4]) \
                                  * (inst[i4 + 3] - inst[i4])
                    else:
                        # Partial interval integration
                        tmp = .5 * (t - inst[i4]) / (inst[i4 + 3] - inst[i4])
                        ref += (t - inst[i4]) * \
                            (coefs[i4] + tmp * (coefs[i4+3] - coefs[i4]))
                        break
                else: # i.e. if t> inst[-1]:
                    ref += coefs[-1] * (t - inst[-1])

        elif isinstance(x, Profiles.C1_Step):
            vout, vstep = coefs[0], coefs[3]
            dv = vstep - vout
            vmean = .5 * (vstep + vout)
            t0, t3, t7, t11 = inst[0], inst[3], inst[7], inst[11]
            if t < t0:
                ref = vout * (t - t0)
            elif t0 < t < t3:
                u = (t - t0) / (t3 - t0)
                ref = (t - t0) * (vout + dv * (1. - .5 * u) * u ** 2)
            else:
                ref = (t3 - t0) * vmean

            if t3 < t < t7:
                ref +=  (t - t3) * vstep
            elif t7 < t:
                ref += (t7 - t3) * vstep

            if t7 < t < t11:
                u = (t - t7) / (t11 - t7)
                ref +=  (t - t7) * (vstep - dv * (1. - .5 * u) * u ** 2)
            elif t11 < t:
                ref += (t11 - t7) * vmean + (t - t11) * vout

        elif isinstance(x, Profiles.C2_Step):
            warnings.warn("Skipping TestIntegrate for C2_Step Profile.")
            vout, vstep = coefs[0], coefs[11]
            dv = vstep - vout
            vmean = .5 * (vstep + vout)
            t0, ta, td, t1 = inst[0], inst[4], inst[8], inst[11]
            # TODO : check that t0, ta , td, t1 are equally spaced
            dt0 = (t1 - t0)

            if t < t0:
                ref = vout * (t - t0)
            elif t0 < t < ta:
                u = (t - t0) / (ta - t0)
                ref = (t - t0) * vout + dv / 72. * dt0 * u ** 4
            else:
                ref = (ta - t0) * vout + dv / 72. * dt0

            if ta < t < td:
                u = (t - ta) / (td - ta)
                ref +=  (t - ta) * vout
                ref += dv * dt0 / 36. * (2*u + 3*u**2 + 2*u**3 - u**4)
            elif td < t:
                ref += (td - ta) * (vout + dv / 2.)

            if td < t < t1:
                u = (t - td) / (t1 - td)
                ref += (t - td) * vout
                ref += dv * dt0 / 72. * (20*u + 6*u**2 - 4*u**3 + u**4)
            elif t1 < t:
                ref += (t1 - td) * vout + 23 * dv / 72. * dt0
                
            tt.assert_almost_equal(t1, inst[12])
            t1, t2 = inst[12], inst[15]
            if t1 < t < t2:
                ref += vstep * (t - t1)
            elif t2 < t:
                ref += vstep * (t2 - t1)
            
            tt.assert_almost_equal(t2, inst[16])
            t2, ta, td, t3 = inst[16], inst[20], inst[24], inst[27]
            # TODO : check that t2, ta , td, t3 are equally spaced
            dt2 = (t3 - t2)
            if t2 < t < ta:
                u = (t - t2) / (ta - t2)
                ref += (t - t2) * vstep - dv / 72. * dt2 * u ** 4
            elif ta < t:
                ref += (ta - t2) * vstep - dv / 72. * dt2

            if ta < t < td:
                u = (t - ta) / (td - ta)
                ref +=  (t - ta) * vstep
                ref -= dv * dt2 / 36. * (2*u + 3*u**2 + 2*u**3 - u**4)
            elif td < t:
                ref += (td - ta) * (vstep - dv / 2.)

            if td < t < t3:
                u = (t - td) / (t3 - td)
                ref += (t - td) * vstep
                ref -= dv * dt2 / 72. * (20*u + 6*u**2 - 4*u**3 + u**4)
            elif t3 < t:
                ref += (t3 - td) * vstep - 23 * dv / 72. * dt2
                ref += vout * (t - t3)

        elif isinstance(x, Profiles.Spline):
            ref = x.integrate_as_tck(inst[0], t)
        tt.assert_almost_equal(val, ref)


class TestGroupProfiles(UnaryTypes):
#    __test__ = False
    def do(self, x):
        if Profiles.isEmpty(x):
            return
        # GroupProfiles with all Profile subclass
        gp = Profiles.GroupProfiles(list_profiles(with_empty=False))
        tmax = np.max(np.r_[gp.instants_array.max(), 1.])
        tmin = np.min(np.r_[gp.instants_array.min(), 0.])
        t = np.sort((tmax + tmin) / 2. + (tmax - tmin) / 2. * rd.randn(128))
        valx = x(t)
        valgp = gp(t)

        gpt = gp.__copy__()
        gpt += x
        tt.assert_allclose(gpt(t), valgp + valx)

        gpt = gp.__copy__()
        gpt -= x
        tt.assert_allclose(gpt(t), valgp - valx)

        gpt = gp.__copy__()
        if Profiles.isEmpty(x):
            def fun(group):
                group *= x
            tt.assert_raises(ValueError, fun, gpt)
        else:
            gpt *= x
            assert_allclose_appop(gpt(t), valgp * valx)

        gpt = gp.__copy__()
        if Profiles.isEmpty(x) or not(Profiles.isInversible(x, valx)):
            def fun(group):
                group /= x
            tt.assert_raises(ValueError, fun, gpt)
        else:
            gpt /= x
            assert_allclose_appop(gpt(t), valgp / valx)


def test_groupprofiles_IO():
    gp = Profiles.GroupProfiles(list_profiles())
    f = tmpfile()
    gp.save(f)
    gp2 = Profiles.load_groupprofiles(f)
    tt.assert_equal(gp, gp2)


def test_convert_value():
    " Test conversion from scalar"
    t, val = rd.randn(2)
    tt.assert_equal(Profiles.toProfile(val)(t), val)


def test_convert_prof():
    " Test conversion from Profile"
    t = rd.randn(1)
    a = get_random_Spline()[0]
    tt.assert_equal(Profiles.toProfile(a)(t), a(t))


#@tt.decorators.knownfailureif(True)
def test_spline_editor():
    # TODO: functional testing of GUI
    pass

if __name__ == '__main__':
    tt.run_module_suite()
    del_tmpfile()
