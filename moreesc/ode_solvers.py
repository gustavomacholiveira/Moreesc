#! /usr/bin/python
# -*- coding: utf-8 -*-
# Copyright Fabrice Silva, 2012.

"""
"""
import warnings
import numpy as np
from scipy.integrate._ode import IntegratorBase

class Euler(IntegratorBase):
    " Euler's method, the simplest Runge-Kutta method. "
    runner = True

    def __init__(self, dt=.01, **kwargs):
        self.dt = dt
        self.eps = np.sqrt(np.finfo(np.float).eps)

    def reset(self,n,has_jac):
        pass

    def run(self,f,jac,y0,t0,t1,f_params,jac_params):
        if isinstance(y0, np.ndarray):
            yo = y0.__copy__()
        else:
            yo = np.asarray(y0)
        times = np.arange(t0, t1+self.eps, self.dt)
        for t in times[1:]:
            yo += f(t, yo, *f_params) * self.dt

        self.success = bool(np.isfinite(yo[-1]))
        return yo, t1


class EulerRichardson(Euler):
    def run(self,f,jac,y0,t0,t1,f_params,jac_params):
        if isinstance(y0, np.ndarray):
            yo = y0.__copy__()
        else:
            yo = np.asarray(y0)
        ymid = np.empty_like(yo)
        times = np.arange(t0, t1+self.eps, self.dt)
        dt_mid = .5 * self.dt

        for t in times[1:]:
            ymid[:] = yo + f(t, yo, *f_params) * dt_mid
            yo += f(t + dt_mid, ymid, *f_params)

        self.success = bool(np.isfinite(yo[-1]))
        return yo,t

IntegratorBase.integrator_classes.append(Euler)    
IntegratorBase.integrator_classes.append(EulerRichardson)

try:
    import lsoda as _lsoda
except:
    pass
else:
    class lsoda(IntegratorBase):

        runner = getattr(_lsoda, 'lsoda', None)
        active_global_handle = 0

        messages = {
            2: "Integration successful.",
            -1: "Excess work done on this call (perhaps wrong Dfun type).",
            -2: "Excess accuracy requested (tolerances too small).",
            -3: "Illegal input detected (internal error).",
            -4: "Repeated error test failures (internal error).",
            -5: "Repeated convergence failures (perhaps bad Jacobian or tolerances).",
            -6: "Error weight became zero during problem.",
            -7: "Internal workspace insufficient to finish (internal error)."
        }

        def __init__(self,
                     with_jacobian=0,
                     rtol=1e-6, atol=1e-12,
                     lband=None, uband=None,
                     nsteps=500,
                     max_step=0.0,  # corresponds to infinite
                     min_step=0.0,
                     first_step=0.0,  # determined by solver
                     ixpr=0,
                     max_hnil=0,
                     max_order_ns=12,
                     max_order_s=5):

            self.with_jacobian = with_jacobian
            self.rtol = rtol
            self.atol = atol
            self.mu = uband
            self.ml = lband

            self.max_order_ns = max_order_ns
            self.max_order_s = max_order_s
            self.nsteps = nsteps
            self.max_step = max_step
            self.min_step = min_step
            self.first_step = first_step
            self.ixpr = ixpr
            self.max_hnil = max_hnil
            self.success = 1

            self.initialized = False

        def reset(self, n, has_jac):
            # Calculate parameters for Fortran subroutine dvode.
            if has_jac:
                if self.mu is None and self.ml is None:
                    jt = 1
                else:
                    if self.mu is None:
                        self.mu = 0
                    if self.ml is None:
                        self.ml = 0
                    jt = 4
            else:
                if self.mu is None and self.ml is None:
                    jt = 2
                else:
                    if self.mu is None:
                        self.mu = 0
                    if self.ml is None:
                        self.ml = 0
                    jt = 5
            lrn = 20 + (self.max_order_ns + 4) * n
            if jt in [1, 2]:
                lrs = 22 + (self.max_order_s + 4) * n + n * n
            elif jt in [4, 5]:
                lrs = 22 + (self.max_order_s + 5 + 2 * self.ml + self.mu) * n
            else:
                raise ValueError('Unexpected jt=%s' % jt)
            lrw = max(lrn, lrs)
            liw = 20 + n
            rwork = np.zeros((lrw,), float)
            rwork[4] = self.first_step
            rwork[5] = self.max_step
            rwork[6] = self.min_step
            self.rwork = rwork
            iwork = np.zeros((liw,), np.int32)
            if self.ml is not None:
                iwork[0] = self.ml
            if self.mu is not None:
                iwork[1] = self.mu
            iwork[4] = self.ixpr
            iwork[5] = self.nsteps
            iwork[6] = self.max_hnil
            iwork[7] = self.max_order_ns
            iwork[8] = self.max_order_s
            self.iwork = iwork
            self.call_args = [self.rtol, self.atol, 1, 1,
                              self.rwork, self.iwork, jt]
            self.success = 1
            self.initialized = False

        def run(self, f,jac,y0,t0,t1,f_params,jac_params):
            if self.initialized:
                self.check_handle()
            else:
                self.initialized = True
                self.acquire_new_handle()
            args = [f, y0, t0, t1] + self.call_args[:-1] + \
                   [jac, self.call_args[-1], f_params, 0, jac_params]
            y1, t, istate = self.runner(*args)
            if istate < 0:
                warnings.warn('lsoda: ' +
                    self.messages.get(istate, 'Unexpected istate=%s' % istate))
                self.success = 0
            else:
                self.call_args[3] = 2  # upgrade istate from 1 to 2
            return y1, t

        def step(self, *args):
            itask = self.call_args[2]
            self.call_args[2] = 2
            r = self.run(*args)
            self.call_args[2] = itask
            return r

        def run_relax(self, *args):
            itask = self.call_args[2]
            self.call_args[2] = 3
            r = self.run(*args)
            self.call_args[2] = itask
            return r

    if lsoda.runner:
        IntegratorBase.integrator_classes.append(lsoda)
