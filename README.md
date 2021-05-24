# Moreesc

I. Installation
    Dependancies :
    - python (>=2.5)
    - numpy (>1.0)
    - scipy (>0.7)
    - matplotlib (>1.0)
    - a C compiler
    Optional :
    - scikits.audiolab (in order to save wav files)
    - cython (>=0.15)
    - aubio (for audio analysis)
    - h5py (for storing results in HDF5 format instead of pickle binary format)
    
    One requirements are ready:
    - extract files from the archive
    - run the following commands in a terminal from the same folder 
      as setup.py and README:
      $ python setup.py build
      $ python setup.py install
      The last command may require root permissions, otherwise use the prefix:
      $ python setup.py install --prefix=/usr/local/

II. Minimal example

import sys
# Tell python where to find moreesc (if non-standard path)
sys.path.append('/your/path/to/the/folder/containing/the/moreesc/folder')
# Importing moreesc
import moreesc as mo
mp = mo.Profiles
mac = mo.AcousticResonator
mv = mo.Valve
ms = mo.Simulation
import numpy as np

# Bore and reed defition
D = mv.ReedDynamics(wr=2*np.pi*1500., qr=0.4, kr=8e6)
Ze = mac.Cylinder(L=.5, r=7e-3, radiates=False, nbmodes=10)

# Control parameters
tsim = 1.
H0 = mp.Constant(3e-4)
Pm = mp.SmoothStep([.001, .003, tsim-.2, tsim-.15], 1e3, 1e2)
# 2ms attack, almost .8s sustain, 50ms decay

# Creating the simulation object
sim = ms.TimeDomainSimulation(D, Ze, pm=Pm, h0=H0, 
    fs=44100, piecewise_constant=False)
# changing integrator and solving 
sim.set_integrator('vode', nsteps=20000)
sim.integrate(t=tsim)

sim.save('/tmp/test.h5')
sim.save_wav('/tmp/test.wav', where='in')
sim.trace(trace_all=True)
