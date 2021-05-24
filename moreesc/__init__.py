from . import Profiles
from . import AcousticResonator
from . import Valve
from . import Simulation
from . import utils

__version__ = '2.2'

try:
    import nose
except ImportError:
    print('Nose needed for testing.')
else:
    test = nose.run

#__all__ = [ \
#    'Valve', 'Solveur_dim', 'SimuTempo', 'Profiles',\
#    'AcousticResonator', 'ModalExpansionEstimation', \
#    'ModulePOM', \
#    'AnalyseLineaireStabilite', 'GraphThetaGamma', 'ModesGraph','system_fortran']
#__all__ = [ \
#    'Valve', 'Profiles', 'AcousticResonator', 'Simulation', 'system_fortran']

