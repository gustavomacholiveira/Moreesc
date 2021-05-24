name = "moreesc"
try:
    from moreesc import __version__ as version
except ImportError:
    import os
    f = open(os.path.join(name, "__init__.py"), 'r')
    for l in f:
        if l.strip().startswith("__version__"):
            version = str(l.strip()[len("__version__"):].strip(" ="))
            print(version)
            break
    del f, l, os
    
author = "F. Silva, Ch. Vergez, J. Kergomard and Ph. Guillemain"
author_email = "silva@lma.cnrs-mrs.fr"
maintainer = "Fabrice Silva"
maintainer_email = "silva@lma.cnrs-mrs.fr"
url = r"http://moreesc.lma.cnrs-mrs.fr/"
copyright = "Copyright 2007-2016 Fabrice Silva"
license = "CECILL-C"
platforms = ["unix", "windows","mac-osx"]
requires = ['numpy(>=1.1)', 'scipy(>=0.7)', 'matplotlib(>=0.98)', 'tempfile']
description = "Modal Resonator-Reed Interaction Simulation Code"
long_description = """MOREESC is an application for the calculations of the auto-oscillations in wind instruments, designed originally for single reed instruments but it can be used for brass too. It takes advantage of a modal decomposition of the acoustic pressure field in the bore, or of its input impedance, to let you compute the sound of any strangely-shaped instrument.
"""
