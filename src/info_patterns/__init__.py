"""
info_patterns

Library for optical information-pattern simulations of nanoparticles.
"""

# Import full submodules
from . import constants
from . import generate_nanoparticle
from . import information_patterns_simulation
from . import light_matter_interaction_simulation
from . import measurement_tools
from . import parameters
from . import plots

# Import all public objects from each submodule into the package namespace
from .constants import *
from .generate_nanoparticle import *
from .information_patterns_simulation import *
from .light_matter_interaction_simulation import *
from .measurement_tools import *
from .parameters import *
from .plots import *

# Build __all__ automatically from all imported public names
__all__ = [name for name in globals() if not name.startswith("_")]