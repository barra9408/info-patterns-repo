import numpy as np

# Physical constants
HBAR: float = 1.054571817e-34
C: float = 299792458.0
EPS0: float = 8.8541878128e-12

# Unit conversions
NM_TO_M: float = 1e-9
NM2_TO_M2: float = 1e-18
NM3_TO_M3: float = 1e-27

# pyGDM force and torque conversion factors
FORCE_CONVERSION: float = 4.0 * np.pi * EPS0 * NM2_TO_M2
TORQUE_CONVERSION: float = FORCE_CONVERSION * NM_TO_M

# Coordinate conventions
AXES: tuple[str, str, str] = ("X", "Y", "Z")
AXIS_INDICES: dict[str, int] = {"x": 0, "y": 1, "z": 2, "X": 0, "Y": 1, "Z": 2}

# Material densities in kg / m^3
MATERIAL_DENSITIES_KG_M3: dict[str, float] = {"sio2": 2200.0, "gold": 19300.0, "silver": 10490.0, "alu": 2700.0, "silicon": 2330.0}

# Angular integration limits
FULL_THETA_MIN: float = 0.0
FULL_THETA_MAX: float = np.pi
FULL_PHI_MIN: float = 0.0
FULL_PHI_MAX: float = 2.0 * np.pi

PLUS_Z_HEMISPHERE_THETA_MAX: float = np.pi / 2.0