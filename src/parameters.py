import numpy as np

# Geometry parameters
DEFAULT_STEP_NM: float = 10.0
DEFAULT_CENTER_GEOMETRY: bool = True
DEFAULT_MESH: str = "hex"

SPHERE_GEOMETRY_PARAMS: dict = {"particle_type": "sphere", "step_nm": DEFAULT_STEP_NM, "center": DEFAULT_CENTER_GEOMETRY, "R": 5, "mesh": DEFAULT_MESH}
ELLIPSOID_GEOMETRY_PARAMS: dict = {"particle_type": "spheroid", "step_nm": DEFAULT_STEP_NM, "center": DEFAULT_CENTER_GEOMETRY, "R1": 8, "R2": 5, "R3": 5, "mesh": DEFAULT_MESH}

# Material parameters
DEFAULT_MATERIAL_NAME: str = "sio2"
DEFAULT_MATERIAL_KWARGS: dict = {}

# Optical sampling parameters
DEFAULT_WAVELENGTH_NM: float = 1550.0
DEFAULT_FARFIELD_PARAMS: dict = {"Nteta": 100, "Nphi": 360, "field_index": 0, "r": 1e5}
FAST_FARFIELD_PARAMS: dict = {"Nteta": 30, "Nphi": 60, "field_index": 0, "r": 1e5}

# Incident field parameters
DEFAULT_GAUSSIAN_FIELD_PARAMS: dict = {"field_generator": "gaussian", "wavelengths": [DEFAULT_WAVELENGTH_NM], "NA": 0.1, "polarization_state": (1, 0, 0, 0)}

# Propagator parameters
DEFAULT_DYADS_PARAMS: dict = {"dyads_name": "DyadsQuasistatic123", "n1": 1.0, "n2": 1.0}

# Mechanical perturbation parameters
DEFAULT_COM_DISPLACEMENT_NM: float = 1.0
DEFAULT_LIBRATION_ANGLE_DEG: float = 10.0

DEFAULT_FORCE_DISPLACEMENTS_NM: np.ndarray = np.linspace(-5.0, 5.0, 11)
DEFAULT_ROTATION_ANGLES_DEG: np.ndarray = np.arange(-10.0, 11.0, 5.0)

# Detection parameters
PLUS_Z_HEMISPHERE_THETA_MAX: float = np.pi / 2.0

# Plot parameters
DEFAULT_USETEX: bool = True
DEFAULT_FONT_FAMILY: str = "serif"
DEFAULT_FIGSIZE: tuple[float, float] = (6.5, 4)
DEFAULT_RESULTS_DIR: str = "results"
DEFAULT_SAVE_TYPE: str = "pdf"
DEFAULT_DPI: int = 300

DEFAULT_3D_SCALE: float = 0.47
DEFAULT_MINI_AXIS_POSITION: list[float] = [0.855, 0.25, 0.12, 0.20]

DEFAULT_2D_PLANES: tuple[str, str, str] = ("xy", "xz", "yz")
DEFAULT_2D_NORMALIZE: bool = True
DEFAULT_2D_FILL: bool = True