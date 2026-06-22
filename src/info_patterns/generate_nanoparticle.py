from collections.abc import Callable
from typing import Any, Literal
import numpy as np

from pyGDM2 import structures
from pyGDM2 import materials

from info_patterns.constants import MATERIAL_DENSITIES_KG_M3, NM_TO_M

def nanoparticle_geometry(particle_type: str, step_nm: float, center: bool, **kwargs: Any) -> np.ndarray:
    """
    Generate a nanoparticle geometry using pyGDM2.structures.

    Parameters
    ----------
    particle_type : str
        Name of the pyGDM geometry function.
        Examples:
            "sphere"
            "spheroid"
            "nanodisc"
            "rect_wire"
            "prism"
            "hexagon"
            "polygon"

    step_nm : float
        Discretization step in nm.

    center : bool
        If True, center the geometry around its center of mass.

    **kwargs : Any
        Arguments required by the selected pyGDM geometry.

    Returns
    -------
    geometry : ndarray
        Nanoparticle geometry as dipole positions.
    """

    if not hasattr(structures, particle_type):
        raise ValueError(f"Geometry '{particle_type}' does not exist in pyGDM2.structures.\n"
            "Check the available pyGDM structures here:\n"
            "https://homepages.laas.fr/pwiecha/pygdm_doc/apidoc_sim_description.html#structures"
        )
    
    geometry_function = getattr(structures, particle_type)
    geometry = geometry_function(step_nm, **kwargs)
    if center:
        geometry = geometry - np.mean(geometry, axis=0)

    return geometry

def nanoparticle_material(material_name: str, **kwargs: Any) -> Any:
     """
    Generate a material using pyGDM2.materials.

    Parameters
    ----------
    material_name : str
        Name of the pyGDM material function.
        Examples:
            "sio2"
            "gold"
            "silver"
            "alu"
            "dummy"

    **kwargs : Any
        Arguments required by the selected material function.

    Returns
    -------
    material : Any
        pyGDM material object.
    """
     if not hasattr(materials, material_name):
        raise ValueError(f"Material '{material_name}' does not exist in pyGDM2.materials."
             "Check the available pyGDM materials here:\n"
            "https://homepages.laas.fr/pwiecha/pygdm_doc/apidoc_sim_description.html#materials"
        )
     material_function = getattr(materials, material_name)
     material = material_function(**kwargs)

     return material

def mesh_cell_volume(step_nm: float, mesh: Literal["cube", "hex"]) -> float:
    """
    Compute the effective volume of one pyGDM mesh cell.

    Parameters
    ----------
    step_nm : float
        Discretization step in nm.

    mesh : Literal["cube", "hex"]
        Mesh type.

    Returns
    -------
    cell_volume : float
        Effective cell volume in m^3.
    """

    step_m = step_nm * NM_TO_M

    if mesh == "cube":
        return step_m**3

    if mesh == "hex":
        return step_m**3 / np.sqrt(2.0)

    raise ValueError("mesh must be either 'cube' or 'hex'.")

def sphere_volume_m3(step_nm: float, R: float, **kwargs: Any) -> float:
    """
    Compute analytic volume of a sphere.

    Parameters
    ----------
    step_nm : float
        Discretization step in nm.

    R : float
        Sphere radius in units of step_nm.

    Returns
    -------
    volume : float
        Volume in m^3.
    """

    radius_m = R * step_nm * NM_TO_M
    volume = (4.0 / 3.0) * np.pi * radius_m**3

    return volume


def spheroid_volume_m3(step_nm: float, R1: float, R2: float, R3: float, **kwargs: Any) -> float:
    """
    Compute analytic volume of an ellipsoid/spheroid.

    Parameters
    ----------
    step_nm : float
        Discretization step in nm.

    R1, R2, R3 : float
        Semi-axes in units of step_nm.

    Returns
    -------
    volume : float
        Volume in m^3.
    """

    radius_x_m = R1 * step_nm * NM_TO_M
    radius_y_m = R2 * step_nm * NM_TO_M
    radius_z_m = R3 * step_nm * NM_TO_M

    volume = (4.0 / 3.0) * np.pi * radius_x_m * radius_y_m * radius_z_m

    return volume


def nanodisc_volume_m3(step_nm: float, R: float, H: float, **kwargs: Any) -> float:
    """
    Compute analytic volume of a cylindrical nanodisc.

    Parameters
    ----------
    step_nm : float
        Discretization step in nm.

    R : float
        Disc radius in units of step_nm.

    H : float
        Disc height in units of step_nm.

    Returns
    -------
    volume : float
        Volume in m^3.
    """

    radius_m = R * step_nm * NM_TO_M
    height_m = H * step_nm * NM_TO_M

    volume = np.pi * radius_m**2 * height_m

    return volume


def ellipse_volume_m3(step_nm: float, R1: float, R2: float, H: float, **kwargs: Any) -> float:
    """
    Compute analytic volume of an elliptical cylinder.

    Parameters
    ----------
    step_nm : float
        Discretization step in nm.

    R1, R2 : float
        Ellipse semi-axes in units of step_nm.

    H : float
        Cylinder height in units of step_nm.

    Returns
    -------
    volume : float
        Volume in m^3.
    """

    radius_x_m = R1 * step_nm * NM_TO_M
    radius_y_m = R2 * step_nm * NM_TO_M
    height_m = H * step_nm * NM_TO_M

    volume = np.pi * radius_x_m * radius_y_m * height_m

    return volume

SHAPE_VOLUME_FUNCTIONS: dict[str, Callable[..., float]] = {"sphere": sphere_volume_m3, "spheroid": spheroid_volume_m3, "nanodisc": nanodisc_volume_m3, "ellipse": ellipse_volume_m3}

def nanoparticle_mass_from_geometry(geometry: np.ndarray, step_nm: float, material_name: str, mesh: Literal["cube", "hex"]) -> float:
    """
    Estimate nanoparticle mass from a discretized pyGDM geometry.

    Parameters
    ----------
    geometry : np.ndarray
        Nanoparticle geometry with shape (N, 3).

    step_nm : float
        Discretization step in nm.

    material_name : str
        Material name used to select density.

    mesh : Literal["cube", "hex"]
        Mesh type used in the pyGDM geometry.

    Returns
    -------
    mass : float
        Estimated mass in kg.
    """

    if material_name not in MATERIAL_DENSITIES_KG_M3:
        raise ValueError(f"Density for material '{material_name}' is not defined. "
            f"Available materials are: {list(MATERIAL_DENSITIES_KG_M3.keys())}.")

    density = MATERIAL_DENSITIES_KG_M3[material_name]
    number_of_dipoles = len(geometry)
    cell_volume = mesh_cell_volume(step_nm=step_nm, mesh=mesh)

    mass = density * number_of_dipoles * cell_volume

    return mass

def nanoparticle_mass_from_shape(particle_type: str, step_nm: float, material_name: str, **kwargs: Any) -> float:
    """
    Estimate nanoparticle mass from an analytic shape volume.

    Parameters
    ----------
    particle_type : str
        Name of the particle shape.

    step_nm : float
        Discretization step in nm.

    material_name : str
        Material name used to select density.

    **kwargs : Any
        Shape parameters required by the analytic volume formula.

    Returns
    -------
    mass : float
        Estimated mass in kg.
    """

    if material_name not in MATERIAL_DENSITIES_KG_M3:
        raise ValueError(f"Density for material '{material_name}' is not defined. "
            f"Available materials are: {list(MATERIAL_DENSITIES_KG_M3.keys())}.")

    if particle_type not in SHAPE_VOLUME_FUNCTIONS:
        raise ValueError(f"Analytic volume for shape '{particle_type}' is not defined. "
            f"Available analytic shapes are: {list(SHAPE_VOLUME_FUNCTIONS.keys())}.")

    density = MATERIAL_DENSITIES_KG_M3[material_name]
    volume_function = SHAPE_VOLUME_FUNCTIONS[particle_type]
    volume = volume_function(step_nm=step_nm, **kwargs)

    mass = density * volume

    return mass


def nanoparticle_mass(geometry: np.ndarray, particle_type: str, step_nm: float, material_name: str, mesh: Literal["cube", "hex"], method: Literal["auto", "analytic", "discrete"], 
                      **kwargs: Any) -> float:
    """
    Estimate nanoparticle mass using either analytic or discrete volume.

    Parameters
    ----------
    geometry : np.ndarray
        Nanoparticle geometry with shape (N, 3).

    particle_type : str
        Name of the particle shape.

    step_nm : float
        Discretization step in nm.

    material_name : str
        Material name used to select density.

    mesh : Literal["cube", "hex"]
        Mesh type used in the pyGDM geometry.

    method : Literal["auto", "analytic", "discrete"]
        Mass calculation method.

    **kwargs : Any
        Shape parameters required by the analytic volume formula.

    Returns
    -------
    mass : float
        Estimated mass in kg.
    """

    if method == "analytic":
        return nanoparticle_mass_from_shape(particle_type=particle_type, step_nm=step_nm, material_name=material_name, **kwargs)

    if method == "discrete":
        return nanoparticle_mass_from_geometry(geometry=geometry, step_nm=step_nm, material_name=material_name, mesh=mesh)

    if method == "auto":
        if particle_type in SHAPE_VOLUME_FUNCTIONS:
            return nanoparticle_mass_from_shape(particle_type=particle_type, step_nm=step_nm, material_name=material_name, **kwargs)

        return nanoparticle_mass_from_geometry(geometry=geometry, step_nm=step_nm, material_name=material_name, mesh=mesh)

    raise ValueError("method must be 'auto', 'analytic', or 'discrete'.")