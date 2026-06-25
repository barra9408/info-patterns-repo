from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from pyGDM2 import core
from pyGDM2 import fields
from pyGDM2 import linear
from pyGDM2 import propagators
from pyGDM2 import structures
from scipy.spatial.transform import Rotation as R

from info_patterns.constants import (FULL_THETA_MIN, FULL_THETA_MAX, FULL_PHI_MIN, FULL_PHI_MAX)

def w0_from_filling_factor(gamma: float, NA: float, f_mm: float) -> float:
    """
    Convert a filling factor into the input waist before focusing.

    The convention used for the pyGDM Richards-Wolf vector beam generators is

        gamma = w0 / (NA f)

    where both w0 and f are expressed in mm.

    Parameters
    ----------
    gamma : float
        Filling factor.

    NA : float
        Numerical aperture.

    f_mm : float
        Lens focal distance in mm.

    Returns
    -------
    w0_mm : float
        Beam waist before focusing, in mm.
    """

    w0_mm = gamma * NA * f_mm

    return w0_mm

def hermite_gauss00_power_amplitude_factor(optical_power_W: float, w0_mm: float, impedance_ohm: float) -> float:
    """
    Compute the electric-field amplitude factor for pyGDM's HermiteGauss00
    field when normalize=False.

    The convention used here follows the requested scaling:

        A = sqrt(2 * P * Z0 / (pi * w0))

    where P is the optical power before focusing, Z0 is the free-space
    impedance, and w0 is the input waist before focusing.

    Parameters
    ----------
    optical_power_W : float
        Optical power before focusing, in W.

    w0_mm : float
        Beam waist before focusing, in mm.

    impedance_ohm : float
        Free-space impedance, approximately 377 Ohm.

    Returns
    -------
    amplitude : float
        Multiplicative field-amplitude factor.
    """

    w0_m = w0_mm * 1e-3
    amplitude = np.sqrt(2.0 * optical_power_W * impedance_ohm / (np.pi * w0_m))

    return amplitude


def incident_field(field_generator: str | Callable[..., Any], wavelengths: Sequence[float], **kwargs: Any) -> Any:
    """
    Generate an incident field using either a default pyGDM field
    or a user-defined field function.

    Parameters
    ----------
    field_generator : str | Callable[..., Any]
        If str, name of a field function inside pyGDM2.fields.
        Examples:
            "gaussian"
            "plane_wave"

        If callable, user-defined field function compatible with pyGDM.
        Example:
            parabolic_mirror_efield

    wavelengths : Sequence[float]
        Wavelengths in nm.

    **kwargs : Any
        Arguments required by the selected field generator.

    Returns
    -------
    efield : Any
        pyGDM incident field object.
    """

    if isinstance(field_generator, str):
        if not hasattr(fields, field_generator):
            raise ValueError(f"Field '{field_generator}' does not exist in pyGDM2.fields.\n"
                "Check the available pyGDM fields here:\n"
                "https://homepages.laas.fr/pwiecha/pygdm_doc/apidoc_sim_description.html#fields"
            )
        field_function = getattr(fields, field_generator)
    else:
        field_function = field_generator

    efield = fields.efield(field_function, wavelengths=wavelengths, kwargs=kwargs)
    
    return efield

def field_propagation(dyads_name: str, **kwargs: Any) -> Any:
    """
    Generate the electromagnetic propagator used by pyGDM2.

    Parameters
    ----------
    dyads_name : str
        Name of the dyadic propagator in pyGDM2.propagators.
        Example:
            "DyadsQuasistatic123"
            "DyadsQuasistatic2D123"
            "DyadsQuasistaticPeriodic123"

    **kwargs : Any
        Arguments required by the selected propagator.

    Returns
    -------
    dyads : Any
        pyGDM dyadic propagator object.
    """

    if not hasattr(propagators, dyads_name):
        raise ValueError(f"Dyads '{dyads_name}' does not exist in pyGDM2.propagators."
            "Check the available pyGDM propagators here:\n"
            "https://homepages.laas.fr/pwiecha/pygdm_doc/apidoc_sim_description.html#environment-set-of-green-s-dyads"
        )

    dyads_function = getattr(propagators, dyads_name)
    dyads = dyads_function(**kwargs)

    return dyads

def simulation_from_geometry(geometry: np.ndarray, step_nm: float, material: Any, efield: Any, dyads: Any) -> Any:
    """
    Build and run a pyGDM simulation from a given geometry.

    Parameters
    ----------
    geometry : np.ndarray
        Nanoparticle geometry as dipole positions in nm.

    step_nm : float
        Discretization step in nm.

    material : Any
        pyGDM material object.

    efield : Any
        pyGDM incident field object.

    dyads : Any
        pyGDM dyadic propagator object.

    Returns
    -------
    sim : Any
        Solved pyGDM simulation object.
    """

    struct = structures.struct(step_nm, geometry, material)
    sim = core.simulation(struct, efield, dyads)
    core.scatter(sim)

    return sim

def scattered_farfield_from_simulation(geometry: np.ndarray, step_nm: float, material: Any, efield: Any, dyads: Any, Nteta: int, Nphi: int, field_index: int, r: float) -> tuple[Any, ...]:
    """
    Compute the scattered far field from a given nanoparticle geometry.

    This function builds the pyGDM structure from the input geometry and
    material, creates the electromagnetic simulation, solves the scattering
    problem, and extracts the scattered far-field electric field.

    Parameters
    ----------
    geometry : np.ndarray
        Nanoparticle geometry as dipole positions in nm.
        Each row corresponds to one dipole position [x, y, z].

    step_nm : float
        Discretization step in nm.

    material : Any
        pyGDM material object assigned to the nanoparticle.

    efield : Any
        pyGDM incident field object.

    dyads : Any
        pyGDM dyadic propagator object.

    Nteta : int
        Number of sampling points along the polar angle theta.

    Nphi : int
        Number of sampling points along the azimuthal angle phi.

    field_index : int
        Index of the incident-field configuration to use.

    r : float
        Radius at which the far field is evaluated.

    Returns
    -------
    farfield : tuple[Any, ...]
        Scattered far-field data returned by pyGDM2.linear.farfield.
        Usually contains angular coordinates and the electric field.
    """

    sim = simulation_from_geometry(geometry=geometry, step_nm=step_nm, material=material, efield=efield, dyads=dyads)
    farfield = linear.farfield(sim, field_index=field_index, r_probe=None, r=r, tetamin=FULL_THETA_MIN, tetamax=FULL_THETA_MAX, Nteta=Nteta, phimin=FULL_PHI_MIN, phimax=FULL_PHI_MAX, Nphi=Nphi, polarizerangle="none", 
                               return_value="efield", normalization_E0=False)

    return farfield

def com_scattered_farfield(geometry: np.ndarray, step_nm: float, material: Any, efield: Any, dyads: Any, axis_index: int, disp_nm: float, Nteta: int, Nphi: int, field_index: int, r: float
                           ) -> tuple[tuple[Any, ...], tuple[Any, ...], np.ndarray]:
    """
    Compute the scattered far-field difference for a center-of-mass translation.

    This function applies a positive and negative displacement to the
    nanoparticle geometry along a selected Cartesian axis. It then computes
    the scattered far field for both displaced geometries and returns their
    field difference.

    Parameters
    ----------
    geometry : np.ndarray
        Nanoparticle geometry as dipole positions in nm.
        Each row corresponds to one dipole position [x, y, z].

    step_nm : float
        Discretization step in nm.

    material : Any
        pyGDM material object assigned to the nanoparticle.

    efield : Any
        pyGDM incident field object.

    dyads : Any
        pyGDM dyadic propagator object.

    axis_index : int
        Translation axis:
            0 -> x
            1 -> y
            2 -> z

    disp_nm : float
        Translation amplitude in nm.

    Nteta : int
        Number of sampling points along the polar angle theta.

    Nphi : int
        Number of sampling points along the azimuthal angle phi.

    field_index : int
        Index of the incident-field configuration to use.

    r : float
        Radius at which the far field is evaluated.

    Returns
    -------
    E_plus : tuple[Any, ...]
        Scattered far field for the positively displaced geometry.

    E_minus : tuple[Any, ...]
        Scattered far field for the negatively displaced geometry.

    dE : np.ndarray
        Difference between the scattered electric fields:
        E_plus[2] - E_minus[2].
    """

    shift = np.array([0.0, 0.0, 0.0])
    shift[axis_index] = disp_nm

    geometry_plus = geometry + shift
    geometry_minus = geometry - shift

    E_plus = scattered_farfield_from_simulation(geometry=geometry_plus, step_nm=step_nm, material=material, efield=efield, dyads=dyads, Nteta=Nteta, Nphi=Nphi, field_index=field_index, r=r)
    E_minus = scattered_farfield_from_simulation(geometry=geometry_minus, step_nm=step_nm, material=material, efield=efield, dyads=dyads, Nteta=Nteta, Nphi=Nphi, field_index=field_index, r=r)
    dE = E_plus[2] - E_minus[2]

    return E_plus, E_minus, dE

def librational_scattered_farfield(geometry: np.ndarray, step_nm: float, material: Any, efield: Any, dyads: Any, axis_index: int, angle_deg: float, Nteta: int, Nphi: int, field_index: int, 
                                   r: float) -> tuple[tuple[Any, ...], tuple[Any, ...], np.ndarray]:
    """
    Compute the scattered far-field difference for a librational rotation.

    This function applies a positive and negative rotation to the nanoparticle
    geometry around a selected Cartesian axis. It then computes the scattered
    far field for both rotated geometries and returns their field difference.

    Parameters
    ----------
    geometry : np.ndarray
        Nanoparticle geometry as dipole positions in nm.
        Each row corresponds to one dipole position [x, y, z].

    step_nm : float
        Discretization step in nm.

    material : Any
        pyGDM material object assigned to the nanoparticle.

    efield : Any
        pyGDM incident field object.

    dyads : Any
        pyGDM dyadic propagator object.

    axis_index : int
        Rotation axis:
            0 -> x
            1 -> y
            2 -> z

    angle_deg : float
        Rotation amplitude in degrees.

    Nteta : int
        Number of sampling points along the polar angle theta.

    Nphi : int
        Number of sampling points along the azimuthal angle phi.

    field_index : int
        Index of the incident-field configuration to use.

    r : float
        Radius at which the far field is evaluated.

    Returns
    -------
    E_plus : tuple[Any, ...]
        Scattered far field for the positively rotated geometry.

    E_minus : tuple[Any, ...]
        Scattered far field for the negatively rotated geometry.

    dE : np.ndarray
        Difference between the scattered electric fields:
        E_plus[2] - E_minus[2].
    """

    angles_plus = [0.0, 0.0, 0.0]
    angles_minus = [0.0, 0.0, 0.0]

    angles_plus[axis_index] = angle_deg
    angles_minus[axis_index] = -angle_deg

    rot_plus = R.from_euler("xyz", angles_plus, degrees=True)
    rot_minus = R.from_euler("xyz", angles_minus, degrees=True)

    geometry_plus = np.dot(geometry, rot_plus.as_matrix().T)
    geometry_minus = np.dot(geometry, rot_minus.as_matrix().T)

    E_plus = scattered_farfield_from_simulation(geometry=geometry_plus, step_nm=step_nm, material=material, efield=efield, dyads=dyads, Nteta=Nteta, Nphi=Nphi, field_index=field_index, r=r)
    E_minus = scattered_farfield_from_simulation(geometry=geometry_minus, step_nm=step_nm, material=material, efield=efield, dyads=dyads, Nteta=Nteta, Nphi=Nphi, field_index=field_index, r=r)
    dE = E_plus[2] - E_minus[2]

    return E_plus, E_minus, dE