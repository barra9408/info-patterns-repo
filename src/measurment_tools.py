from typing import Any, Literal

import numpy as np
from pyGDM2 import linear
from scipy.spatial.transform import Rotation as R

from scr.simulation import simulation_from_geometry

def max_detection_efficiency(I_pattern: np.ndarray, Nteta: int, Nphi: int, theta_max: float) -> float:
    """
    Compute the detection efficiency of an information pattern
    inside the angular region theta <= theta_max.

    Parameters
    ----------
    I_pattern : np.ndarray
        Information pattern with shape (Nteta, Nphi).

    Nteta : int
        Number of theta samples.

    Nphi : int
        Number of phi samples.

    theta_max : float
        Maximum collection angle in radians.

    Returns
    -------
    eta : float
        Detection efficiency.
    """

    theta = np.linspace(0, np.pi, Nteta)
    phi = np.linspace(0, 2 * np.pi, Nphi)

    Theta, Phi = np.meshgrid(theta, phi, indexing="ij")

    weights = np.sin(Theta)
    detection_mask = Theta <= theta_max

    numerator = np.sum(I_pattern[detection_mask] * weights[detection_mask])
    denominator = np.sum(I_pattern * weights)

    eta = numerator / denominator

    return eta

def optical_force(sim: Any, field_index: int, return_value: Literal["force", "torque"]) -> np.ndarray:
    """
    Compute total optical force or torque from a solved pyGDM simulation.

    Parameters
    ----------
    sim : Any
        pyGDM simulation after core.scatter(sim).

    field_index : int
        Incident-field index.

    return_value : Literal["force", "torque"]
        Quantity to return.

    Returns
    -------
    result : np.ndarray
        Total force [Fx, Fy, Fz] or total torque [Tx, Ty, Tz].
    """

    if sim.E is None:
        raise ValueError("Run core.scatter(sim) before computing force or torque.")

    Eint = sim.E[field_index][1]
    alpha_tensor = sim.dyads.getPolarizabilityTensor(sim.E[field_index][0]["wavelength"], sim.struct)
    P = np.matmul(alpha_tensor, Eint[..., None])[..., 0]

    gradE = linear.field_gradient(sim, field_index)
    dEdx = gradE[0][..., 3:]
    dEdy = gradE[1][..., 3:]
    dEdz = gradE[2][..., 3:]

    Fx = 0.5 * np.real(P[:, 0] * np.conj(dEdx[:, 0]) + P[:, 1] * np.conj(dEdx[:, 1]) + P[:, 2] * np.conj(dEdx[:, 2]))
    Fy = 0.5 * np.real(P[:, 0] * np.conj(dEdy[:, 0]) + P[:, 1] * np.conj(dEdy[:, 1]) + P[:, 2] * np.conj(dEdy[:, 2]))
    Fz = 0.5 * np.real(P[:, 0] * np.conj(dEdz[:, 0]) + P[:, 1] * np.conj(dEdz[:, 1]) + P[:, 2] * np.conj(dEdz[:, 2]))
    force_density = np.column_stack([Fx, Fy, Fz])

    if return_value == "force":
        return np.sum(force_density, axis=0)

    elif return_value == "torque":
        r = sim.struct.geometry
        torque_density = np.cross(r, force_density)
        return np.sum(torque_density, axis=0)

    else:
        raise ValueError("return_value must be either 'force' or 'torque'.")
    
def force_vs_displacement(geometry: np.ndarray, step_nm: float, material: Any, efield: Any, dyads: Any, displacements: np.ndarray, field_index: int) -> dict[str, np.ndarray]:
    """
    Compute optical force as a function of COM displacement.

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

    displacements : np.ndarray
        Array of COM displacements in nm.

    field_index : int
        Incident-field index.

    Returns
    -------
    Force : dict[str, np.ndarray]
        Force["X"], Force["Y"], Force["Z"].
        Each one has shape (len(displacements), 3).
    """

    eps0 = 8.854e-12
    force_conv = 4 * np.pi * eps0 * 1e-18

    axes = ["X", "Y", "Z"]
    Force = {}

    for axis_index, axis_name in enumerate(axes):
        values = []

        for disp in displacements:
            shifted_geometry = geometry.copy()
            shifted_geometry[:, axis_index] += disp
            sim = simulation_from_geometry(geometry=shifted_geometry, step_nm=step_nm, material=material, efield=efield, dyads=dyads)
            F = optical_force(sim=sim, field_index=field_index, return_value="force")
            values.append(F)
        Force[axis_name] = force_conv * np.array(values)

    return Force

def torque_vs_rotation(geometry: np.ndarray, step_nm: float, material: Any, efield: Any, dyads: Any, angles_deg: np.ndarray, field_index: int) -> dict[str, np.ndarray]:
    """
    Compute optical torque as a function of angular displacement.

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

    angles_deg : np.ndarray
        Array of angular displacements in degrees.

    field_index : int
        Incident-field index.

    Returns
    -------
    Torque : dict[str, np.ndarray]
        Torque["X"], Torque["Y"], Torque["Z"].
        Each one has shape (len(angles_deg), 3).
    """

    eps0 = 8.854e-12
    force_conv = 4 * np.pi * eps0 * 1e-18
    torque_conv = force_conv * 1e-9

    axes = ["X", "Y", "Z"]
    Torque = {}

    for axis_index, axis_name in enumerate(axes):
        values = []

        for angle in angles_deg:
            angles = [0.0, 0.0, 0.0]
            angles[axis_index] = angle
            rot = R.from_euler("xyz", angles, degrees=True)
            rotated_geometry = geometry @ rot.as_matrix().T
            sim = simulation_from_geometry(geometry=rotated_geometry, step_nm=step_nm,material=material, efield=efield, dyads=dyads)
            T = optical_force(sim=sim, field_index=field_index, return_value="torque")
            values.append(T)
        Torque[axis_name] = torque_conv * np.array(values)

    return Torque

def recoil_force_noise_psd(Escat: np.ndarray, wavelength_nm: float, Nteta: int, Nphi: int, mode_axis: int) -> float:
    """
    Compute the recoil force-noise PSD S_FF along a mechanical mode.

    Parameters
    ----------
    Escat : np.ndarray
        Scattered far-field electric field with shape (Nteta * Nphi, 3).

    wavelength_nm : float
        Optical wavelength in nm.

    Nteta : int
        Number of theta points.

    Nphi : int
        Number of phi points.

    mode_axis : int
        Mechanical axis:
            0 -> x
            1 -> y
            2 -> z

    Returns
    -------
    S_FF : float
        Recoil force-noise PSD along the selected mode.
    """

    hbar = 1.054571817e-34
    c = 299792458.0
    eps0 = 8.8541878128e-12

    wavelength_m = wavelength_nm * 1e-9
    omega = 2 * np.pi * c / wavelength_m

    theta = np.linspace(0, np.pi, Nteta)
    phi = np.linspace(0, 2 * np.pi, Nphi)
    Theta, Phi = np.meshgrid(theta, phi, indexing="ij")

    n_x = np.sin(Theta) * np.cos(Phi)
    n_y = np.sin(Theta) * np.sin(Phi)
    n_z = np.cos(Theta)

    n_components = [n_x, n_y, n_z]
    n_mu = n_components[mode_axis]

    Escat_abs2 = np.sum(np.abs(Escat)**2, axis=1).reshape(Nteta, Nphi)

    integrand = Escat_abs2 * n_mu**2 * np.sin(Theta)

    dtheta = theta[1] - theta[0]
    dphi = phi[1] - phi[0]

    S_FF = (hbar * omega * eps0 * c / 2) * np.sum(integrand) * dtheta * dphi

    return S_FF

def trap_frequency(displacements_nm: np.ndarray, forces_N: np.ndarray, mass: float) -> tuple[float, float]:
    """
    Compute the harmonic trap frequency from a force-displacement curve.

    Parameters
    ----------
    displacements_nm : np.ndarray
        Particle displacements in nm.

    forces_N : np.ndarray
        Optical force along the same direction in N.

    mass : float
        Particle mass in kg.

    Returns
    -------
    Omega : float
        Angular trap frequency in rad/s. Returns np.nan if not restoring.

    k_trap : float
        Trap stiffness in N/m.
    """

    displacements_m = displacements_nm * 1e-9
    slope = np.polyfit(displacements_m, forces_N, 1)[0]
    k_trap = -slope

    if k_trap <= 0:
        return np.nan, k_trap

    Omega = np.sqrt(k_trap / mass)

    return Omega, k_trap

def heating_rate(S_FF: float, mass: float, Omega_mu: float) -> float:
    """
    Compute the recoil heating rate Gamma_mu.

    Parameters
    ----------
    S_FF : float
        Recoil force-noise PSD.

    mass : float
        Particle mass in kg.

    Omega_mu : float
        Mechanical frequency in rad/s.

    Returns
    -------
    Gamma_mu : float
        Recoil heating rate.
    """

    hbar = 1.054571817e-34
    Gamma_mu = np.pi * S_FF / (mass * hbar * Omega_mu)

    return Gamma_mu