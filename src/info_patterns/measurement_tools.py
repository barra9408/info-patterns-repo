from typing import Any, Literal
from contextlib import nullcontext

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None

import os
import numpy as np
from pyGDM2 import linear
from scipy.spatial.transform import Rotation as R
from concurrent.futures import ProcessPoolExecutor, as_completed

from info_patterns.generate_nanoparticle import nanoparticle_material
from info_patterns.light_matter_interaction_simulation import (incident_field, field_propagation, simulation_from_geometry)
from info_patterns.constants import (HBAR, C, EPS0, MU0, NM_TO_M, FORCE_CONVERSION, TORQUE_CONVERSION, AXES)
from info_patterns.parallel_utils import (blas_single_thread_context, resolve_parallel_execution)

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

    gradE = linear.field_gradient(sim, field_index, which_fields=["E0"])
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

def _automatic_chunks_per_axis(n_values: int, n_axes: int, n_workers: int) -> int:
    """
    Choose an automatic number of chunks per axis.

    The goal is to create enough tasks to use the available workers, but without
    creating one tiny task per displacement or angle.

    Parameters
    ----------
    n_values : int
        Number of displacements or angles.

    n_axes : int
        Number of axes.

    n_workers : int
        Number of resolved worker processes.

    Returns
    -------
    chunks_per_axis : int
        Number of chunks to split each axis into.
    """

    if n_values < 1:
        raise ValueError("n_values must be >= 1.")
    if n_axes < 1:
        raise ValueError("n_axes must be >= 1.")
    if n_workers < 1:
        raise ValueError("n_workers must be >= 1.")

    target_total_chunks = min(n_workers, n_axes * n_values)
    chunks_per_axis = int(np.ceil(target_total_chunks / n_axes))

    return max(1, min(chunks_per_axis, n_values))


def _force_vs_displacement_worker(geometry: np.ndarray, step_nm: float, material_name: str, material_kwargs: dict[str, Any], efield_params: dict[str, Any], dyads_params: dict[str, Any], 
                                  axis_index: int, displacement_indices: np.ndarray, displacements: np.ndarray, field_index: int) -> tuple[int, np.ndarray, np.ndarray]:
    """
    Worker process for one force-vs-displacement axis chunk.
    """

    material = nanoparticle_material(material_name, **material_kwargs)
    efield = incident_field(**efield_params)
    dyads = field_propagation(**dyads_params)

    values = []
    for idx in displacement_indices:
        disp = displacements[idx]
        shifted_geometry = geometry.copy()
        shifted_geometry[:, axis_index] += float(disp)

        with blas_single_thread_context():
            sim = simulation_from_geometry(geometry=shifted_geometry, step_nm=step_nm, material=material, efield=efield, dyads=dyads)
            F = optical_force(sim=sim, field_index=field_index, return_value="force")
        values.append(F)

    return axis_index, displacement_indices, FORCE_CONVERSION * np.array(values)

def force_vs_displacement(geometry: np.ndarray, step_nm: float, material: Any, efield: Any, dyads: Any, displacements: np.ndarray, field_index: int, *, 
                          parallel: bool | Literal["auto"] = False, max_workers: int | None = None, material_name: str | None = None, material_kwargs: dict[str, Any] | None = None, 
                          efield_params: dict[str, Any] | None = None, dyads_params: dict[str, Any] | None = None, verbose: bool = False) -> dict[str, np.ndarray]:
    """
    Compute optical force as a function of COM displacement.

    Parameters
    ----------
    geometry : np.ndarray
        Nanoparticle geometry.

    step_nm : float
        Discretization step in nm.

    material : Any
        pyGDM material object. Used in serial mode.

    efield : Any
        pyGDM incident field object. Used in serial mode.

    dyads : Any
        pyGDM dyadic propagator object. Used in serial mode.

    displacements : np.ndarray
        COM displacements in nm.

    field_index : int
        Incident-field configuration index.

    parallel : bool | str, optional
        Execution mode. Accepted values are False, True, and "auto".
        Default is False.

    max_workers : int | None, optional
        Maximum number of worker processes. Default is None.

    material_name : str | None, optional
        Material name used to reconstruct the material inside each process.
        Required when parallel execution is used. Default is None.

    material_kwargs : dict[str, Any] | None, optional
        Keyword arguments for nanoparticle_material.
        Required when parallel execution is used. Default is None.

    efield_params : dict[str, Any] | None, optional
        Keyword arguments for incident_field.
        Required when parallel execution is used. Default is None.

    dyads_params : dict[str, Any] | None, optional
        Keyword arguments for field_propagation.
        Required when parallel execution is used. Default is None.

    verbose : bool, optional
        If True, print execution-mode messages. Default is None.

    Returns
    -------
    Force : dict[str, np.ndarray]
        Dictionary with keys "X", "Y", "Z". Each entry is an array with the
        force vectors evaluated at the requested displacements.
    """

    axes = AXES
    displacements = np.asarray(displacements)
    n_displacements = len(displacements)
    n_tasks = len(axes) * n_displacements

    use_parallel, resolved_workers = resolve_parallel_execution(parallel=parallel, max_workers=max_workers, n_tasks=n_tasks, verbose=verbose)
    if not use_parallel:
        Force = {}
        for axis_index, axis_name in enumerate(axes):
            values = []
            for disp in displacements:
                shifted_geometry = geometry.copy()
                shifted_geometry[:, axis_index] += float(disp)
                sim = simulation_from_geometry(geometry=shifted_geometry, step_nm=step_nm, material=material, efield=efield, dyads=dyads)
                F = optical_force(sim=sim, field_index=field_index, return_value="force")
                values.append(F)

            Force[axis_name] = FORCE_CONVERSION * np.array(values)

        return Force

    required_parallel_args = {"material_name": material_name, "material_kwargs": material_kwargs, "efield_params": efield_params, "dyads_params": dyads_params}
    missing = [name for name, value in required_parallel_args.items() if value is None]
    if missing:
        raise ValueError("Parallel execution requires the following additional arguments: "
            + ", ".join(missing)
            + ". This is needed because each process must reconstruct "
            "material, efield, and dyads independently.")

    Force = {axis_name: np.empty((n_displacements, 3)) for axis_name in axes}
    chunks_per_axis = _automatic_chunks_per_axis(n_values=n_displacements, n_axes=len(axes), n_workers=resolved_workers)

    if verbose:
        print(f"Using {chunks_per_axis} chunks per axis "
            f"({chunks_per_axis * len(axes)} total chunks).")

    tasks = []
    for axis_index in range(len(axes)):
        indices = np.arange(n_displacements)
        index_chunks = np.array_split(indices, chunks_per_axis)
        for chunk in index_chunks:
            if len(chunk) > 0:
                tasks.append((axis_index, chunk))

    with ProcessPoolExecutor(max_workers=resolved_workers) as executor:
        futures = [executor.submit(_force_vs_displacement_worker, geometry, step_nm, material_name, material_kwargs, efield_params, dyads_params, axis_index, chunk, displacements, 
                                   field_index) for axis_index, chunk in tasks]
        for future in as_completed(futures):
            axis_index, displacement_indices, force_values = future.result()
            axis_name = axes[axis_index]
            Force[axis_name][displacement_indices, :] = force_values

    return Force    

def _torque_vs_rotation_worker(geometry: np.ndarray, step_nm: float, material_name: str, material_kwargs: dict[str, Any], efield_params: dict[str, Any], dyads_params: dict[str, Any], 
                               axis_index: int, angle_indices: np.ndarray, angles_deg: np.ndarray, field_index: int) -> tuple[int, np.ndarray, np.ndarray]:
    """
    Worker process for one torque-vs-rotation axis chunk.
    """

    material = nanoparticle_material(material_name, **material_kwargs)
    efield = incident_field(**efield_params)
    dyads = field_propagation(**dyads_params)

    values = []
    for idx in angle_indices:
        angle = angles_deg[idx]
        angles = [0.0, 0.0, 0.0]
        angles[axis_index] = float(angle)
        rot = R.from_euler("xyz", angles, degrees=True)
        rotated_geometry = geometry @ rot.as_matrix().T

        with blas_single_thread_context():
            sim = simulation_from_geometry(geometry=rotated_geometry, step_nm=step_nm, material=material, efield=efield, dyads=dyads)
            T = optical_force(sim=sim, field_index=field_index, return_value="torque")
        values.append(T)

    return axis_index, angle_indices, TORQUE_CONVERSION * np.array(values)

def torque_vs_rotation(geometry: np.ndarray, step_nm: float, material: Any, efield: Any, dyads: Any, angles_deg: np.ndarray, field_index: int, *, 
                        parallel: bool | Literal["auto"] = False, max_workers: int | None = None, material_name: str | None = None, material_kwargs: dict[str, Any] | None = None, 
                        efield_params: dict[str, Any] | None = None, dyads_params: dict[str, Any] | None = None, verbose: bool = False) -> dict[str, np.ndarray]:
    """
    Compute optical torque as a function of angular displacement.

    Parameters
    ----------
    geometry : np.ndarray
        Nanoparticle geometry as dipole positions in nm.

    step_nm : float
        Discretization step in nm.

    material : Any
        pyGDM material object. Used in serial mode.

    efield : Any
        pyGDM incident field object. Used in serial mode.

    dyads : Any
        pyGDM dyadic propagator object. Used in serial mode.

    angles_deg : np.ndarray
        Angular displacements in degrees.

    field_index : int
        Incident-field configuration index.

    parallel : bool | str, optional
        Execution mode. Accepted values are False, True, and "auto".
        Default is False.

    max_workers : int | None, optional
        Maximum number of worker processes. Default is None.

    material_name : str | None, optional
        Material name used to reconstruct the material inside each process.
        Required when parallel execution is used. Default is None.

    material_kwargs : dict[str, Any] | None, optional
        Keyword arguments for nanoparticle_material.
        Required when parallel execution is used. Default is None.

    efield_params : dict[str, Any] | None, optional
        Keyword arguments for incident_field.
        Required when parallel execution is used. Default is None.

    dyads_params : dict[str, Any] | None, optional
        Keyword arguments for field_propagation.
        Required when parallel execution is used. Default is None.

    verbose : bool, optional
        If True, print execution-mode messages. Default is None.

    Returns
    -------
    Torque : dict[str, np.ndarray]
        Dictionary with keys "X", "Y", "Z". Each entry is an array with the
        torque vectors evaluated at the requested angular displacements.
    """

    axes = AXES
    angles_deg = np.asarray(angles_deg)
    n_angles = len(angles_deg)
    n_tasks = len(axes) * n_angles

    use_parallel, resolved_workers = resolve_parallel_execution(parallel=parallel, max_workers=max_workers, n_tasks=n_tasks, verbose=verbose)
    if not use_parallel:
        Torque = {}
        for axis_index, axis_name in enumerate(axes):
            values = []
            for angle in angles_deg:
                angles = [0.0, 0.0, 0.0]
                angles[axis_index] = float(angle)
                rot = R.from_euler("xyz", angles, degrees=True)
                rotated_geometry = geometry @ rot.as_matrix().T
                sim = simulation_from_geometry(geometry=rotated_geometry, step_nm=step_nm, material=material, efield=efield, dyads=dyads)
                T = optical_force(sim=sim, field_index=field_index, return_value="torque")
                values.append(T)

            Torque[axis_name] = TORQUE_CONVERSION * np.array(values)

        return Torque

    required_parallel_args = {"material_name": material_name, "material_kwargs": material_kwargs, "efield_params": efield_params, "dyads_params": dyads_params}
    missing = [name for name, value in required_parallel_args.items() if value is None]
    if missing:
        raise ValueError("Parallel execution requires the following additional arguments: "
            + ", ".join(missing)
            + ". This is needed because each process must reconstruct "
            "material, efield, and dyads independently.")

    Torque = {axis_name: np.empty((n_angles, 3)) for axis_name in axes}
    chunks_per_axis = _automatic_chunks_per_axis(n_values=n_angles, n_axes=len(axes), n_workers=resolved_workers)

    if verbose:
        print(f"Using {chunks_per_axis} chunks per axis "
            f"({chunks_per_axis * len(axes)} total chunks).")

    tasks = []
    for axis_index in range(len(axes)):
        indices = np.arange(n_angles)
        index_chunks = np.array_split(indices, chunks_per_axis)
        for chunk in index_chunks:
            if len(chunk) > 0:
                tasks.append((axis_index, chunk))

    with ProcessPoolExecutor(max_workers=resolved_workers) as executor:
        futures = [
            executor.submit(_torque_vs_rotation_worker, geometry, step_nm, material_name, material_kwargs, efield_params, dyads_params, axis_index, chunk, angles_deg, field_index)
            for axis_index, chunk in tasks]
        for future in as_completed(futures):
            axis_index, angle_indices, torque_values = future.result()
            axis_name = axes[axis_index]
            Torque[axis_name][angle_indices, :] = torque_values

    return Torque

def recoil_force_noise_psd(E_scat: np.ndarray, wavelength_nm: float, Nteta: int, Nphi: int, r_nm: float, axis_index: int) -> float:
    """
    Compute the recoil force-noise PSD S_FF along a mechanical mode.

    The implemented expression is

        S_FF^(mu) = (hbar * omega * eps0 * r^2 / (2c))
                   * int |E_scat(theta, phi)|^2
                         (n_hat · e_mu)^2 dOmega

    Parameters
    ----------
    E_scat : np.ndarray
        Scattered far-field electric field with shape (Nteta * Nphi, 3)
        or (Nteta, Nphi, 3).

    wavelength_nm : float
        Optical wavelength in nm.

    Nteta : int
        Number of theta points.

    Nphi : int
        Number of phi points.

    r_nm : float
        Far-field evaluation radius in nm.

    axis_index : int
        Mechanical axis:
            0 -> x
            1 -> y
            2 -> z

    Returns
    -------
    S_FF : float
        Recoil force-noise PSD along the selected mode.
    """

    wavelength_m = wavelength_nm * NM_TO_M
    r_m = r_nm * NM_TO_M

    omega = 2.0 * np.pi * C / wavelength_m

    theta = np.linspace(0.0, np.pi, Nteta)
    phi = np.linspace(0.0, 2.0 * np.pi, Nphi)
    Theta, Phi = np.meshgrid(theta, phi, indexing="ij")

    n_x = np.sin(Theta) * np.cos(Phi)
    n_y = np.sin(Theta) * np.sin(Phi)
    n_z = np.cos(Theta)
    n_components = [n_x, n_y, n_z]
    n_mu = n_components[axis_index]

    E_scat = np.asarray(E_scat)
    if E_scat.ndim == 2:
        E_scat_abs2 = np.sum(np.abs(E_scat) ** 2, axis=1).reshape(Nteta, Nphi)
    elif E_scat.ndim == 3:
        E_scat_abs2 = np.sum(np.abs(E_scat) ** 2, axis=-1)
    else:
        raise ValueError("E_scat must have shape (Nteta*Nphi, 3) or (Nteta, Nphi, 3).")

    integrand = E_scat_abs2 * n_mu**2 * np.sin(Theta)
    integral_theta_phi = np.trapz(np.trapz(integrand, phi, axis=1), theta, axis=0)
    S_FF = (HBAR * omega * EPS0 * r_m**2 / (2.0 * C)) * integral_theta_phi

    return float(np.real(S_FF))

def trap_frequency(displacements_nm: np.ndarray, force_N: np.ndarray, mass_kg: float) -> tuple[float, float]:
    """
    Compute the harmonic trap frequency from a force-displacement curve.

    The fitted relation is

        F = slope * x + intercept.

    For a stable trap,

        slope < 0
        k_trap = -slope
        Omega = sqrt(k_trap / mass)

    Parameters
    ----------
    displacements_nm : np.ndarray
        Particle displacements in nm.

    force_N : np.ndarray
        Optical force along the same direction in N.

    mass_kg : float
        Particle mass in kg.

    Returns
    -------
    Omega_rad_s : float
        Angular trap frequency in rad/s. Returns np.nan if not restoring.

    k_trap : float
        Trap stiffness in N/m.
    """

    displacements_m = np.asarray(displacements_nm) * NM_TO_M
    force_N = np.asarray(force_N)
    slope_N_m = np.polyfit(displacements_m, force_N, 1)[0]
    k_trap = -slope_N_m
    Omega_rad_s = np.sqrt(k_trap / mass_kg)

    return float(Omega_rad_s), float(k_trap)

def heating_rate(S_FF: float, mass_kg: float, Omega_rad_s: float) -> float:
    """
    Compute the recoil heating rate Gamma_mu.

    Gamma_mu = pi * S_FF / (m * hbar * Omega_mu)

    Parameters
    ----------
    S_FF : float
        Recoil force-noise PSD.

    mass_kg : float
        Particle mass in kg.

    Omega_rad_s : float
        Mechanical angular frequency in rad/s.

    Returns
    -------
    Gamma_mu : float
        Recoil heating rate in phonons/s.
    """

    Gamma_mu = np.pi * S_FF / (mass_kg * HBAR * Omega_rad_s)

    return float(Gamma_mu)

def maxwell_stress_tensor(total_electric_field: np.ndarray, total_magnetic_field: np.ndarray, permittivity: float = EPS0, permeability: float = MU0) -> np.ndarray:
    electric_term = permittivity * np.einsum("...i,...j->...ij", total_electric_field, np.conj(total_electric_field))
    magnetic_term = permeability * np.einsum("...i,...j->...ij", total_magnetic_field, np.conj(total_magnetic_field))
    mixed_term =  permittivity * np.sum(np.abs(total_electric_field)**2, axis=-1) + permeability * np.sum(np.abs(total_magnetic_field)**2, axis=-1)
    return 0.5 * np.real(electric_term + magnetic_term - 0.5 * mixed_term[..., None, None] * np.eye(3))

def spherical_integration_surface(radius: float, n_theta: int, n_phi: int, center: np.ndarray = np.zeros(3)):
    theta = (np.arange(n_theta) + 0.5) * np.pi / n_theta
    phi = np.arange(n_phi) * 2 * np.pi / n_phi
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
    normal_vectors = np.stack([np.sin(theta_grid) * np.cos(phi_grid), np.sin(theta_grid) * np.sin(phi_grid), np.cos(theta_grid)], axis=-1).reshape(-1, 3)
    surface_points = center + radius * normal_vectors
    area_elements = (radius**2 * np.sin(theta_grid) * (np.pi / n_theta) * (2 * np.pi / n_phi)).reshape(-1)
    return surface_points, normal_vectors, area_elements

def force_from_stress_tensor(total_electric_field: np.ndarray, total_magnetic_field: np.ndarray, radius: float, n_theta: int, n_phi: int) -> np.ndarray:
    _, normal_vectors, area_elements = spherical_integration_surface(radius=radius, n_theta=n_theta, n_phi=n_phi)
    stress_tensor = maxwell_stress_tensor(total_electric_field=total_electric_field, total_magnetic_field=total_magnetic_field)
    traction = np.einsum("nij,nj->ni", stress_tensor, normal_vectors)
    return np.sum(traction * area_elements[:, None], axis=0)


def torque_from_stress_tensor(total_electric_field: np.ndarray, total_magnetic_field: np.ndarray, radius: float, n_theta: int, n_phi: int, center: np.ndarray = np.zeros(3)) -> np.ndarray:
    surface_points, normal_vectors, area_elements = spherical_integration_surface(radius=radius, n_theta=n_theta, n_phi=n_phi, center=center)
    stress_tensor = maxwell_stress_tensor(total_electric_field=total_electric_field, total_magnetic_field=total_magnetic_field)
    traction = np.einsum("nij,nj->ni", stress_tensor, normal_vectors)
    position_vectors = surface_points - center
    torque_density = np.cross(position_vectors, traction)
    return np.sum(torque_density * area_elements[:, None], axis=0)