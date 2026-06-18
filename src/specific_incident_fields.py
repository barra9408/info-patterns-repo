from typing import Any, Literal, Sequence
from scipy import special
import numpy as np

def filling_factor(w_0: float, f: float, NA: float, n_medium: float) -> float:
    """
    Compute the objective filling factor.

    gamma = w0 / (f sin(theta0))

    Parameters
    ----------
    w_0 : float
        Beam waist at the objective pupil. Same units as f.

    f : float
        Focal length. Same units as w_0.

    NA : float
        Numerical aperture.

    n_medium : float, optional
        Refractive index of the surrounding medium.

    Returns
    -------
    gamma : float
        Objective filling factor.
    """

    theta0 = np.arcsin(NA / n_medium)
    gamma = w_0 / (f * np.sin(theta0))

    return gamma

def richards_wolf_gaussian_efield(pos: np.ndarray, env_dict: dict[str, Any], wavelength: float, n_medium: float, NA: float, Ntheta: int, Nphi: int, gamma: float | None, f: float | None, 
                                  w_0: float | None, E0: float, polarization_state: Sequence[complex], xSpot: float, ySpot: float, zSpot: float, 
                                  returnField: Literal["E", "B", "H"]) -> np.ndarray:
    """
    Richards-Wolf focused Gaussian incident electric field for pyGDM2.

    This function is meant to be used as a `field_generator` inside
    `pyGDM2.fields.efield`.

    pyGDM2 supplies `pos`, `env_dict`, and `wavelength` automatically.
    The remaining parameters are passed through `kwargs`.

    Parameters
    ----------
    pos : np.ndarray
        Dipole positions in nm with shape approximately (Npoints, 3).

    env_dict : dict[str, Any]
        Environment dictionary provided by pyGDM2.
        Not used here, kept for API compatibility.

    wavelength : float
        Wavelength in nm.

    n_medium : float
        Refractive index of the surrounding medium.

    NA : float
        Numerical aperture of the focusing objective.

    Ntheta : int
        Number of theta integration points.

    Nphi : int
        Number of phi integration points.

    gamma : float | None
        Objective filling factor. If None, it is computed from w_0, f, and NA.

    f : float | None
        Focal length in nm. Required only if gamma is None.

    w_0 : float | None
        Beam waist at the objective pupil in nm. Required only if gamma is None.

    E0 : float
        Overall field amplitude.

    polarization_state : Sequence[complex]
        Input polarization state. Only the first two entries are used:
            (Ex, Ey, 0, 0)

    xSpot, ySpot, zSpot : float
        Focus position in nm. The field is evaluated at pos - spot.

    returnField : Literal["E", "B", "H"]
        Requested field type from pyGDM2. Only "E" is provided.

    Returns
    -------
    E : np.ndarray
        Incident electric field components [Ex, Ey, Ez].
    """

    # We do not provide B/H field
    if returnField != "E":
        return np.asfortranarray(np.zeros((len(pos), 3), dtype=np.complex64))

    # Positions relative to the focus
    pos = np.asarray(pos, dtype=float)
    x = pos[:, 0] - xSpot
    y = pos[:, 1] - ySpot
    z = pos[:, 2] - zSpot

    # Wave number in pyGDM units [1/nm]
    k = 2 * np.pi * n_medium / wavelength

    # Maximum focusing angle
    theta0 = np.arcsin(NA / n_medium)

    # Compute filling factor if not provided
    if gamma is None:
        gamma = filling_factor(w_0=w_0, f=f, NA=NA, n_medium=n_medium)

    # Integration grids
    theta = np.linspace(0.0, theta0, Ntheta)
    phi = np.linspace(0.0, 2 * np.pi, Nphi, endpoint=False)

    dtheta = theta[1] - theta[0]
    dphi = phi[1] - phi[0]

    # Input polarization in the pupil plane
    Ein_x = polarization_state[0]
    Ein_y = polarization_state[1]
    e_in = np.array([Ein_x, Ein_y, 0.0], dtype=float)

    # Output field
    Ex = np.zeros(len(pos), dtype=complex)
    Ey = np.zeros(len(pos), dtype=complex)
    Ez = np.zeros(len(pos), dtype=complex)

    # Richards-Wolf global prefactor
    prefactor = 1j * k * E0 / (2 * np.pi)

    for th in theta:
        sin_th = np.sin(th)
        cos_th = np.cos(th)

        # Gaussian pupil profile with filling factor
        pupil = np.sqrt(cos_th) * np.exp(-(sin_th / (gamma * np.sin(theta0)))**2)

        for ph in phi:
            cos_ph = np.cos(ph)
            sin_ph = np.sin(ph)

            # Local basis at the reference sphere
            e_rho = np.array([cos_ph, sin_ph, 0.0])
            e_phi = np.array([-sin_ph, cos_ph, 0.0])
            e_theta = np.array([cos_th * cos_ph, cos_th * sin_ph, -sin_th])

            # Richards-Wolf polarization mapping
            e_out = np.dot(e_in, e_rho) * e_theta + np.dot(e_in, e_phi) * e_phi

            # Plane-wave vector components
            kx = k * sin_th * cos_ph
            ky = k * sin_th * sin_ph
            kz = k * cos_th

            phase = np.exp(1j * (kx * x + ky * y + kz * z))

            # Solid-angle integration weight
            weight = prefactor * pupil * sin_th * dtheta * dphi

            Ex += weight * e_out[0] * phase
            Ey += weight * e_out[1] * phase
            Ez += weight * e_out[2] * phase

    E = np.transpose([Ex, Ey, Ez])

    return np.asfortranarray(E).astype(np.complex64)

from scipy import special

def parabolic_mirror_efield(pos: np.ndarray, env_dict: dict[str, Any], wavelength: float, n_medium: float, f: float, alpha_0: float, alpha_1: float, w_0: float, N: float, 
                            theta_size: int, xSpot: float, ySpot: float, zSpot: float, returnField: Literal["E", "B", "H"]) -> np.ndarray:
    """
    Parabolic-mirror incident electric field for pyGDM2
    (Lieb & Meixner, Opt. Express 2001), based on Eqs. (2)–(3)
    for a linearly x-polarized focused field.

    This function is meant to be used as a `field_generator` inside
    `pyGDM2.fields.efield`. pyGDM2 supplies `pos`, `env_dict`, and
    `wavelength` automatically. The remaining parameters are passed
    through `kwargs`.

    Parameters
    ----------
    pos : np.ndarray
        Dipole positions in nm with shape approximately (Npoints, 3).

    env_dict : dict[str, Any]
        Environment dictionary provided by pyGDM2.
        Not used here, kept for API compatibility.

    wavelength : float
        Wavelength in nm.

    n_medium : float
        Refractive index of the medium around the focus.

    f : float
        Parabolic mirror focal length in nm.

    alpha_0 : float
        Minimum acceptance angle in radians. Use > 0 to model
        a central obscuration.

    alpha_1 : float
        Maximum acceptance/opening angle of the mirror in radians.

    w_0 : float
        Waist radius of the illuminating Gaussian beam on the mirror
        pupil in nm.

    N : float
        Overall amplitude/normalization factor of the incident beam
        profile l0(theta).

    theta_size : int
        Number of samples for the theta integration.

    xSpot, ySpot, zSpot : float
        Focus position in nm. The field is evaluated at pos - spot.

    returnField : Literal["E", "B", "H"]
        Requested field type from pyGDM2. Only "E" is provided.

    Returns
    -------
    E : np.ndarray
        Incident electric field components [Ex, Ey, Ez] evaluated at
        each dipole position. Returned as complex64 and Fortran-ordered.
    """
    # We do not provide B/H field
    if returnField !="E":
        return np.asfortranarray(np.zeros((len(pos), 3), dtype=np.complex64))
    
    # Initial position of the particle
    pos = np.asarray(pos, dtype=float)
    x_p = pos[:, 0] - xSpot
    y_p = pos[:, 1] - ySpot
    z_p = pos[:, 2] - zSpot

    k = 2 * np.pi * n_medium / wavelength # [1/nm] wavelength is in nm in pyGDM2

    # Integration variable theta = theta_m
    theta = np.linspace(alpha_0, alpha_1, theta_size) # [rad]
    r = 2 * f * np.tan(theta / 2) # [nm]
    
    l0_theta = N * np.exp(-(r**2) / (w_0**2))
    apod = (2 * np.sin(theta) / (1 + np.cos(theta))) # Mirror apodization factor: 2 sinθ / (1+cosθ)

    # Spherical coords of observation points P (each dipole position)
    r_p = np.sqrt(x_p**2 + y_p**2 + z_p**2) # [nm]
    phi_p = np.arctan2(y_p, x_p)

    # handle r_p = 0 safely
    safe = np.where(r_p > 0, r_p, 1.0)
    cos_theta_p = z_p / safe
    cos_theta_p = np.clip(cos_theta_p, -1.0, 1.0)
    theta_p = np.arccos(cos_theta_p)
    sin_theta_p = np.sin(theta_p)

    # Broadcast over (theta_size, Npoints)
    ct = np.cos(theta)[:, None]
    st = np.sin(theta)[:, None]

    phase = np.exp(-1j * k * (r_p[None, :]) * ct * (cos_theta_p[None, :]))
    argument = k * (r_p[None, :]) * st * (sin_theta_p[None, :])

    # Integrands (Eq. 3), with J0/J1/J2
    integrand_I0 = (l0_theta[:, None] * apod[:, None] * (1 + np.cos(theta))[:, None] * special.jv(0, argument) * phase)
    integrand_I1 = (l0_theta[:, None] * apod[:, None] * (np.sin(theta))[:, None] * special.jv(1, argument) * phase)
    integrand_I2 = (l0_theta[:, None] * apod[:, None] * (1 - np.cos(theta))[:, None] * special.jv(2, argument) * phase)

    I_0l = np.trapz(integrand_I0, theta, axis=0)
    I_1l = np.trapz(integrand_I1, theta, axis=0)
    I_2l = np.trapz(integrand_I2, theta, axis=0)

    # Electric field components
    Ex = ((1j * k * f) / 2) * (I_0l + I_2l * np.cos(2 * phi_p))
    Ey = ((1j * k * f) / 2) * (I_2l * np.sin(2 * phi_p))
    Ez = -k * f * I_1l * np.cos(phi_p)

    return np.asfortranarray(np.transpose([Ex, Ey, Ez])).astype(np.complex64)