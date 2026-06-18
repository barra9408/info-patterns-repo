import numpy as np

def info_patterns_from_scattered_field(dE: np.ndarray, delta_mu: float, wavelength_nm: float, Nteta: int, Nphi: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the information pattern from a scattered-field derivative.

    This function computes

        I_mu(theta, phi) = 2 / (hbar * omega * c) * |dE/dmu|^2

    using a central finite-difference approximation:

        dE/dmu ≈ (E_plus - E_minus) / (2 * delta_mu)

    It returns both the total information pattern and the Cartesian
    component-wise contributions.

    Parameters
    ----------
    dE : np.ndarray
        Difference between scattered electric fields:
        E_plus[2] - E_minus[2].
        Expected shape: (Nteta * Nphi, 3).

    delta_mu : float
        Mechanical perturbation amplitude.
        For COM motion, this is the displacement amplitude.
        For libration, this is the rotation amplitude in radians.

    wavelength_nm : float
        Optical wavelength in nm.

    Nteta : int
        Number of sampling points along the polar angle theta.

    Nphi : int
        Number of sampling points along the azimuthal angle phi.

    Returns
    -------
    I_total : np.ndarray
        Total information pattern with shape (Nteta, Nphi).

    I_x : np.ndarray
        Contribution from the x-component of the scattered field.

    I_y : np.ndarray
        Contribution from the y-component of the scattered field.

    I_z : np.ndarray
        Contribution from the z-component of the scattered field.
    """

    hbar = 1.054571817e-34
    c = 299792458.0

    wavelength_m = wavelength_nm * 1e-9
    omega = 2 * np.pi * c / wavelength_m
    prefactor = 2 / (hbar * omega * c)

    dE_dmu = dE / (2 * delta_mu)

    I_x = prefactor * np.abs(dE_dmu[:, 0])**2
    I_y = prefactor * np.abs(dE_dmu[:, 1])**2
    I_z = prefactor * np.abs(dE_dmu[:, 2])**2

    I_total = I_x + I_y + I_z

    I_total = I_total.reshape(Nteta, Nphi)
    I_x = I_x.reshape(Nteta, Nphi)
    I_y = I_y.reshape(Nteta, Nphi)
    I_z = I_z.reshape(Nteta, Nphi)

    return I_total, I_x, I_y, I_z