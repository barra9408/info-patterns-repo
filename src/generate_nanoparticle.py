import numpy as np
from pyGDM2 import structures

def nanoparticle_geometry(particle_type, step_nm, center=True, **kwargs):
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

    **kwargs
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