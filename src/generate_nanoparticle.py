from typing import Any
import numpy as np

from pyGDM2 import structures
from pyGDM2 import materials

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

