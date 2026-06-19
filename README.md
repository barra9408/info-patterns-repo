# Information Patterns Repository

This repository provides a modular workflow to simulate optical information patterns of mechanical degrees of freedom in nanoparticles. It uses `pyGDM2` to solve the light–matter interaction problem and custom routines to convert changes in the scattered far field into angular information patterns, detection efficiencies, optical forces, optical torques, and recoil-heating estimates.

The main goal is to keep the physical pipeline readable and reusable:

1. Build the nanoparticle geometry.
2. Assign material, incident field, and electromagnetic propagator.
3. Perturb the particle according to a mechanical degree of freedom.
4. Compute the corresponding scattered far-field variation.
5. Transform that variation into an angular information pattern.
6. Analyze visualization and specific detection quantities (efficiencies, recoil heating, torque, stiffness, force, etc.).

---

## Repository structure

```text
info-patterns-repo/
│
├── requirements.txt
│
├── src/
│   ├── __init__.py
│   ├── constants.py
│   ├── parameters.py
│   ├── generate_nanoparticle.py
│   ├── light_matter_interaction_simulation.py
│   ├── information_patterns_simulation.py
│   ├── measurement_tools.py
│   ├── plots.py
│   └── specific_incident_fields.py
│
└── notebooks/
    ├── 01_com_case.ipynb
    ├── 02_librational_case.ipynb
    ├── 03_maxdetection_efficiency.ipynb
    ├── 04_arbitrary_incident_field.ipynb
    ├── 05_force_and_torque.ipynb
    ├── 06_filling_factor.ipynb
    ├── 07_recoil_heating.ipynb
    └── 08_complete_pipeline.ipynb
```

The `src/` folder contains reusable simulation, analysis, and plotting routines. The `notebooks/` folder contains concrete examples that reproduce specific physical cases.

---

## Installation

Create and activate a virtual environment from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Then launch the notebooks with:

```bash
jupyter lab
```

The main dependencies are:

```text
numpy
pandas
matplotlib
scipy
pyGDM2
jupyter
ipykernel
```

---

## Physical idea

The repository estimates how much information about a mechanical coordinate $\mu$ is encoded in the optical field scattered by a nanoparticle.

The coordinate $\mu$ can represent, for example:

- Center-of-mass displacement along $x$, $y$ and $z$.
- Libration around a Cartesian axis.
- Any other mechanical or geometrical perturbation that changes the scattered field.

Numerically, the particle is evaluated at two nearby configurations:

$$
\mu_+ = \mu_0 + \delta\mu,
\qquad
\mu_- = \mu_0 - \delta\mu.
$$

For each configuration, the scattered far field

$$
\mathbf{E}_{\mathrm{scat}}^{+}(\theta,\phi),
\qquad
\mathbf{E}_{\mathrm{scat}}^{-}(\theta,\phi).
$$

is obtained. Afterwards, the field derivative is computed with a centered finite difference

$$
\frac{\partial \mathbf{E}_{\mathrm{scat}}}{\partial \mu}
\approx
\frac{\mathbf{E}_{\mathrm{scat}}^{+} -\mathbf{E}_{\mathrm{scat}}^{-}}{2\delta\mu},
$$

so that we obtain the angular information patterns by

$$
I_\mu(\theta,\phi) =
\frac{2}{\hbar \omega c}\left|\frac{\partial \mathbf{E}_{\mathrm{scat}}}{\partial \mu}\right|^2.
$$

In the code, this is implemented in:

```python
info_patterns_from_scattered_field(...)
```

from:

```text
src/information_patterns_simulation.py
```

The function returns:

```python
I_total, I_x, I_y, I_z
```

where `I_total` is the full information pattern and `I_x`, `I_y`, `I_z` are the Cartesian electric-field component contributions.

---

## Methodological pipeline

<p align="center">
  <img src="figures/repo_process.svg" alt="Repository conceptual workflow" width="700">
</p>

This modular structure allows the geometry, material, incident field, propagator, mechanical mode, and detection metric to be changed independently.

---

## Main modules

### `constants.py`

Defines physical constants, unit conversions, axis conventions, and material densities.

Typical contents include:

```python
HBAR
C
EPS0
NM_TO_M
NM2_TO_M2
NM3_TO_M3
AXES
AXIS_INDICES
MATERIAL_DENSITIES_KG_M3
```

Use this module whenever a calculation requires consistent SI conversions or shared physical constants.

---

### `parameters.py`

Stores default simulation and plotting parameters used across the notebooks.

This avoids repeatedly defining values such as:

```python
DEFAULT_STEP_NM
DEFAULT_RADIUS_NM
DEFAULT_WAVELENGTH_NM
DEFAULT_NTETA
DEFAULT_NPHI
DEFAULT_SAVE_TYPE
DEFAULT_RESULTS_DIR
```

It is the preferred place to modify global defaults for a new simulation campaign.

---

### `generate_nanoparticle.py`

Contains routines to generate nanoparticle geometries and assign materials.

Main responsibilities:

- generate pyGDM-compatible geometries;
- center geometries when needed;
- assign materials such as `sio2`, `Au`, `Ag`, `Al`, or `Si`;
- estimate nanoparticle mass from dipole geometry and material density.

Typical functions:

```python
nanoparticle_geometry(...)
nanoparticle_material(...)
nanoparticle_mass(...)
```

---

### `light_matter_interaction_simulation.py`

Builds and solves the electromagnetic scattering problem.

Main responsibilities:

- define the incident field;
- define the propagator;
- build a `pyGDM2` simulation object;
- solve the scattering problem;
- extract the scattered far field.

Typical functions:

```python
incident_field(...)
field_propagation(...)
simulation_from_geometry(...)
scattered_farfield_from_simulation(...)
scattered_farfield_from_geometry(...)
```

This module is the bridge between the physical geometry and the optical response computed by `pyGDM2`.

---

### `information_patterns_simulation.py`

Contains the core information-pattern logic.

Main responsibilities:

- compute scattered-field changes for center-of-mass motion;
- compute scattered-field changes for librational motion;
- calculate information patterns from field derivatives;
- handle rotations and mechanical perturbations.

Typical functions:

```python
com_scattered_farfield(...)
librational_scattered_farfield(...)
info_patterns_from_scattered_field(...)
simulate_rotated_geometry(...)
dE_for_rotation(...)
```

This is the main module for turning electromagnetic simulations into mechanical-information observables.

---

### `measurement_tools.py`

Implements measurement and diagnostic quantities derived from the simulated fields.

Main responsibilities:

- compute maximum detection efficiency over angular regions;
- analyze Gaussian local oscillator detection;
- compute optical forces and torques;
- estimate recoil force-noise PSD and recoil-heating rates;
- evaluate filling factors for objective illumination.

Typical functions:

```python
max_detection_efficiency(...)
gaussian_local_oscillator(...)
run_gaussian_local_oscillator(...)
filling_factor(...)
optical_force(...)
force_vs_displacement(...)
torque_vs_rotation(...)
recoil_force_noise_psd(...)
heating_rate(...)
```

---

### `plots.py`

Contains visualization utilities for information patterns, force curves, torque curves, and related diagnostics.

Typical functions:

```python
plot2d_info_patterns(...)
plot3d_info_patterns(...)
add_mini_axis(...)
plot_force_displacement(...)
plot_torque_rotation(...)
```

Most plotting functions support saving figures to a `results/` directory through arguments such as:

```python
savefig=True
save_type="pdf"   # or "png"
filename="my_figure"
results_dir="results"
```

---

### `specific_incident_fields.py`

Defines custom incident-field models beyond the built-in `pyGDM2` options.

This is useful when the simulation requires a structured beam or a particular focusing model, such as a Richards–Wolf-type focused Gaussian field.

---

## Notebook guide

### `01_com_case.ipynb`

Computes information patterns for center-of-mass displacement modes.

Use this notebook to study:

- COM motion along \(x\), \(y\), and \(z\);
- corresponding far-field differences;
- 2D and 3D information-pattern visualizations.

---

### `02_librational_case.ipynb`

Computes information patterns for librational degrees of freedom.

Use this notebook to study how small rotations of a non-spherical particle modify the scattered far field.

---

### `03_maxdetection_efficiency.ipynb`

Evaluates how much of the total information can be collected by a detector with restricted angular aperture.

This is useful for estimating detection performance as a function of numerical aperture, hemisphere, or angular collection region.

---

### `04_arbitrary_incident_field.ipynb`

Demonstrates how to run the pipeline with a custom incident field.

Use this notebook when replacing the default plane wave or Gaussian beam with a user-defined field profile.

---

### `05_force_and_torque.ipynb`

Computes optical force and torque curves from the simulated electromagnetic response.

Use this notebook to analyze mechanical stability, force slopes, and torque response around equilibrium.

---

### `06_filling_factor.ipynb`

Evaluates objective filling factors and related focusing parameters.

This helps connect beam waist, focal length, numerical aperture, and medium refractive index.

---

### `07_recoil_heating.ipynb`

Estimates recoil force-noise PSD and recoil-heating rates from information-pattern quantities.

Use this notebook to connect optical information extraction with mechanical heating.

---

### `08_complete_pipeline.ipynb`

Runs the full workflow in a single notebook.

Use this notebook as the best starting point for understanding how all modules interact.

---

## Basic usage example

A typical workflow inside a notebook is:

```python
from src.generate_nanoparticle import nanoparticle_geometry, nanoparticle_material
from src.light_matter_interaction_simulation import (
    incident_field,
    field_propagation,
    scattered_farfield_from_geometry,
)
from src.information_patterns_simulation import (
    com_scattered_farfield,
    info_patterns_from_scattered_field,
)
from src.plots import plot2d_info_patterns, plot3d_info_patterns
```

Then define the geometry, material, field, and propagator:

```python
geometry = nanoparticle_geometry(
    particle_type="sphere",
    step_nm=10,
    R=5,
    mesh="hex",
    center=True,
)

material = nanoparticle_material("sio2")
efield = incident_field("gaussian", wavelength=1550, NA=0.1)
dyads = field_propagation("quasistatic123", n1=1.0, n2=1.0)
```

After solving the scattering problem for perturbed configurations, the information patterns can be computed and plotted:

```python
I_total, I_x, I_y, I_z = info_patterns_from_scattered_field(
    dE=dE_com_x,
    delta_mu=disp_nm,
    wavelength_nm=1550,
)

plot2d_info_patterns(
    Nteta=Nteta,
    Nphi=Nphi,
    info_patterns=[I_x, I_y, I_z],
    labels=["x", "y", "z"],
)

plot3d_info_patterns(
    Nteta=Nteta,
    Nphi=Nphi,
    info_patterns=[I_x, I_y, I_z],
    labels=["x", "y", "z"],
)
```

The exact arguments may vary depending on the notebook and field model being used.

---

## Recommended workflow for new simulations

For a new physical case, follow this order:

1. Start from `08_complete_pipeline.ipynb`.
2. Choose the nanoparticle geometry in `generate_nanoparticle.py`.
3. Set the material and optical parameters.
4. Select or define the incident field.
5. Choose the mechanical degree of freedom: COM displacement, libration, or another perturbation.
6. Compute the perturbed scattered fields.
7. Compute `I_total`, `I_x`, `I_y`, and `I_z`.
8. Visualize the patterns using `plots.py`.
9. Evaluate detection efficiency, optical forces, torque, or recoil heating if needed.

---

## Notes on units

Most geometric inputs are handled in nanometers because `pyGDM2` commonly works with nanometric particle geometries. However, physical observables such as mass, force, torque, and heating rates require SI units.

For that reason, unit conversions are centralized in `src/constants.py`. When adding new physical quantities, use the existing conversion constants instead of hard-coding powers of ten in notebooks.

---

## Extending the repository

To add a new case cleanly:

- add general-purpose logic to `src/`;
- keep notebook cells focused on parameter choices, execution, and visualization;
- avoid duplicating long functions inside notebooks;
- add new default parameters to `parameters.py` only when they are reused across multiple notebooks;
- save generated figures and tables in a dedicated `results/` directory.

This keeps the repository easier to debug, reproduce, and extend.

---

## Suggested citation note

If this repository is used for a report, thesis, or article, describe the method as a finite-difference far-field information-pattern calculation based on pyGDM2 electromagnetic scattering simulations.

