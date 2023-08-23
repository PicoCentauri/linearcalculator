import logging
import os
from typing import List, Union

import ase.io
import equistore
import matplotlib.pyplot as plt
import numpy as np
from rascaline import (
    AtomicComposition,
    LodeSphericalExpansion,
    SoapPowerSpectrum,
    SphericalExpansion,
)
from rascaline.utils import PowerSpectrum

from .radial_basis import KspaceRadialBasis


logger = logging.getLogger(__name__)


def setup_dataset(
    filenames: List[List[ase.Atoms]],
    labels: Union[List[str], str],
    cell_length: float = -1,
):
    """
    Read and process `ase.Atoms` from files, filter by label and set cell length.

    Read the given list of ASE Atoms objects from the provided filenames. If filenames
    is a single object instead of a list, it will be converted to a list.

    If label is provided, return only the atoms objects with info["label"] == label. If
    label is "all", return all the atoms objects.

    Set the cell length of all atoms objects to `cell_length` and enable periodic
    boundary conditions.

    Parameters:
    -----------
    filenames : list of `ase.Atoms` objects
        List of ASE Atoms objects to read from.
    labels : list of `str` or `str`
        The labels to match with info["label"] to select specific Atoms objects. If
        "all", returns all the Atoms objects.
    cell_length : float
        The length of the cell for all Atoms objects.

    Returns:
    --------
    frames: list of `ase.Atoms` objects
        The list of ASE Atoms objects that match the given label (or all Atoms objects
        if label="all"). The cell length and periodic boundary conditions are set for
        all returned Atoms objects.
    """
    if type(filenames) not in [list, tuple]:
        filenames = [filenames]

    frames = []
    for filename in filenames:
        frames += ase.io.read(filename, ":")

    if type(labels) not in [list, tuple]:
        labels = [labels.lower()]
    else:
        labels = [label.lower() for label in labels]

    if "all" not in labels:
        frames_new = []
        for atoms in frames:
            label = atoms.info["label"].lower()
            if label in labels:
                frames_new.append(atoms)
        frames = frames_new

    if cell_length != -1:
        for frame in frames:
            frame.set_cell(cell_length * np.ones(3))
            frame.pbc = True

    return frames


def compute_descriptors(
    frames: List[ase.Atoms],
    config: dict,
    potential_exponent: int = 0,
    gradients: List[str] = ["positions"],
    ps_fname: str = "descriptor_ps.npz",
):
    """Compute atomic the composition and the power spectrum descriptor.

    Parameters
    ----------
    frames : List[ase.Atoms]
        A list of atomic systems represented as ASE Atoms objects.
    config : dict
        A dictionary containing the configuration parameters for descriptor computation.
    potential_exponent : int
        The potential exponent to taken for the LODE descriptor. If 0 only a classical
        short-range SOAP model is calculated.
    ps_fname : str
        file name for saving the power spectrum.

    Returns
    -------
    co : equistore.TensorMap
        composition descriptor (co) representing the atomic composition of the
        structures.
    ps : equistore.TensorMap
        power spectrum descriptor (ps) representing the rotationally invariant power
        spectrum.
    """

    compute_args = {"systems": frames, "gradients": gradients}

    if config["recalc_descriptors"]:
        logger.info(f"Compute descriptors for potential_exponent={potential_exponent}.")

        # Compute spherical expanions and power spectrum
        if potential_exponent != 0:
            lr_hypers = config["lr_hypers"].copy()
            radial_basis = list(lr_hypers["radial_basis"].keys())[0]

            # Use TabulatedRadialIntegral if the basis is not Gto
            if radial_basis != "Gto":
                if radial_basis in ["gto", "gto_primitive", "gto_normalized"]:
                    orthonormalization_radius = 5 * lr_hypers["cutoff"]
                else:
                    orthonormalization_radius = lr_hypers["cutoff"]

                rad = KspaceRadialBasis(
                    radial_basis,
                    max_radial=lr_hypers["max_radial"],
                    max_angular=lr_hypers["max_angular"],
                    projection_radius=lr_hypers["cutoff"],
                    orthonormalization_radius=orthonormalization_radius,
                )

                k_cut = 1.2 * np.pi / lr_hypers["atomic_gaussian_width"]
                spline_points = rad.spline_points(
                    cutoff_radius=k_cut, requested_accuracy=1e-8
                )

                lr_hypers["radial_basis"] = {
                    "TabulatedRadialIntegral": {
                        "points": spline_points,
                        "center_contribution": [
                            0.0 for _ in range(lr_hypers["max_radial"])
                        ],
                    }
                }

            sr_calculator = SphericalExpansion(**config["sr_hypers"])
            lr_calculator = LodeSphericalExpansion(
                potential_exponent=potential_exponent, **lr_hypers
            )
            calculator = PowerSpectrum(sr_calculator, lr_calculator)
            ts = calculator.compute(**compute_args, fill_species_neighbor=True)
        else:
            calculator = SoapPowerSpectrum(**config["soap_hypers"])
            ts = calculator.compute(**compute_args)

        ts = ts.keys_to_samples("species_center")

        if potential_exponent == 0:
            ts = ts.keys_to_properties(["species_neighbor_1", "species_neighbor_2"])

        ps = equistore.sum_over_samples(ts, ["center", "species_center"])
        del ts

        equistore.save(os.path.join(config["output"], ps_fname), ps)
    else:
        logger.info("Load descriptors from file")
        ps = equistore.load(os.path.join(config["output"], ps_fname))

    # Compute structure calculator
    co_calculator = AtomicComposition(per_structure=True)
    co_descriptor = co_calculator.compute(**compute_args)
    co = co_descriptor.keys_to_properties("species_center")

    return co, ps
