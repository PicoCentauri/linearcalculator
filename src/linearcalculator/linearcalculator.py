import logging
import os
import warnings

import ase.io
import equistore
import numpy as np
from equisolve.numpy.models import Ridge
from equisolve.utils.convert import ase_to_tensormap
from equistore import Labels
from numpy.linalg import LinAlgError
from rascaline import AtomicComposition, LodeSphericalExpansion, SphericalExpansion
from sklearn.metrics import mean_squared_error
from sklearn.utils import Bunch
from tqdm.auto import tqdm

from .utils import PARAMETER_KEYS_DICT, compute_power_spectrum


logger = logging.getLogger(__name__)


def setup_dataset(filename, label):
    frames = ase.io.read(filename, ":")

    if label.lower() != "all":
        frames = [f for f in frames if f.info["label"].lower() == label.lower()]

    L = 50

    for frame in frames:
        frame.set_cell(L * np.ones(3))
        frame.pbc = True

    return frames


def compute_linear_models(config):
    frames = setup_dataset(config["dataset"], config["label"])
    compute_args = {"systems": frames, "gradients": ["positions"]}

    # Compute descriptor
    fname_sr_descriptor = os.path.join(config["output"], "sr_descriptor.npz")
    fname_lr_descriptor = os.path.join(config["output"], "lr_descriptor.npz")

    if config["recalc_descriptors"]:
        logger.info("Compute short-range descriptor")
        sr_calculator = SphericalExpansion(**config["sr_hypers"])
        sr_descriptor = sr_calculator.compute(**compute_args)
        equistore.io.save(fname_sr_descriptor, sr_descriptor)

        if config["lr_hypers"]["potential_exponent"] != 0:
            logger.info("Compute long-range descriptor")
            lr_calculator = LodeSphericalExpansion(**config["lr_hypers"])
            lr_descriptor = lr_calculator.compute(**compute_args)
            equistore.io.save(fname_lr_descriptor, lr_descriptor)
    else:
        logger.info("Load short-range descriptor")
        sr_descriptor = equistore.io.load(fname_sr_descriptor)

        if config["lr_hypers"]["potential_exponent"] != 0:
            logger.info("Load long-range descriptor")
            lr_descriptor = equistore.io.load(fname_lr_descriptor)

    # Compute powerspectrum
    fname_ps = "power_spectrum.npz"
    if config["recalc_power_spectrum"]:
        logger.info("Compute power spectrum")
        if config["lr_hypers"]["potential_exponent"] == 0:
            ts = compute_power_spectrum(sr_descriptor)
            del sr_descriptor
        else:
            ts = compute_power_spectrum(sr_descriptor, lr_descriptor)
            del lr_descriptor
            del sr_descriptor

        ts = ts.keys_to_samples(["species_center"])
        ts = ts.keys_to_properties(
            ["species_neighbor_1", "species_neighbor_2", "spherical_harmonics_l"]
        )

        # Sum over all center species per structure
        ps = equistore.sum_over_samples(ts, ["center", "species_center"])
        del ts

        equistore.io.save(fname_ps, ps)
    else:
        logger.info("Load power spectrum")
        ps = equistore.io.load(fname_ps)

    # Compute structure calculator
    descriptor_co = AtomicComposition(per_structure=True).compute(**compute_args)
    co = descriptor_co.keys_to_properties(["species_center"])

    # Setup training curve
    results = Bunch()
    for r_cut in config["training_cutoffs"]:
        idx_train = [i for i, f in enumerate(frames) if f.info["distance"] < r_cut]
        idx_test = [i for i, f in enumerate(frames) if f.info["distance"] >= r_cut]

        if len(idx_train) == 0:
            raise ValueError(f"No training samples for r_cut={r_cut}!")

        results[r_cut] = Bunch(idx_train=idx_train, idx_test=idx_test)

    labels = Labels(["structure"], np.array([[0]]))

    # Set composition calculator to machine precision
    alpha_co = equistore.slice(equistore.ones_like(co), axis="samples", labels=labels)
    alpha_co *= np.finfo(alpha_co[0].values[0, 0]).eps

    alpha_ps = equistore.slice(equistore.ones_like(ps), axis="samples", labels=labels)

    # Setup variables for model paramaters
    y = ase_to_tensormap(frames, energy="energy", forces="forces")
    X = equistore.join([co, ps], axis="properties")

    monomer_energies = np.array([f.info["energyA"] + f.info["energyB"] for f in frames])
    alpha_values = np.logspace(-12, 3, 20)

    # Fit the models
    for realization in tqdm(results.values(), desc="fit models"):
        # Select samples for current run
        samples_train = Labels(
            ["structure"], np.reshape(realization.idx_train, (-1, 1))
        )
        samples_test = Labels(["structure"], np.reshape(realization.idx_test, (-1, 1)))

        split_args = {
            "axis": "samples",
            "grouped_labels": [samples_train, samples_test],
        }
        X_train, X_test = equistore.split(X, **split_args)
        y_train, y_test = equistore.split(y, **split_args)

        # Only used for energy error
        y_train_red = y_train[0].values.flatten()
        y_test_red = y_test[0].values.flatten()

        y_train_red -= monomer_energies[realization.idx_train]
        y_test_red -= monomer_energies[realization.idx_test]

        # Forces
        f_train_all = y_train[0].gradient("positions").data
        f_test_all = y_test[0].gradient("positions").data

        # Select mol_ids for predicting force per molecule
        mol_idx_train = []
        mol_idx_test = []
        for i, f in enumerate(frames):
            idx = [f.info["indexB"], len(f) - f.info["indexB"]]

            if i in realization.idx_train:
                mol_idx_train += idx
            elif i in realization.idx_test:
                mol_idx_test += idx

        mol_idx_train_cumsum = np.cumsum(mol_idx_train)
        mol_idx_test_cumsum = np.cumsum(mol_idx_test)

        f_train_mol = [
            np.sum(m, axis=0) for m in np.split(f_train_all, mol_idx_train_cumsum)
        ]
        f_test_mol = [
            np.sum(m, axis=0) for m in np.split(f_test_all, mol_idx_test_cumsum)
        ]

        f_train_mol = np.array(f_train_mol)
        f_test_mol = np.array(f_test_mol)

        sigma_energy = np.std(y_train_red)
        sigma_force_all = np.std(f_train_all)
        sigma_force_mol = np.std(f_train_mol)

        for i, (key, parameter_keys) in enumerate(PARAMETER_KEYS_DICT.items()):
            l_rmse_f_train = np.nan * np.ones(len(alpha_values))
            l_rmse_f_test = np.nan * np.ones(len(alpha_values))

            l_rmse_e_train = np.nan * np.ones(len(alpha_values))
            l_rmse_e_test = np.nan * np.ones(len(alpha_values))

            l_clf = len(alpha_values) * [None]

            for i_alpha, alpha_value in enumerate(alpha_values):
                clf = Ridge(parameter_keys=parameter_keys)

                alpha_ps[0].values[:] = alpha_value
                alpha = equistore.join([alpha_co, alpha_ps], axis="properties")

                try:
                    with warnings.catch_warnings(record=True) as warns:
                        clf.fit(X_train, y_train, alpha=alpha)
                        for w in warns:
                           logger.warn(f"{alpha_value:.1e}, {key}: {w.message}")
                    
                except LinAlgError as e:
                    logger.warn(f"{alpha_value:.1e}, {key}: {e}")

                pred_train = clf.predict(X_train)[0]
                pred_test = clf.predict(X_test)[0]

                # Take force error (gradient wrt to positions) as scorer.
                f_pred_train = pred_train.gradient("positions").data
                f_pred_test = pred_test.gradient("positions").data

                # For energy models we take the force per molecule!
                if key == "e_f":
                    f_train = f_train_all
                    f_test = f_test_all

                if key == "e":
                    f_train = f_train_mol
                    f_test = f_test_mol

                    f_pred_train = [
                        np.sum(m, axis=0)
                        for m in np.split(f_pred_train, mol_idx_train_cumsum)
                    ]
                    f_pred_test = [
                        np.sum(m, axis=0)
                        for m in np.split(f_pred_test, mol_idx_test_cumsum)
                    ]

                    f_pred_train = np.array(f_pred_train)
                    f_pred_test = np.array(f_pred_test)

                l_clf[i_alpha] = clf

                l_rmse_f_train[i_alpha] = mean_squared_error(
                    f_pred_train.flatten(),
                    f_train.flatten(),
                    squared=False,
                )
                l_rmse_f_test[i_alpha] = mean_squared_error(
                    f_pred_test.flatten(),
                    f_test.flatten(),
                    squared=False,
                )

                # energy error as scorer.
                y_pred_train = pred_train.values.flatten()
                y_pred_test = pred_test.values.flatten()

                y_pred_train -= monomer_energies[realization.idx_train]
                y_pred_test -= monomer_energies[realization.idx_test]

                l_rmse_e_train[i_alpha] = mean_squared_error(
                    y_pred_train,
                    y_train_red,
                    squared=False,
                )
                l_rmse_e_test[i_alpha] = mean_squared_error(
                    y_pred_test,
                    y_test_red,
                    squared=False,
                )

            l_rmse_e_train *= 100 / sigma_energy
            l_rmse_e_test *= 100 / sigma_energy

            if key == "e_f":
                l_rmse_f_train *= 100 / sigma_force_all
                l_rmse_f_test *= 100 / sigma_force_all
            else:
                l_rmse_f_train *= 100 / sigma_force_mol
                l_rmse_f_test *= 100 / sigma_force_mol

            # Find index of best model
            best_idx = np.nanargmin((l_rmse_e_test + l_rmse_f_test) / 2)

            # Save data
            realization[key] = Bunch(
                alpha_values=alpha_values,
                l_clf=l_clf,
                l_rmse_e_train=l_rmse_e_train,
                l_rmse_e_test=l_rmse_e_test,
                l_rmse_f_train=l_rmse_f_train,
                l_rmse_f_test=l_rmse_f_test,
                best_idx=best_idx,
                clf=l_clf[best_idx],
                alpha=alpha_values[best_idx],
                rmse_e_train=l_rmse_e_train[best_idx],
                rmse_e_test=l_rmse_e_test[best_idx],
                rmse_f_train=l_rmse_f_train[best_idx],
                rmse_f_test=l_rmse_f_test[best_idx],
            )

    return results
