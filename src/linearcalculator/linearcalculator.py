"""The main function for generating the linear model."""

import logging
import os
import warnings
from typing import List

import ase
import equistore
import numpy as np
from equisolve.numpy.models import Ridge
from equisolve.utils.convert import ase_to_tensormap
from equistore import Labels
from numpy.linalg import LinAlgError
from rascaline import (
    AtomicComposition,
    LodeSphericalExpansion,
    SoapPowerSpectrum,
    SphericalExpansion,
)
from sklearn.metrics import mean_squared_error
from sklearn.utils import Bunch
from tqdm.auto import tqdm

from .utils import (
    PARAMETER_KEYS_DICT,
    compute_power_spectrum,
    setup_dataset,
    training_curve_split,
)


logger = logging.getLogger(__name__)


def compute_descriptors(frames: List[ase.Atoms], config: dict):
    """Compute atomic the composition and the power spectrum descriptor.

    Parameters
    ----------
    frames : List[ase.Atoms]
        A list of atomic systems represented as ASE Atoms objects.
    config : dict
        A dictionary containing the configuration parameters for descriptor computation.

    Returns
    -------
    co : equistore.TensorMap
        composition descriptor (co) representing the atomic composition of the
        structures.
    ps : equistore.TensorMap
        power spectrum descriptor (ps) representing the rotationally invariant power
        spectrum.
    """

    # Compute descriptor
    fname_sr_descriptor = os.path.join(config["output"], "sr_descriptor.npz")
    fname_lr_descriptor = os.path.join(config["output"], "lr_descriptor.npz")

    compute_args = {"systems": frames, "gradients": ["positions"]}
    if config["lr_hypers"]["potential_exponent"] != 0:
        if config["recalc_descriptors"]:
            logger.info("Compute short-range descriptor")
            sr_calculator = SphericalExpansion(**config["sr_hypers"])
            sr_descriptor = sr_calculator.compute(**compute_args)
            equistore.io.save(fname_sr_descriptor, sr_descriptor)

            logger.info("Compute long-range descriptor")
            lr_calculator = LodeSphericalExpansion(**config["lr_hypers"])
            lr_descriptor = lr_calculator.compute(**compute_args)
            equistore.io.save(fname_lr_descriptor, lr_descriptor)
        else:
            logger.info("Load short-range descriptor")
            sr_descriptor = equistore.io.load(fname_sr_descriptor)

            logger.info("Load long-range descriptor")
            lr_descriptor = equistore.io.load(fname_lr_descriptor)

    # Compute powerspectrum
    fname_ps = "power_spectrum.npz"
    if config["recalc_power_spectrum"]:
        logger.info("Compute power spectrum")
        if config["lr_hypers"]["potential_exponent"] == 0:
            sr_calculator = SoapPowerSpectrum(**config["sr_hypers"])
            ts = sr_calculator.compute(**compute_args)
        else:
            ts = compute_power_spectrum(sr_descriptor, lr_descriptor)
            ts = ts.keys_to_properties(["spherical_harmonics_l"])

            del lr_descriptor
            del sr_descriptor

        ts = ts.keys_to_properties(["species_neighbor_1", "species_neighbor_2"])
        ts = ts.keys_to_samples(["species_center"])

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

    return co, ps


def compute_linear_models(config: dict):
    frames = setup_dataset(config["dataset"])

    co, ps = compute_descriptors(frames, config)
    X = equistore.join([co, ps], axis="properties")

    # Setup training curve
    l_idx_train, idx_test = training_curve_split(
        n_structures=len(frames),
        train_size=config["train_size"],
        n_train_num=config["n_train_num"],
        n_train_start=config["n_train_start"],
        random_state=config["random_state"],
    )

    results = Bunch()
    for idx_train in l_idx_train:
        results[len(idx_train)] = Bunch(idx_train=idx_train, idx_test=idx_test)

    labels = Labels(["structure"], np.array([[0]]))

    # Set composition calculator to machine precision
    alpha_co = equistore.slice(equistore.ones_like(co), axis="samples", labels=labels)
    alpha_co *= np.finfo(alpha_co[0].values[0, 0]).eps

    alpha_ps = equistore.slice(equistore.ones_like(ps), axis="samples", labels=labels)

    # Setup variables for model paramaters
    y = ase_to_tensormap(frames, energy="energy", forces="forces")

    alpha_values = np.logspace(-12, 3, 20)

    # Fit the models
    for realization in tqdm(results.values(), desc="Fit models"):
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
        tensor_y_train, tensor_y_test = equistore.split(y, **split_args)

        # Forces
        f_train = tensor_y_train[0].gradient("positions").data.flatten()
        f_test = tensor_y_test[0].gradient("positions").data.flatten()

        # Energies
        y_train = tensor_y_train[0].values.flatten()
        y_test = tensor_y_test[0].values.flatten()

        sigma_y = np.std(y_train)
        sigma_f = np.std(f_train)

        for key, parameter_keys in PARAMETER_KEYS_DICT.items():
            # Create lists for storing values
            l_clf = len(alpha_values) * [None]

            l_f_pred_train = len(alpha_values) * [None]
            l_f_pred_test = len(alpha_values) * [None]
            l_rmse_f_train = np.nan * np.ones(len(alpha_values))
            l_rmse_f_test = np.nan * np.ones(len(alpha_values))

            l_y_pred_train = len(alpha_values) * [None]
            l_y_pred_test = len(alpha_values) * [None]
            l_rmse_y_train = np.nan * np.ones(len(alpha_values))
            l_rmse_y_test = np.nan * np.ones(len(alpha_values))

            for i_alpha, alpha_value in enumerate(alpha_values):
                clf = Ridge(parameter_keys=parameter_keys)

                alpha_ps[0].values[:] = alpha_value
                alpha = equistore.join([alpha_co, alpha_ps], axis="properties")

                try:
                    with warnings.catch_warnings(record=True) as warns:
                        clf.fit(X_train, tensor_y_train, alpha=alpha)
                        for w in warns:
                            logger.warn(f"{alpha_value:.1e}, {key}: {w.message}")

                except LinAlgError as e:
                    logger.warn(f"{alpha_value:.1e}, {key}: {e}")

                # Predict values and gradients
                pred_train = clf.predict(X_train)[0]
                pred_test = clf.predict(X_test)[0]

                # Compute gradient RMSE
                f_pred_train = pred_train.gradient("positions").data.flatten()
                f_pred_test = pred_test.gradient("positions").data.flatten()

                # rmse_f_train = mean_squared_error(f_pred_train, f_train, squared=False)
                rmse_f_test = mean_squared_error(f_pred_test, f_test, squared=False)

                # Compute energy RMSE
                y_pred_train = pred_train.values.flatten()
                y_pred_test = pred_test.values.flatten()

                rmse_y_train = mean_squared_error(y_pred_train, y_train, squared=False)
                rmse_y_test = mean_squared_error(y_pred_test, y_test, squared=False)

                # Store predictions
                l_clf[i_alpha] = clf

                l_f_pred_train[i_alpha] = f_pred_train
                l_f_pred_test[i_alpha] = f_pred_test
                l_rmse_f_test[i_alpha] = rmse_f_test
                # l_rmse_f_train[i_alpha] = rmse_f_train

                l_y_pred_train[i_alpha] = y_pred_train
                l_y_pred_test[i_alpha] = y_pred_test
                l_rmse_y_train[i_alpha] = rmse_y_train
                l_rmse_y_test[i_alpha] = rmse_y_test

            l_rmse_f_train *= 100 / sigma_f
            l_rmse_f_test *= 100 / sigma_f

            l_rmse_y_train *= 100 / sigma_y
            l_rmse_y_test *= 100 / sigma_y

            # Find index of best model
            best_idx = np.nanargmin((l_rmse_y_test + l_rmse_f_test) / 2)

            # Save data
            realization[key] = Bunch(
                alpha_values=alpha_values,
                l_clf=l_clf,
                best_idx=best_idx,
                clf=l_clf[best_idx],
                alpha=alpha_values[best_idx],
                # gradients
                l_rmse_f_train=l_rmse_f_train,
                l_rmse_f_test=l_rmse_f_test,
                f_pred_test=l_f_pred_test[best_idx],
                f_pred_train=l_f_pred_train[best_idx],
                rmse_f_train=l_rmse_f_train[best_idx],
                rmse_f_test=l_rmse_f_test[best_idx],
                # values
                l_rmse_y_train=l_rmse_y_train,
                l_rmse_y_test=l_rmse_y_test,
                y_pred_test=l_y_pred_test[best_idx],
                y_pred_train=l_y_pred_train[best_idx],
                rmse_y_train=l_rmse_y_train[best_idx],
                rmse_y_test=l_rmse_y_test[best_idx],
            )

    return results
