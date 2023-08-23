import itertools
import logging
import os
import pickle
import warnings

import equistore
import numpy as np
from equisolve.numpy.models import Ridge
from equisolve.utils.convert import ase_to_tensormap
from equistore import Labels
from numpy.linalg import LinAlgError
from sklearn.metrics import mean_squared_error
from sklearn.utils import Bunch
from tqdm.auto import tqdm

from .utils import compute_descriptors, setup_dataset


logger = logging.getLogger(__name__)


def compute_linear_models(config):
    frames = setup_dataset(config["dataset"], config["label"], config["cell_length"])

    potential_exponents = config["potential_exponents"]
    if type(potential_exponents) == int:
        potential_exponents = [potential_exponents]

    ps = []
    parameter_keys = ["e"]

    if config["forces"]:
        gradients = ["positions"]
        parameter_keys.append("e_f")
    else:
        gradients = None

    for i, potential_exponent in enumerate(potential_exponents):
        ps_fname = f"descriptor_ps_{potential_exponent}.npz"
        co, ps_current = compute_descriptors(
            frames=frames,
            config=config,
            potential_exponent=potential_exponent,
            gradients=gradients,
            ps_fname=ps_fname,
        )
        ps.append(ps_current)

    X = equistore.join([co] + ps, axis="properties")

    # Setup training curve
    results = Bunch()
    for training_cutoff in config["training_cutoffs"]:
        idx_train = []
        idx_test = []
        for i, atoms in enumerate(frames):
            delta_distance = atoms.info["distance"] - atoms.info["distance_initial"]
            if delta_distance <= training_cutoff:
                idx_train.append(i)
            else:
                idx_test.append(i)

        if not idx_train:
            raise ValueError(
                f"No training samples for " f"training_cutoff={training_cutoff}!"
            )

        if not idx_test:
            warnings.warn(
                f"No training samples for " f"training_cutoff={training_cutoff}!"
            )

        results[training_cutoff] = Bunch(idx_train=idx_train, idx_test=idx_test)

    alpha_params = {"axis": "samples", "labels": Labels(["structure"], np.array([[0]]))}
    # Set composition calculator to machine precision
    alpha_co = equistore.slice(equistore.ones_like(co), **alpha_params)
    alpha_co *= np.finfo(alpha_co[0].values[0, 0]).eps

    alpha_ps = []
    for i in range(len(potential_exponents)):
        alpha_ps.append(equistore.slice(equistore.ones_like(ps[i]), **alpha_params))

    # Setup variables for model paramaters
    y = ase_to_tensormap(frames, energy="energy", forces="forces")

    monomer_energies = np.array([f.info["energyA"] + f.info["energyB"] for f in frames])

    # Create combinations for gridsearch
    if "alpha_values" in config.keys():
        alpha_values = config["alpha_values"]
    else:
        alpha_values_single = np.logspace(-12, 3, config["n_alpha_values"])
        alpha_values = list(
            itertools.product(*(len(potential_exponents) * [alpha_values_single]))
        )

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
        f_train = y_train[0].gradient("positions").values
        f_test = y_test[0].gradient("positions").values

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
            np.sum(m, axis=0) for m in np.split(f_train, mol_idx_train_cumsum)
        ]
        f_test_mol = [np.sum(m, axis=0) for m in np.split(f_test, mol_idx_test_cumsum)]

        f_train_mol = np.array(f_train_mol)
        f_test_mol = np.array(f_test_mol)

        sigma_energy = np.std(y_train_red)
        sigma_force = np.std(f_train)
        sigma_force_mol = np.std(f_train_mol)

        for key in parameter_keys:
            if key == "e":
                X_train_cur = equistore.remove_gradients(X_train)
                y_train_cur = equistore.remove_gradients(y_train)
            else:
                X_train_cur = X_train.copy()
                y_train_cur = y_train.copy()

            # Create lists for storing values
            l_clf = len(alpha_values) * [None]

            l_f_pred_train = len(alpha_values) * [None]
            l_f_pred_test = len(alpha_values) * [None]
            l_rmse_f_train = np.nan * np.ones(len(alpha_values))
            l_rmse_f_test = np.nan * np.ones(len(alpha_values))

            l_f_pred_train_mol = len(alpha_values) * [None]
            l_f_pred_test_mol = len(alpha_values) * [None]
            l_rmse_f_train_mol = np.nan * np.ones(len(alpha_values))
            l_rmse_f_test_mol = np.nan * np.ones(len(alpha_values))

            l_y_pred_train = len(alpha_values) * [None]
            l_y_pred_test = len(alpha_values) * [None]
            l_rmse_y_train = np.nan * np.ones(len(alpha_values))
            l_rmse_y_test = np.nan * np.ones(len(alpha_values))

            for i_alpha, alpha_value in enumerate(alpha_values):
                clf = Ridge()

                # Set alpha_value for each potential exponent
                for i in range(len(potential_exponents)):
                    alpha_ps[i][0].values[:] = alpha_value[i]

                alpha = equistore.join([alpha_co] + alpha_ps, axis="properties")

                if key == "e":
                    alpha_cur = equistore.remove_gradients(alpha)
                else:
                    alpha_cur = alpha.copy()

                try:
                    with warnings.catch_warnings(record=True) as warns:
                        clf.fit(X_train_cur, y_train_cur, alpha=alpha_cur)
                        for w in warns:
                            logger.warn(f"{alpha_value}, {key}: {w.message}")

                except LinAlgError as e:
                    logger.warn(f"{alpha_value}, {key}: {e}")

                # Predict values and gradients
                pred_train = clf.predict(X_train)[0]
                pred_test = clf.predict(X_test)[0]

                # Compute gradient RMSE
                if config["forces"]:
                    f_pred_train = pred_train.gradient("positions").values
                    f_pred_test = pred_test.gradient("positions").values

                    rmse_f_train = mean_squared_error(
                        f_pred_train.flatten(),
                        f_train.flatten(),
                        squared=False,
                    )

                    if idx_test:
                        rmse_f_test = mean_squared_error(
                            f_pred_test.flatten(),
                            f_test.flatten(),
                            squared=False,
                        )
                    else:
                        rmse_f_test = np.nan

                    # Compute gradient per molecules RMSE
                    f_pred_train_mol = np.array(
                        [
                            np.sum(m, axis=0)
                            for m in np.split(f_pred_train, mol_idx_train_cumsum)
                        ]
                    )
                    if idx_test:
                        f_pred_test_mol = np.array(
                            [
                                np.sum(m, axis=0)
                                for m in np.split(f_pred_test, mol_idx_test_cumsum)
                            ]
                        )
                    else:
                        f_pred_test_mol = np.nan

                    rmse_f_train_mol = mean_squared_error(
                        f_pred_train_mol.flatten(),
                        f_train_mol.flatten(),
                        squared=False,
                    )
                    if idx_test:
                        rmse_f_test_mol = mean_squared_error(
                            f_pred_test_mol.flatten(),
                            f_test_mol.flatten(),
                            squared=False,
                        )
                    else:
                        rmse_f_test_mol = np.nan

                    l_f_pred_train[i_alpha] = f_pred_train
                    l_f_pred_test[i_alpha] = f_pred_test
                    l_rmse_f_test[i_alpha] = rmse_f_test
                    l_rmse_f_train[i_alpha] = rmse_f_train

                    l_f_pred_train_mol[i_alpha] = f_pred_train_mol
                    l_f_pred_test_mol[i_alpha] = f_pred_test_mol
                    l_rmse_f_test_mol[i_alpha] = rmse_f_test_mol
                    l_rmse_f_train_mol[i_alpha] = rmse_f_train_mol

                # Compute energy RMSE
                y_pred_train = pred_train.values.flatten()
                y_pred_test = pred_test.values.flatten()

                y_pred_train -= monomer_energies[realization.idx_train]
                y_pred_test -= monomer_energies[realization.idx_test]

                rmse_y_train = mean_squared_error(
                    y_pred_train, y_train_red, squared=False
                )
                if idx_test:
                    rmse_y_test = mean_squared_error(
                        y_pred_test, y_test_red, squared=False
                    )
                else:
                    rmse_y_test = np.nan

                # Store predictions
                l_clf[i_alpha] = clf

                l_y_pred_train[i_alpha] = y_pred_train
                l_y_pred_test[i_alpha] = y_pred_test
                l_rmse_y_train[i_alpha] = rmse_y_train
                l_rmse_y_test[i_alpha] = rmse_y_test

                _conclude(
                    config=config,
                    realization=realization,
                    results=results,
                    key=key,
                    alpha_values=alpha_values,
                    l_clf=l_clf,
                    l_rmse_y_train=l_rmse_y_train,
                    l_rmse_y_test=l_rmse_y_test,
                    l_y_pred_train=l_y_pred_train,
                    l_y_pred_test=l_y_pred_test,
                    sigma_energy=sigma_energy,
                    sigma_force=sigma_force,
                    l_rmse_f_train=l_rmse_f_train,
                    l_rmse_f_test=l_rmse_f_test,
                    l_f_pred_train=l_f_pred_train,
                    l_f_pred_test=l_f_pred_test,
                    sigma_force_mol=sigma_force_mol,
                    l_rmse_f_train_mol=l_rmse_f_train_mol,
                    l_rmse_f_test_mol=l_rmse_f_test_mol,
                    l_f_pred_train_mol=l_f_pred_train_mol,
                    l_f_pred_test_mol=l_f_pred_test_mol,
                )


def _conclude(
    config,
    realization,
    results,
    key,
    alpha_values,
    l_clf,
    l_rmse_y_train,
    l_rmse_y_test,
    l_y_pred_train,
    l_y_pred_test,
    sigma_energy,
    sigma_force=None,
    l_rmse_f_train=None,
    l_rmse_f_test=None,
    l_f_pred_train=None,
    l_f_pred_test=None,
    sigma_force_mol=None,
    l_rmse_f_train_mol=None,
    l_rmse_f_test_mol=None,
    l_f_pred_train_mol=None,
    l_f_pred_test_mol=None,
):
    """Calculate results and save them to a pickle file."""
    if config["forces"]:
        l_rmse_f_train_out = l_rmse_f_train * 100 / sigma_force
        l_rmse_f_test_out = l_rmse_f_test * 100 / sigma_force

        l_rmse_f_train_mol_out = l_rmse_f_train_mol * 100 / sigma_force_mol
        l_rmse_f_test_mol_out = l_rmse_f_test_mol * 100 / sigma_force_mol

    l_rmse_y_train_out = l_rmse_y_train * 100 / sigma_energy
    l_rmse_y_test_out = l_rmse_y_test * 100 / sigma_energy

    # Find index of best model
    try:
        if key == "e_f":
            best_idx = np.nanargmin((l_rmse_y_test + l_rmse_f_test) / 2)
        elif key == "e" and config["forces"]:
            best_idx = np.nanargmin((l_rmse_y_test + l_rmse_f_test_mol) / 2)
        else:
            best_idx = np.nanargmin(l_rmse_y_test)
    except ValueError as err:
        logger.error("All test values are no numbers. Use train values instead.")
        if key == "e_f":
            best_idx = np.nanargmin((l_rmse_y_train + l_rmse_f_train) / 2)
        elif key == "e" and config["forces"]:
            best_idx = np.nanargmin((l_rmse_y_train + l_rmse_f_train_mol) / 2)
        else:
            best_idx = np.nanargmin(l_rmse_y_train)

    # Save data
    realization[key] = Bunch(
        # model
        alpha_values=alpha_values,
        l_clf=l_clf,
        best_idx=best_idx,
        clf=l_clf[best_idx],
        alpha=alpha_values[best_idx],
        # values
        l_rmse_y_train=l_rmse_y_train_out,
        l_rmse_y_test=l_rmse_y_test_out,
        y_pred_test=l_y_pred_test[best_idx],
        y_pred_train=l_y_pred_train[best_idx],
        rmse_y_train=l_rmse_y_train_out[best_idx],
        rmse_y_test=l_rmse_y_test_out[best_idx],
        # auxiliary
        sigma_energy=sigma_energy,
    )

    if config["forces"]:
        # gradients
        realization[key].sigma_force = sigma_force
        realization[key].l_rmse_f_train = l_rmse_f_train_out
        realization[key].l_rmse_f_test = l_rmse_f_test_out
        realization[key].f_pred_train = l_f_pred_train[best_idx]
        realization[key].f_pred_test = l_f_pred_test[best_idx]
        realization[key].rmse_f_train = l_rmse_f_train_out[best_idx]
        realization[key].rmse_f_test = l_rmse_f_test_out[best_idx]
        # gradients per molecule
        realization[key].sigma_force_mol = sigma_force_mol
        realization[key].l_rmse_f_train_mol = l_rmse_f_train_mol_out
        realization[key].l_rmse_f_test_mol = l_rmse_f_test_mol_out
        realization[key].f_pred_train_mol = l_f_pred_train_mol[best_idx]
        realization[key].f_pred_test_mol = l_f_pred_test_mol[best_idx]
        realization[key].rmse_f_train_mol = l_rmse_f_train_mol_out[best_idx]
        realization[key].rmse_f_test_mol = l_rmse_f_test_mol_out[best_idx]

    # Save models to pickle file
    results_out = results.copy()
    results_out["config"] = config

    with open(os.path.join(config["output"], "results.pickle"), "wb") as f:
        pickle.dump(results_out, f)
