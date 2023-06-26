import logging

import equistore
import numpy as np
from equisolve.utils.convert import ase_to_tensormap
from equistore import Labels
from sklearn.utils import Bunch
from tqdm.auto import tqdm

import torch

from .utils import compute_descriptors, setup_dataset


logger = logging.getLogger(__name__)


def compute_nn_models(config):
    frames = setup_dataset(config["dataset"], config["label"], config["cell_length"])

    potential_exponents = config["potential_exponents"]
    if type(potential_exponents) == int:
        potential_exponents = [potential_exponents]

    ps = []
    for i, potential_exponent in enumerate(potential_exponents):
        ps_fname = f"descriptor_ps_{i}.npz"
        co, ps_current = compute_descriptors(
            frames=frames,
            config=config,
            potential_exponent=potential_exponent,
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

        if len(idx_train) == 0:
            raise ValueError(
                f"No training samples for " f"training_cutoff={training_cutoff}!"
            )

        if len(idx_test) == 0:
            raise ValueError(
                f"No test samples for " f"training_cutoff={training_cutoff}!"
            )

        results[training_cutoff] = Bunch(idx_train=idx_train, idx_test=idx_test)

    y = ase_to_tensormap(frames, energy="energy", forces="forces")
    monomer_energies = np.array([f.info["energyA"] + f.info["energyB"] for f in frames])

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

        # Convert numpy arrays to torch tensors
        X_train = torch.from_numpy(X_train)
        X_test = torch.from_numpy(X_test)

        y_train = torch.from_numpy(y_train)
        y_test = torch.from_numpy(y_test)

        y_train_red = torch.from_numpy(y_train)
        y_test_red = torch.from_numpy(y_test)

        sigma_energy = torch.std(y_train_red)


        # # Compute energy RMSE
        # y_pred_train = pred_train.values.flatten()
        # y_pred_test = pred_test.values.flatten()

        # y_pred_train -= monomer_energies[realization.idx_train]
        # y_pred_test -= monomer_energies[realization.idx_test]

        # rmse_y_train = mean_squared_error(
        #     y_pred_train, y_train_red, squared=False
        # )
        # rmse_y_test = mean_squared_error(y_pred_test, y_test_red, squared=False)

        # Store results
        realization.xxx = Bunch()

    return results
