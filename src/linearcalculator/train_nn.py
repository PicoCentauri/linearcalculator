#!/usr/bin/env python3
import argparse
import sys

from typing import List

import ase
import ase.io
import equistore
import numpy as np
import torch
from rascaline import AtomicComposition, LodeSphericalExpansion, SphericalExpansion, SoapPowerSpectrum
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from .utils import compute_power_spectrum

# Create the parser
parser = argparse.ArgumentParser(description="Process some integers.")

# Add the arguments
parser.add_argument("-p", "--prefix", type=str, default="test", help="The prefix")
parser.add_argument("-d", "--dataset", type=str, help="dataset")
parser.add_argument("-rc", "--recalc", type=str, help="recalculate descriptors")
parser.add_argument("-r", "--rcut", type=float, default=4.0, help="training cutoff")
parser.add_argument("-s", "--stride", type=int, default=1, help="The stride")
parser.add_argument("-n", "--epochs", type=int, default=500, help="number of epochs")
parser.add_argument("-l", "--layer", type=int, default=128, help="The layer size")
parser.add_argument("-b", "--batch", type=int, default=128, help="The batch size")
parser.add_argument("-v", "--exponents", type=str, default="0136", help=' e.g. "016")')

# Parse the arguments
args = parser.parse_args()

# Store the arguments in variables
PREFIX = args.prefix
RECALC = args.recalc
DATASET = args.dataset
RCUT = args.rcut
STRIDE = args.stride
N_EPOCHS = args.epochs
LAYER_SIZE = args.layer
BATCH_SIZE = args.batch
EXPONENTS = args.exponents

####
# UTILITY FUNCTION DEFINITIONS
####

def split_mono(f):
    symbols_a = []
    positions_a = []
    symbols_b = []
    positions_b = []
    on_first = True
    prev_s = "C"
    for s, p in zip(f.symbols, f.positions):
        if s == "C" and prev_s != "C":
            on_first = False
        prev_s = s
        if on_first:
            symbols_a.append(s)
            positions_a.append(p)
        else:
            symbols_b.append(s)
            positions_b.append(p)
    # NB : this assumes monomers are in the right order, i.e. that a "CA" molecule will
    # have "C" first and "A" next this is not true so these assignments may be wrong
    mono_a = ase.Atoms(symbols_a, positions_a)
    mono_a.info["distance"] = 0
    mono_a.info["distance_initial"] = 0
    mono_a.info["label"] = f.info["label"][0]
    mono_a.info["energy"] = f.info["energyA"]
    mono_b = ase.Atoms(symbols_b, positions_b)
    mono_b.info["distance"] = 0
    mono_b.info["distance_initial"] = 0
    mono_b.info["label"] = f.info["label"][1]
    mono_b.info["energy"] = f.info["energyB"]
    return mono_a, mono_b


########
#  LOADS STUFF
##########

print("Loading structures")
frames = ase.io.read(DATASET, ":")
idx = list(range(len(frames)))

idxpair = 0
d = frames[0].info["distance"]
for f in frames:
    if f.info["distance"] < d:
        idxpair += 1
    d = f.info["distance"]
    f.info["indexAB"] = idxpair


mono = []
for f in frames:
    mono_a, mono_b = split_mono(f)
    # this avoids adding "wrong" monomers. see split_mono. ugly workaround but it works
    if mono_a.info["label"] == mono_b.info["label"]:
        mono.append(mono_a)
        mono.append(mono_b)

emono = []
for m in mono:
    emono.append(m.info["energy"])
euni, eidx = np.unique(emono, return_index=True)
mono_unique = [mono[idx] for idx in eidx]


mono_true_unique = [mono_unique[0]]
emono = [mono_unique[0].info["energy"]]
for m in mono:
    if np.abs(np.array(emono) - m.info["energy"]).min() > 2e-3:
        mono_true_unique.append(m)
        emono.append(m.info["energy"])

# THESE ARE THE FINAL FRAMES WE USE
combined_frames = frames + mono_true_unique

# COMPOSITION BASELINING
co_calculator = AtomicComposition(per_structure=True)

co_X = co_calculator.compute(combined_frames).values()
y = np.array([f.info["energy"] for f in combined_frames])


rcv = RidgeCV(alphas=np.geomspace(1e-8, 1e2, 10), fit_intercept=False)
rcv.fit(co_X, y)
yp_base = rcv.predict(co_X)

targets = torch.from_numpy(y - yp_base)

##################
# COMPUTE FEATURES
##################

def compute_descriptors(
    frames: List[ase.Atoms],
    potential_exponent: int = 0,
    ps_fname: str = "descriptor_ps.npz",
):
    compute_args = {"systems": frames}

    if RECALC:
        # Compute spherical expanions and power spectrum
        if potential_exponent != 0:
            sr_calculator = SphericalExpansion(**sr_hypers)
            sr_descriptor = sr_calculator.compute(**compute_args)

            lr_calculator = LodeSphericalExpansion(
                potential_exponent=potential_exponent, **lr_hypers)
            lr_descriptor = lr_calculator.compute(**compute_args)

            ts = compute_power_spectrum(sr_descriptor, lr_descriptor)

            del lr_descriptor
            del sr_descriptor
        else:
            sr_calculator = SoapPowerSpectrum(**sr_hypers)
            ts = sr_calculator.compute(**compute_args)

        ts = ts.keys_to_properties(["species_neighbor_1", "species_neighbor_2"])

        equistore.save(ps_fname, ps)
    else:
        print("Load descriptors from file")
        ps = equistore.load(ps_fname)

    return ps


# RHO
sr_hypers = {
    "cutoff": 3.0,
    "max_radial": 6,
    "max_angular": 4,
    "atomic_gaussian_width": 0.3,
    "radial_basis": {"Gto": {}},
    "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
    "center_atom_weight": 1.0,
}


lr_hypers = {
    "cutoff": 3.0,
    "max_radial": 6,
    "max_angular": 2,
    "atomic_gaussian_width": 1.0,
    "radial_basis": {"Gto": {}},
    "center_atom_weight": 1.0,
}


lfeats = []
for i, potential_exponent in enumerate(EXPONENTS):
    ps_fname = f"descriptor_ps_{i}.npz"
    ps_current = compute_descriptors(
        frames=combined_frames,
        potential_exponent=potential_exponent,
        ps_fname=ps_fname,
    ).block().values

    # Truncating features and remove some! maybe remove for final learning
    tsvd = TruncatedSVD(n_components=ps_current.shape[1] // 2)
    ps_current = tsvd.fit_transform(ps_current[::5])

    # Standardizing features
    scaler = StandardScaler()
    ps_current = scaler.fit_transform(ps_current)

    lfeats.append(torch.from_numpy(ps_current))

feats = torch.hstack(lfeats)

print("Setting up training")


emono = torch.zeros(len(combined_frames))
itrain = []
itest = []

for i, atoms in combined_frames:
    if "energyA" in atoms.info:
        emono[i] = atoms.info["energyA"] + atoms.info["energyB"]
    else:
        emono[i] = atoms.info["energy"]

    delta_distance = atoms.info["distance"] - atoms.info["distance_initial"]
    if delta_distance <= RCUT:
        itest.append(i)
    else:
        itest.append(i)

itrain = torch.from_numpy(itrain)
itest = torch.from_numpy(itest)


# find sample species for index add
sr_calculator = SphericalExpansion(**sr_hypers)
sr_descriptor = sr_calculator.compute(combined_frames)
smpl = sr_descriptor.keys_to_samples(["species_center"])[0].samples
structure_samples = torch.tensor(
    smpl.view(dtype=np.int32).reshape(len(smpl), len(smpl.names))
)


def loss_mse(predicted, actual):
    return torch.sum((predicted.flatten() - actual.flatten()) ** 2)

energy_model = torch.nn.Sequential(
    torch.nn.Linear(feats.shape[-1], LAYER_SIZE),
    torch.nn.Tanh(),
    torch.nn.LayerNorm(LAYER_SIZE),
    torch.nn.Linear(LAYER_SIZE, LAYER_SIZE),
    torch.nn.Tanh(),
    torch.nn.LayerNorm(LAYER_SIZE),
    torch.nn.Linear(LAYER_SIZE, 1),
)

optimizer = torch.optim.AdamW(energy_model.parameters(), lr=1e-3)

# Decay learning rate by a factor of 0.5 every 50 epochs after step 300
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)


batch_idx = itrain.copy()

for epoch in range(N_EPOCHS):
    np.random.shuffle(batch_idx)

    # manual dataloder...
    for ibatch in range(len(batch_idx) // BATCH_SIZE):
        batch_structs = batch_idx[ibatch * BATCH_SIZE : (ibatch + 1) * BATCH_SIZE]
        batch_sel = np.where(np.isin(structure_samples[:, 0], batch_structs))[0]
        batch_samples = structure_samples[batch_sel, 0]
        batch_feats = feats[batch_sel]
        batch_tgt = targets[batch_structs]

        predicted = energy_model(batch_feats)
        predicted_structure = torch.zeros((len(targets), 1))
        predicted_structure.index_add_(0, batch_samples, predicted)
        loss = loss_mse(predicted_structure[batch_structs], batch_tgt)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    if epoch >= 300:
        scheduler.step()

    if epoch % 2 == 0:
        predicted = energy_model(feats)
        predicted_structure = torch.zeros((len(targets), 1))
        predicted_structure.index_add_(0, structure_samples[:, 0], predicted)

        print(
            "Epoch:",
            epoch,
            "Energy RMSE: train ",
            np.sqrt(loss.detach().cpu().numpy().flatten()[0] / len(itrain)),
            "test",
            np.sqrt(
                loss_mse(predicted_structure[itest], targets[itest])
                .detach()
                .cpu()
                .numpy()
                .flatten()[0]
                / len(itest)
            ),
        )

        sys.stdout.flush()

        for i, f in enumerate(combined_frames):
            f.info["predicted_energy"] = (
                yp_base[i] + predicted_structure[i].detach().cpu().numpy().flatten()[0]
            )
            f.info["predicted_binding"] = (
                yp_base[i]
                + predicted_structure[i].detach().cpu().numpy().flatten()[0]
                - emono[i]
            )
            f.info["binding"] = y[i] - emono[i]
            f.info["error_binding"] = (
                yp_base[i]
                + predicted_structure[i].detach().cpu().numpy().flatten()[0]
                - y[i]
            )
            if i in itrain:
                f.info["split"] = "train"
            else:
                f.info["split"] = "test"

        torch.save(energy_model.state_dict(), PREFIX + "-checkpoint.torch")
        ase.io.write(PREFIX + "-checkpoint.xyz", combined_frames)

torch.save(energy_model.state_dict(), PREFIX + "-model.torch")

predicted = energy_model(feats)
predicted_structure = torch.zeros((len(targets), 1))
predicted_structure.index_add_(0, structure_samples[:, 0], predicted)

for i, f in enumerate(combined_frames):
    f.info["predicted_energy"] = (
        yp_base[i] + predicted_structure[i].detach().cpu().numpy().flatten()[0]
    )
    f.info["predicted_binding"] = (
        yp_base[i]
        + predicted_structure[i].detach().cpu().numpy().flatten()[0]
        - emono[i]
    )
    f.info["binding"] = y[i] - emono[i]
    f.info["error_binding"] = (
        yp_base[i] + predicted_structure[i].detach().cpu().numpy().flatten()[0] - y[i]
    )
    if i in itrain:
        f.info["split"] = "train"
    else:
        f.info["split"] = "test"

ase.io.write(PREFIX + "-final.xyz", combined_frames)
