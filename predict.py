"""
Script for generating molecules using a trained model and saving them.

Note that the script currently does not save the molecules in batches - all of the molecules are generated and then
all saved together in one Smol batch. If generating many molecules ensure you have enough memory to store them.
"""

import argparse
import os
from functools import partial
from pathlib import Path

import lightning as L
import torch
from rdkit import Chem

import semlaflow.scriptutil as util
from semlaflow.data.datamodules import GeometricInterpolantDM
from semlaflow.data.datasets import GeometricDataset
from semlaflow.data.interpolate import GeometricInterpolant, GeometricNoiseSampler
# from semlaflow.models.fm import Integrator, MolecularCFM
from semlaflow.models.temp_fm import Integrator, MolecularCFM
from semlaflow.models.semla import EquiInvDynamics, SemlaGenerator
from semlaflow.util.molrepr import GeometricMolBatch

# Default script arguments
DEFAULT_SAVE_FILE = "predictions.smol"
DEFAULT_DATASET_SPLIT = "train"
DEFAULT_N_MOLECULES = 500
DEFAULT_BATCH_COST = 8192
DEFAULT_BUCKET_COST_SCALE = "linear"
DEFAULT_INTEGRATION_STEPS = 100
DEFAULT_CAT_SAMPLING_NOISE_LEVEL = 1
DEFAULT_ODE_SAMPLING_SCHEDULER = "arcsin"
DEFAULT_COORD_NOISE_STD_DEV = 0.0
DEFAULT_MAX_SIGMA = 80
DEFAULT_MIN_SIGMA = 0.001
DEFAULT_MASK_TIMES_FACTOR = 1.0
DEFAULT_MASK_RATE_STRATEGY = 'edm'
DEFAULT_INCLUDE_CHARGE=True
DEFAULT_FIRST_TERM_COEF=1
DEFAULT_SAMPLING_STRATEGY_FACTOR=2.0
DEFAULT_SAMPLER = "euler"
DEFAULT_TEMPERATURE = 1.0
DEFAULT_LOW_CONFIDENCE_REMASK = None

def load_model(args, vocab, dm):
    checkpoint = torch.load(args.ckpt_path)
    hparams = checkpoint["hyper_parameters"]

    hparams["compile_model"] = False
    hparams["integration-steps"] = args.integration_steps
    hparams["sampling_scheduler"] = args.ode_sampling_scheduler
    if 'use_dist_loss' not in hparams:
        hparams['use_dist_loss'] = args.use_dist_loss

    n_bond_types = util.get_n_bond_types(hparams["integration-type-strategy"])

    # Set default arch to semla if nothing has been saved
    if hparams.get("architecture") is None:
        hparams["architecture"] = "semla"

    if hparams["architecture"] == "semla":
        dynamics = EquiInvDynamics(
            hparams["d_model"],
            hparams["d_message"],
            hparams["n_coord_sets"],
            hparams["n_layers"],
            n_attn_heads=hparams["n_attn_heads"],
            d_message_hidden=hparams["d_message_hidden"],
            d_edge=hparams["d_edge"],
            self_cond=hparams["self_cond"],
            coord_norm=hparams["coord_norm"],
        )
        egnn_gen = SemlaGenerator(
            hparams["d_model"],
            dynamics,
            vocab.size,
            hparams["n_atom_feats"],
            d_edge=hparams["d_edge"],
            n_edge_types=n_bond_types,
            self_cond=hparams["self_cond"],
            size_emb=hparams["size_emb"],
            max_atoms=hparams["max_atoms"],
        )

    elif hparams["architecture"] == "eqgat":
        from semlaflow.models.eqgat import EqgatGenerator

        egnn_gen = EqgatGenerator(
            hparams["d_model"],
            hparams["n_layers"],
            hparams["n_equi_feats"],
            vocab.size,
            hparams["n_atom_feats"],
            hparams["d_edge"],
            hparams["n_edge_types"]
        )

    elif hparams["architecture"] == "egnn":
        from semlaflow.models.egnn import VanillaEgnnGenerator

        n_layers = args.n_layers if hparams.get("n_layers") is None else hparams["n_layers"]
        if n_layers is None:
            raise ValueError("No hparam for n_layers was saved, use script arg to provide n_layers")

        egnn_gen = VanillaEgnnGenerator(
            hparams["d_model"],
            n_layers,
            vocab.size,
            hparams["n_atom_feats"],
            d_edge=hparams["d_edge"],
            n_edge_types=n_bond_types
        )

    else:
        raise ValueError(f"Unknown architecture hyperparameter.")

    type_mask_index = (
        vocab.indices_from_tokens(["<MASK>"])[0] if hparams["train-type-interpolation"] == "unmask" else None
    )
    bond_mask_index = util.BOND_MASK_INDEX if hparams["train-bond-interpolation"] == "unmask" else None
    if args.include_charge:
        charge_sampling_strategy = hparams["integration-type-strategy"]
    else:
        charge_sampling_strategy = None
    integrator = Integrator(
        args.integration_steps,
        type_strategy=hparams["integration-type-strategy"],
        bond_strategy=hparams["integration-bond-strategy"],
        charge_strategy=charge_sampling_strategy,
        coord_noise_std=args.coord_noise_std_dev,
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index,
        cat_noise_level=args.cat_sampling_noise_level,
        prior_sampler = dm.prior_sampler,
        data_module = dm,
        mask_times_factor = args.mask_times_factor,
        use_edm_mask_step = args.use_edm_mask_step,
        mask_rate_strategy = args.mask_rate_strategy,
        max_sigma = args.max_sigma,
        min_sigma = args.min_sigma,
        use_heun = args.use_heun,
        include_charge=args.include_charge,
        first_term_coef=args.first_term_coef,
        adaptive_cat_noise_level=args.adaptive_cat_noise_level,
        sampler=args.sampler,
        temperature=args.temperature,
    )
    fm_model = MolecularCFM.load_from_checkpoint(
        args.ckpt_path,
        gen=egnn_gen,
        vocab=vocab,
        integrator=integrator,
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index,
        include_charge=args.include_charge,
        sampling_strategy_factor=args.sampling_strategy_factor,
        low_confidence_remask=args.low_confidence_remask,
        **hparams,
    )
    return fm_model


def build_dm(args, hparams, vocab):
    if args.dataset == "qm9":
        coord_std = util.QM9_COORDS_STD_DEV
        bucket_limits = util.QM9_BUCKET_LIMITS

    elif args.dataset == "geom-drugs":
        coord_std = util.GEOM_COORDS_STD_DEV
        bucket_limits = util.GEOM_DRUGS_BUCKET_LIMITS

    else:
        raise ValueError(f"Unknown dataset {args.dataset}")
 
    n_bond_types = util.get_n_bond_types(hparams["bond_strategy"])
    transform = partial(util.mol_transform, vocab=vocab, n_bonds=n_bond_types, coord_std=coord_std)

    if args.dataset_split == "train":
        dataset_path = Path(args.data_path) / "train.smol"
    elif args.dataset_split == "val":
        dataset_path = Path(args.data_path) / "val.smol"
    elif args.dataset_split == "test":
        dataset_path = Path(args.data_path) / "test.smol"

    dataset = GeometricDataset.load(dataset_path, transform=transform)
    dataset = dataset.sample(args.n_molecules, replacement=True)

    type_mask_index = vocab.indices_from_tokens(["<MASK>"])[0] if hparams["val-type-interpolation"] == "unmask" else None
    bond_mask_index = util.BOND_MASK_INDEX if hparams["val-bond-interpolation"] == "unmask" else None

    prior_sampler = GeometricNoiseSampler(
        vocab.size,
        n_bond_types,
        coord_noise="gaussian",
        type_noise=hparams["val-prior-type-noise"],
        bond_noise=hparams["val-prior-bond-noise"],
        scale_ot=hparams["val-prior-noise-scale-ot"],
        zero_com=True,
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index
    )
    eval_interpolant = GeometricInterpolant(
        prior_sampler,
        coord_interpolation="linear",
        type_interpolation=hparams["val-type-interpolation"],
        bond_interpolation=hparams["val-bond-interpolation"],
        equivariant_ot=False,
        batch_ot=False,
        mask_times_factor = args.mask_times_factor,
        mask_rate_strategy = args.mask_rate_strategy,
    )
    dm = GeometricInterpolantDM(
        None,
        None,
        dataset,
        args.batch_cost,
        test_interpolant=eval_interpolant,
        bucket_limits=bucket_limits,
        bucket_cost_scale=args.bucket_cost_scale,
        pad_to_bucket=False,
        prior_sampler = prior_sampler
    )
    return dm


def dm_from_ckpt(args, vocab):
    checkpoint = torch.load(args.ckpt_path)
    hparams = checkpoint["hyper_parameters"]
    dm = build_dm(args, hparams, vocab)
    return dm


def generate_smol_mols(output, model):
    coords = output["coords"]
    atom_dists = output["atomics"]
    bond_dists = output["bonds"]
    charge_dists = output["charges"]
    masks = output["mask"]

    mols = model.builder.smol_from_tensors(coords, atom_dists, masks, bond_dists=bond_dists, charge_dists=charge_dists)
    return mols


def save_raw_smol(args, raw_outputs, model):
    # Generate GeometricMols and then combine into one GeometricMolBatch
    mol_lists = [generate_smol_mols(output, model) for output in raw_outputs]
    mols = [mol for mol_list in mol_lists for mol in mol_list]
    batch = GeometricMolBatch.from_list(mols)

    save_path = Path(args.save_dir) / args.save_file
    batch_bytes = batch.to_bytes()
    save_path.write_bytes(batch_bytes)


def save_rdkit_sdf(args, mols):
    path = os.path.join(args.save_dir, args.save_file) + ".sdf"
    writer = Chem.SDWriter(path)
    for m in mols:
        if m is not None:
            writer.write(m)
    writer.close()

def save_rdkit_sdf_with_metrics(args, mols, per_mol_metrics):
    """
    mols: List[rdkit.Chem.Mol]
    per_mol_metrics: List[Dict[str, float]]，和 mols
    """
    # print(len(mols), len(per_mol_metrics))
    # assert len(mols) == len(per_mol_metrics)
    path = os.path.join(args.save_dir, args.save_file) + ".sdf"
    writer = Chem.SDWriter(path)
    for m, metrics in zip(mols, per_mol_metrics):
        if m is None:
            continue
        for metric_name, value in metrics.items():
            m.SetDoubleProp(metric_name, float(value))
        writer.write(m)
    writer.close()

def main(args):
    print(f"Running prediction script for {args.n_molecules} molecules...")
    print(f"Using model stored at {args.ckpt_path}")

    L.seed_everything(12345)
    util.disable_lib_stdout()
    util.configure_fs()

    print("Building model vocab...")
    vocab = util.build_vocab()
    print("Vocab complete.")

    print("Loading datamodule...")
    dm = dm_from_ckpt(args, vocab)
    print("Datamodule complete.")

    print("Loading model...")
    model = load_model(args, vocab, dm)
    print("Model complete.")

    print("Initialising metrics...")
    metrics, _ = util.init_metrics(args.data_path, model, no_novelty=args.no_novelty)
    print("Metrics complete.")

    print("Running generation...")
    molecules, raw_outputs, stabilities = util.generate_molecules(model, dm, args.integration_steps, stabilities=True)
    print("Generation complete.")
    print(len(molecules))
    import semlaflow.util.rdkit as smolRD

    if args.save_per_mol_metrics:
        per_mol_metrics = list()
        for mol_i, mol in enumerate(molecules):
            result = util.calc_metrics_([mol], metrics)
            # util.print_results(result)
            per_mol_metrics.append(result)

    # print("Calculating generative metrics...")
    # results = util.calc_metrics_(molecules, metrics, mol_stabs=stabilities)
    # util.print_results(results)
    # print("Generation script complete!")


    print(f"Saving predictions to {args.save_dir}/{args.save_file}")
    # save_rdkit_sdf(args, molecules)
    if args.save_per_mol_metrics:
        save_rdkit_sdf_with_metrics(args, molecules, per_mol_metrics)
    else:
        save_rdkit_sdf(args, molecules)
    # save_raw_smol(args, raw_outputs, model)
    print("Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--save_file", type=str, default=DEFAULT_SAVE_FILE)
    parser.add_argument("--save_per_mol_metrics", action="store_true")

    parser.add_argument("--batch_cost", type=int, default=DEFAULT_BATCH_COST)
    parser.add_argument("--dataset_split", type=str, default=DEFAULT_DATASET_SPLIT)
    parser.add_argument("--n_molecules", type=int, default=DEFAULT_N_MOLECULES)
    parser.add_argument("--integration_steps", type=int, default=DEFAULT_INTEGRATION_STEPS)
    parser.add_argument("--cat_sampling_noise_level", type=float, default=DEFAULT_CAT_SAMPLING_NOISE_LEVEL)
    parser.add_argument("--ode_sampling_scheduler", type=str, default=DEFAULT_ODE_SAMPLING_SCHEDULER)
    parser.add_argument("--coord_noise_std_dev", type=float, default=DEFAULT_COORD_NOISE_STD_DEV)

    parser.add_argument("--bucket_cost_scale", type=str, default=DEFAULT_BUCKET_COST_SCALE)
    parser.add_argument("--max_sigma", type=float, default=DEFAULT_MAX_SIGMA)
    parser.add_argument("--min_sigma", type=float, default=DEFAULT_MIN_SIGMA)
    parser.add_argument("--use_dist_loss", action="store_true")
    # Allow overridding for EGNN arch since some models were not saved with a value for n_layers
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--mask_times_factor", type=float, default=DEFAULT_MASK_TIMES_FACTOR)
    parser.add_argument("--mask_rate_strategy", type=str, default=DEFAULT_MASK_RATE_STRATEGY)
    parser.add_argument("--use_edm_mask_step", action="store_true")
    parser.add_argument("--include_charge", action="store_true")    
    parser.add_argument("--first_term_coef", type=float, default=DEFAULT_FIRST_TERM_COEF)
    parser.add_argument("--adaptive_cat_noise_level", action="store_true")
    parser.add_argument("--no_novelty", action="store_true")
    parser.add_argument("--use_heun", action="store_true")
    parser.add_argument("--sampler", type=str, default=DEFAULT_SAMPLER)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--sampling_strategy_factor", type=float, default=DEFAULT_SAMPLING_STRATEGY_FACTOR)
    parser.add_argument("--low_confidence_remask", type=str, default=DEFAULT_LOW_CONFIDENCE_REMASK)
    args = parser.parse_args()
    main(args)