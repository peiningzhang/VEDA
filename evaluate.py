import argparse
from functools import partial
from pathlib import Path

import lightning as L
import numpy as np
import torch

import semlaflow.scriptutil as util
from semlaflow.data.datamodules import GeometricInterpolantDM
from semlaflow.data.datasets import GeometricDataset
from semlaflow.data.interpolate import GeometricInterpolant, GeometricNoiseSampler
# from semlaflow.models.fm import Integrator, MolecularCFM
from semlaflow.models.temp_fm import Integrator, MolecularCFM
from semlaflow.models.semla import EquiInvDynamics, SemlaGenerator

# Default script arguments
DEFAULT_DATASET_SPLIT = "test"
DEFAULT_N_MOLECULES = 1000
DEFAULT_N_REPLICATES = 1
DEFAULT_BATCH_COST = 8192
DEFAULT_BUCKET_COST_SCALE = "linear"
DEFAULT_INTEGRATION_STEPS = 100
DEFAULT_CAT_SAMPLING_NOISE_LEVEL = 1
DEFAULT_ODE_SAMPLING_SCHEDULER = "arcsin"
DEFAULT_COORD_NOISE_STD_DEV = 0.0
DEFAULT_MAX_SIGMA = 80
DEFAULT_MIN_SIGMA = 0.001
DEFAULT_MASK_TIMES_FACTOR = 1.0
DEFAULT_MASK_RATE_STRATEGY = 'log_uniform'
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
            hparams["n_edge_types"],
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
            n_edge_types=n_bond_types,
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
        bond_mask_index=bond_mask_index,
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

import time
def evaluate(args, model, dm, metrics, stab_metrics):
    if args.use_heun:
        args.integration_steps = args.integration_steps//2+1
    results_list = []
    for replicate_index in range(args.n_replicates):
        print(f"Running replicate {replicate_index + 1} out of {args.n_replicates}")
        time_start = time.time()
        molecules, _, stabilities = util.generate_molecules(
            model, dm, args.integration_steps, stabilities=True
        )
        time_end = time.time()
        print(f"Time taken: {time_end - time_start} seconds")
        print("Calculating metrics...")
        results = util.calc_metrics_(molecules, metrics, stab_metrics=stab_metrics, mol_stabs=stabilities)
        results_list.append(results)

    results_dict = {key: [] for key in results_list[0].keys()}
    for results in results_list:
        for metric, value in results.items():
            results_dict[metric].append(value.item())

    mean_results = {metric: np.mean(values) for metric, values in results_dict.items()}
    std_results = {metric: np.std(values) for metric, values in results_dict.items()}

    return mean_results, std_results, results_dict


def main(args):
    print(f"Running evaluation script for {args.n_replicates} replicates with {args.n_molecules} molecules each...")
    print(f"Using model stored at {args.ckpt_path}")

    if args.n_replicates < 1:
        raise ValueError("n_replicates must be at least 1.")

    L.seed_everything(12345)
    util.disable_lib_stdout()
    util.configure_fs()

    print("Building model vocab...")
    vocab = util.build_vocab()
    print("Vocab complete.")

    print("Loading datamodule...")
    dm = dm_from_ckpt(args, vocab)
    print("Datamodule complete.")

    print(f"Loading model...")
    model = load_model(args, vocab, dm)
    print("Model complete.")

    print("Initialising metrics...")
    metrics, stab_metrics = util.init_metrics(args.data_path, model, no_novelty=args.no_novelty)
    print("Metrics complete.")

    print("Running evaluation...")
    avg_results, std_results, list_results = evaluate(args, model, dm, metrics, stab_metrics)
    print("Evaluation complete.")

    util.print_results(avg_results, std_results=std_results)

    print("All replicate results...")
    print(f"{'Metric':<22}Result")
    print("-" * 30)

    for metric, results_list in list_results.items():
        print(f"{metric:<22}{results_list}")
    print()

    metrics_str = "Results_metrics in one line: "
    result_str = "Results_number in one line: "
    for metric, results_list in list_results.items():
        metrics_str += f"{metric:<22}\t"
        result_str += f"{results_list[0]:.5f}\t "        
    print(metrics_str)
    print(result_str)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--dataset", type=str)

    parser.add_argument("--batch_cost", type=int, default=DEFAULT_BATCH_COST)
    parser.add_argument("--dataset_split", type=str, default=DEFAULT_DATASET_SPLIT)
    parser.add_argument("--n_molecules", type=int, default=DEFAULT_N_MOLECULES)
    parser.add_argument("--n_replicates", type=int, default=DEFAULT_N_REPLICATES)
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