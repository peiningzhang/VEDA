import argparse
from functools import partial
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

import semlaflow.scriptutil as util
from semlaflow.data.datamodules import GeometricInterpolantDM
from semlaflow.data.datasets import GeometricDataset
from semlaflow.data.interpolate import GeometricInterpolant, GeometricNoiseSampler
from semlaflow.models.fm import Integrator, MolecularCFM
from semlaflow.models.semla import EquiInvDynamics, SemlaGenerator
import os
import wandb

os.environ["WANDB__SERVICE_WAIT"] = "300"

DEFAULT_DATASET = "geom-drugs"
DEFAULT_ARCH = "semla"

DEFAULT_D_MODEL = 384
DEFAULT_N_LAYERS = 12
DEFAULT_D_MESSAGE = 128
DEFAULT_D_EDGE = 128
DEFAULT_N_COORD_SETS = 64
DEFAULT_N_ATTN_HEADS = 32
DEFAULT_D_MESSAGE_HIDDEN = 128
DEFAULT_COORD_NORM = "length"
DEFAULT_SIZE_EMB = 64

DEFAULT_MAX_ATOMS = 256

DEFAULT_EPOCHS = 200
DEFAULT_LR = 0.0003
DEFAULT_BATCH_COST = 4096
DEFAULT_ACC_BATCHES = 1
DEFAULT_GRADIENT_CLIP_VAL = 1.0
DEFAULT_TYPE_LOSS_WEIGHT = 0.2
DEFAULT_BOND_LOSS_WEIGHT = 1.0
DEFAULT_CHARGE_LOSS_WEIGHT = 1.0
DEFAULT_CATEGORICAL_STRATEGY = "uniform-sample"
DEFAULT_LR_SCHEDULE = "constant"
DEFAULT_WARM_UP_STEPS = 10000
DEFAULT_BUCKET_COST_SCALE = "linear"
DEFAULT_CAT_SKIP_CONNECTION = None

DEFAULT_N_VALIDATION_MOLS = 2000
DEFAULT_VAL_CHECK_EPOCHS = 10
DEFAULT_NUM_INFERENCE_STEPS = 100
DEFAULT_CAT_SAMPLING_NOISE_LEVEL = 1
DEFAULT_COORD_NOISE_STD_DEV = 0.0
DEFAULT_TYPE_DIST_TEMP = 1.0
DEFAULT_MASK_TIMES_FACTOR = 1.0
DEFAULT_MASK_RATE_STRATEGY = None
DEFAULT_DIST_STRATEGY = "constant"
DEFAULT_DIST_LOSS_WEIGHT = 1.0
# DEFAULT_TIME_ALPHA = 2.0
# DEFAULT_TIME_BETA = 1.0
DEFAULT_TIME_MEAN = -1.26
DEFAULT_TIME_SIGMA = 1.41
DEFAULT_OPTIMAL_TRANSPORT = "equivariant"
DEFAULT_MAX_SIGMA = 80
DEFAULT_MIN_SIGMA = 0.001
DEFAULT_RHO = 2.0



def build_model(args, dm, vocab):
    # Get hyperparameeters from the datamodule, pass these into the model to be saved
    hparams = {
        "epochs": args.epochs,
        "gradient_clip_val": args.gradient_clip_val,
        "dataset": args.dataset,
        "precision": "32",
        "architecture": args.arch,
        "shortcut_training": args.shortcut_training,
        **dm.hparams,
    }

    # Add 1 for the time (0 <= t <= 1 for flow matching)
    n_atom_feats = vocab.size + 1
    n_bond_types = util.get_n_bond_types(args.categorical_strategy)

    if args.arch == "semla":
        dynamics = EquiInvDynamics(
            args.d_model,
            args.d_message,
            args.n_coord_sets,
            args.n_layers,
            n_attn_heads=args.n_attn_heads,
            d_message_hidden=args.d_message_hidden,
            d_edge=args.d_edge,
            bond_refine=True,
            self_cond=args.self_condition,
            coord_norm=args.coord_norm,
        )
        egnn_gen = SemlaGenerator(
            args.d_model,
            dynamics,
            vocab.size,
            n_atom_feats,
            d_edge=args.d_edge,
            n_edge_types=n_bond_types,
            self_cond=args.self_condition,
            size_emb=args.size_emb,
            max_atoms=args.max_atoms
        )

    elif args.arch == "eqgat":
        from semlaflow.models.eqgat import EqgatGenerator

        # Hardcode for now since we only need one model size
        d_model_eqgat = 256
        n_equi_feats_eqgat = 256
        n_layers_eqgat = 12
        d_edge_eqgat = 128

        egnn_gen = EqgatGenerator(
            d_model_eqgat, n_layers_eqgat, n_equi_feats_eqgat, vocab.size, n_atom_feats, d_edge_eqgat, n_bond_types
        )

    elif args.arch == "egnn":
        from semlaflow.models.egnn import VanillaEgnnGenerator

        egnn_gen = VanillaEgnnGenerator(
            args.d_model, args.n_layers, vocab.size, n_atom_feats, d_edge=args.d_edge, n_edge_types=n_bond_types
        )

    else:
        raise ValueError(f"Unknown architecture '{args.arch}'; known: `semla`, `eqgat` or `egnn`")

    if args.dataset == "qm9":
        coord_scale = util.QM9_COORDS_STD_DEV
    elif args.dataset == "geom-drugs":
        coord_scale = util.GEOM_COORDS_STD_DEV
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")

    type_mask_index = None
    bond_mask_index = None

    if args.categorical_strategy == "mask":
        type_mask_index = vocab.indices_from_tokens(["<MASK>"])[0]
        bond_mask_index = util.BOND_MASK_INDEX
        train_strategy = "mask"
        sampling_strategy = "mask"

    elif args.categorical_strategy == "uniform-sample":
        train_strategy = "ce"
        sampling_strategy = "uniform-sample"

    elif args.categorical_strategy == "dirichlet":
        train_strategy = "ce"
        sampling_strategy = "dirichlet"

    else:
        raise ValueError(
            f"Interpolation '{args.categorical_strategy}' is not supported. "
            + "Supported are: `mask`, `uniform-sample` and `dirichlet`"
        )
    if args.include_charge:
        charge_sampling_strategy = sampling_strategy
        charge_train_strategy = train_strategy
    else:
        charge_sampling_strategy = None
        charge_train_strategy = None
    train_steps = util.calc_train_steps(dm, args.epochs, args.acc_batches)
    train_smiles = None if args.trial_run else [mols.str_id for mols in dm.train_dataset]
    val_smiles = None if args.trial_run else [mols.str_id for mols in dm.val_dataset]

    print(f"Total training steps {train_steps}")

    integrator = Integrator(
        args.num_inference_steps,
        type_strategy=sampling_strategy,
        bond_strategy=sampling_strategy,
        charge_strategy=charge_sampling_strategy,
        coord_noise_std=args.coord_noise_std_dev,
        cat_noise_level=args.cat_sampling_noise_level,
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index,
        prior_sampler = dm.prior_sampler,
        data_module = dm,
        mask_times_factor = args.mask_times_factor,
        use_edm_mask_step = args.use_edm_mask_step,
        mask_rate_strategy = args.mask_rate_strategy,
        max_sigma=args.max_sigma,
        min_sigma=args.min_sigma,
        include_charge=args.include_charge,
    )

    fm_model = MolecularCFM(
        egnn_gen,
        vocab,
        args.lr,
        integrator,
        coord_scale=coord_scale,
        type_strategy=train_strategy,
        bond_strategy=train_strategy,
        charge_strategy=charge_train_strategy,
        type_loss_weight=args.type_loss_weight,
        bond_loss_weight=args.bond_loss_weight,
        charge_loss_weight=args.charge_loss_weight,
        use_dist_loss=args.use_dist_loss,
        dist_strategy=args.dist_strategy,
        dist_loss_weight=args.dist_loss_weight,
        use_fm_coord_loss=args.use_fm_coord_loss,
        pairwise_metrics=False,
        use_ema=args.use_ema,
        compile_model=False,
        self_condition=args.self_condition,
        distill=False,
        lr_schedule=args.lr_schedule,
        warm_up_steps=args.warm_up_steps,
        total_steps=train_steps,
        train_smiles=train_smiles,
        val_smiles=val_smiles,
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index,
        max_sigma=args.max_sigma,
        min_sigma=args.min_sigma,
        rho=args.rho,
        include_charge=args.include_charge,
        cat_skip_connection=args.cat_skip_connection,
        use_cat_time_based_weight=args.use_cat_time_based_weight,
        use_x_pred = args.use_x_pred,
        x_pred_type = args.x_pred_type,
        **hparams,
    )
    return fm_model


def build_dm(args, vocab):
    if args.dataset == "qm9":
        coord_std = util.QM9_COORDS_STD_DEV
        padded_sizes = util.QM9_BUCKET_LIMITS

    elif args.dataset == "geom-drugs":
        coord_std = util.GEOM_COORDS_STD_DEV
        padded_sizes = util.GEOM_DRUGS_BUCKET_LIMITS

    else:
        raise ValueError(f"Unknown dataset {args.dataset}. Available datasets are `qm9` and `geom-drugs`.")

    data_path = Path(args.data_path)

    n_bond_types = util.get_n_bond_types(args.categorical_strategy)
    transform = partial(util.mol_transform, vocab=vocab, n_bonds=n_bond_types, coord_std=coord_std)

    # Load generated dataset with different transform fn if we are distilling a model
    # if args.distill:
    #     distill_transform = partial(util.distill_transform, coord_std=coord_std)
    #     train_dataset = GeometricDataset.load(data_path / "distill.smol", transform=distill_transform)
    # else:
    #     train_dataset = GeometricDataset.load(data_path / "train.smol", transform=transform)

    train_dataset = GeometricDataset.load(data_path / "train.smol", transform=transform, test_run=args.test_run)
    val_dataset = GeometricDataset.load(data_path / "val.smol", transform=transform, test_run=args.test_run)
    val_dataset = val_dataset.sample(args.n_validation_mols)

    type_mask_index = None
    bond_mask_index = None
    charge_mask_index = None

    if args.categorical_strategy == "mask":
        type_mask_index = vocab.indices_from_tokens(["<MASK>"])[0]
        bond_mask_index = util.BOND_MASK_INDEX
        categorical_interpolation = "unmask"
        categorical_noise = "mask"

    elif args.categorical_strategy == "uniform-sample":
        categorical_interpolation = "unmask"
        categorical_noise = "uniform-sample"


    elif args.categorical_strategy == "dirichlet":
        categorical_interpolation = "dirichlet"
        categorical_noise = "uniform-dist"

    else:
        raise ValueError(
            f"Interpolation '{args.categorical_strategy}' is not supported. "
            + "Supported are: `mask`, `uniform-sample` and `dirichlet`"
        )
    if args.include_charge:
        charge_categorical_interpolation = categorical_interpolation
        charge_categorical_noise = categorical_noise
    else:
        charge_categorical_interpolation = None
        charge_categorical_noise = None
    scale_ot = False
    batch_ot = False
    equivariant_ot = False

    if args.optimal_transport == "batch":
        batch_ot = True
    elif args.optimal_transport == "equivariant":
        equivariant_ot = True
    elif args.optimal_transport == "scale":
        scale_ot = True
        equivariant_ot = True
    elif args.optimal_transport not in ["None", "none", None]:
        raise ValueError(
            f"Unknown value for optimal_transport '{args.optimal_transport}'. "
            + "Acceted values: `batch`, `equivariant` and `scale`."
        )

    # train_fixed_time = 0.5 if args.distill else None
    train_fixed_time = None

    prior_sampler = GeometricNoiseSampler(
        vocab.size,
        n_bond_types,
        coord_noise="gaussian",
        type_noise=categorical_noise,
        bond_noise=categorical_noise,
        charge_noise= charge_categorical_noise,
        scale_ot=scale_ot,
        zero_com=True,
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index,
        charge_mask_index=charge_mask_index,
        include_charge=args.include_charge,
    )
    train_interpolant = GeometricInterpolant(
        prior_sampler,
        coord_interpolation="linear",
        type_interpolation=categorical_interpolation,
        bond_interpolation=categorical_interpolation,
        charge_interpolation=charge_categorical_interpolation,
        coord_noise_std=args.coord_noise_std_dev,
        type_dist_temp=args.type_dist_temp,
        equivariant_ot=equivariant_ot,
        batch_ot=batch_ot,
        time_mean=args.time_mean,
        time_sigma=args.time_sigma,
        fixed_time=train_fixed_time,
        mask_times_factor = args.mask_times_factor,
        mask_rate_strategy = args.mask_rate_strategy,
        include_charge=args.include_charge,
    )
    eval_interpolant = GeometricInterpolant(
        prior_sampler,
        coord_interpolation="linear",
        type_interpolation=categorical_interpolation,
        bond_interpolation=categorical_interpolation,
        charge_interpolation=charge_categorical_interpolation,
        equivariant_ot=False,
        batch_ot=False,
        fixed_time=0.5,
        mask_times_factor = args.mask_times_factor,
        include_charge=args.include_charge,
    )

    dm = GeometricInterpolantDM(
        train_dataset,
        val_dataset,
        None,
        args.batch_cost,
        train_interpolant=train_interpolant,
        val_interpolant=eval_interpolant,
        test_interpolant=None,
        bucket_limits=padded_sizes,
        bucket_cost_scale=args.bucket_cost_scale,
        pad_to_bucket=False,
        prior_sampler = prior_sampler,
        
    )
    return dm


def build_trainer(args):
    epochs = 1 if args.trial_run else args.epochs
    log_steps = 1 if args.trial_run else 50
    val_check_epochs = 1 if args.trial_run else args.val_check_epochs

    project_name = f"{util.PROJECT_PREFIX}-{args.dataset}"
    print("Using precision '32'")
    if args.ckpt_path is not None:
        logger = WandbLogger(project=project_name,save_dir="wandb", log_model=True, id=args.wandb_id,resume='allow')
    else:
        logger = WandbLogger(project=project_name, save_dir="wandb", log_model=True)
    
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpointing = ModelCheckpoint(every_n_epochs=val_check_epochs, monitor="val-validity", mode="max", save_last=True, 
                                    save_top_k=1,
                                    filename="best-{epoch:02d}-{val-validity:.4f}")

    # No logger if doing a trial run
    logger = None if args.trial_run else logger

    trainer = L.Trainer(
        min_epochs=epochs,
        max_epochs=epochs,
        logger=logger,
        log_every_n_steps=log_steps,
        accumulate_grad_batches=args.acc_batches,
        gradient_clip_val=args.gradient_clip_val,
        check_val_every_n_epoch=val_check_epochs,
        callbacks=[lr_monitor, checkpointing],
        precision="32",
    )
    return trainer


def main(args):
    # Set some useful torch properties
    # Float32 precision should only affect computation on A100 and should in theory be a lot faster than the default
    # Increasing the cache size is required since the model will be compiled seperately for each bucket
    torch.set_float32_matmul_precision("high")
    # torch._dynamo.config.cache_size_limit = util.COMPILER_CACHE_SIZE
    # print(f"Set torch compiler cache size to {torch._dynamo.config.cache_size_limit}")

    L.seed_everything(12345)
    util.disable_lib_stdout()
    util.configure_fs()

    print("Building model vocab...")
    vocab = util.build_vocab()
    print("Vocab complete.")

    print("Loading datamodule...")
    dm = build_dm(args, vocab)
    print("Datamodule complete.")

    print("Building equinv model...")
    model = build_model(args, dm, vocab)
    print("Model complete.")

    # if args.ckpt_path is not None:
    #     from semlaflow.evaluate import load_model
    #     print(f"Loading model...")
    #     model = load_model(args, vocab, dm)
    #     print("Model complete.")

    trainer = build_trainer(args)

    print("Fitting datamodule to model...")
    if args.ckpt_path is not None:
        trainer.fit(model, datamodule=dm, ckpt_path=args.ckpt_path)
    else:
        trainer.fit(model, datamodule=dm)
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Setup args
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--trial_run", action="store_true")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--wandb_id", type=str, default=None)

    # Model args
    parser.add_argument("--d_model", type=int, default=DEFAULT_D_MODEL)
    parser.add_argument("--n_layers", type=int, default=DEFAULT_N_LAYERS)
    parser.add_argument("--d_message", type=int, default=DEFAULT_D_MESSAGE)
    parser.add_argument("--d_edge", type=int, default=DEFAULT_D_EDGE)
    parser.add_argument("--n_coord_sets", type=int, default=DEFAULT_N_COORD_SETS)
    parser.add_argument("--n_attn_heads", type=int, default=DEFAULT_N_ATTN_HEADS)
    parser.add_argument("--d_message_hidden", type=int, default=DEFAULT_D_MESSAGE_HIDDEN)
    parser.add_argument("--coord_norm", type=str, default=DEFAULT_COORD_NORM)
    parser.add_argument("--size_emb", type=int, default=DEFAULT_SIZE_EMB)
    parser.add_argument("--max_atoms", type=int, default=DEFAULT_MAX_ATOMS)
    parser.add_argument("--arch", type=str, default=DEFAULT_ARCH)

    # Training args
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--batch_cost", type=int, default=DEFAULT_BATCH_COST)
    parser.add_argument("--acc_batches", type=int, default=DEFAULT_ACC_BATCHES)
    parser.add_argument("--gradient_clip_val", type=float, default=DEFAULT_GRADIENT_CLIP_VAL)
    parser.add_argument("--type_loss_weight", type=float, default=DEFAULT_TYPE_LOSS_WEIGHT)
    parser.add_argument("--bond_loss_weight", type=float, default=DEFAULT_BOND_LOSS_WEIGHT)
    parser.add_argument("--charge_loss_weight", type=float, default=DEFAULT_CHARGE_LOSS_WEIGHT)
    parser.add_argument("--use_dist_loss", action="store_true")
    parser.add_argument("--dist_strategy", type=str, default=DEFAULT_DIST_STRATEGY)
    parser.add_argument("--dist_loss_weight", type=float, default=DEFAULT_DIST_LOSS_WEIGHT)
    parser.add_argument("--categorical_strategy", type=str, default=DEFAULT_CATEGORICAL_STRATEGY)
    parser.add_argument("--lr_schedule", type=str, default=DEFAULT_LR_SCHEDULE)
    parser.add_argument("--warm_up_steps", type=int, default=DEFAULT_WARM_UP_STEPS)
    parser.add_argument("--bucket_cost_scale", type=str, default=DEFAULT_BUCKET_COST_SCALE)
    parser.add_argument("--no_ema", action="store_false", dest="use_ema")
    parser.add_argument("--self_condition", action="store_true")
    parser.add_argument("--shortcut_training", action="store_true")
    parser.add_argument("--test_run", action="store_true")    
    parser.add_argument("--mask_rate_strategy", type=str, default=DEFAULT_MASK_RATE_STRATEGY)
    parser.add_argument("--use_edm_mask_step", action="store_true")
    parser.add_argument("--include_charge", action="store_true")
    parser.add_argument("--cat_skip_connection", type=str, default=DEFAULT_CAT_SKIP_CONNECTION)
    parser.add_argument("--use_cat_time_based_weight", action="store_true")
    parser.add_argument("--use_fm_coord_loss", action="store_true")

    # Diffusion and sampling args
    parser.add_argument("--val_check_epochs", type=int, default=DEFAULT_VAL_CHECK_EPOCHS)
    parser.add_argument("--n_validation_mols", type=int, default=DEFAULT_N_VALIDATION_MOLS)
    parser.add_argument("--num_inference_steps", type=int, default=DEFAULT_NUM_INFERENCE_STEPS)
    parser.add_argument("--cat_sampling_noise_level", type=int, default=DEFAULT_CAT_SAMPLING_NOISE_LEVEL)
    parser.add_argument("--coord_noise_std_dev", type=float, default=DEFAULT_COORD_NOISE_STD_DEV)
    parser.add_argument("--type_dist_temp", type=float, default=DEFAULT_TYPE_DIST_TEMP)
    parser.add_argument("--mask_times_factor", type=float, default=DEFAULT_MASK_TIMES_FACTOR)
    parser.add_argument("--use_x_pred", action="store_true")
    parser.add_argument("--x_pred_type", type=str, default='v1')

    # parser.add_argument("--time_alpha", type=float, default=DEFAULT_TIME_ALPHA)
    # parser.add_argument("--time_beta", type=float, default=DEFAULT_TIME_BETA)
    parser.add_argument("--time_mean", type=float, default=DEFAULT_TIME_MEAN)
    parser.add_argument("--time_sigma", type=float, default=DEFAULT_TIME_SIGMA)
    parser.add_argument("--optimal_transport", type=str, default=DEFAULT_OPTIMAL_TRANSPORT)
    parser.add_argument("--max_sigma", type=float, default=DEFAULT_MAX_SIGMA)
    parser.add_argument("--min_sigma", type=float, default=DEFAULT_MIN_SIGMA)
    parser.add_argument("--rho", type=float, default=DEFAULT_RHO)
    

    parser.set_defaults(
        trial_run=False,
        use_ema=True,
        self_condition=True,
        include_charge=False,
        # compile_model=False,
        # mixed_precision=False,
        # distill=False
    )

    args = parser.parse_args()
    print(args)
    main(args)
