from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Optional

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, OneCycleLR
from torchmetrics import MetricCollection

from util import functional as smolF
from util import metrics as Metrics
from util import rdkit as smolRD

from models.semla import MolecularGenerator
from util.molrepr import GeometricMol
from util.tokeniser import Vocabulary
import math
from util.molrepr import GeometricMolBatch
import random
import time
_T = torch.Tensor
_BatchT = dict[str, _T]

class Integrator:
    def __init__(
        self,
        steps,
        coord_noise_std=0.0,
        type_strategy="mask",
        bond_strategy="mask",
        cat_noise_level=0,
        type_mask_index=None,
        bond_mask_index=None,
        eps=1e-8,
        prior_sampler = None,
        data_module = None,
        # Note: mask_times_factor parameter removed - was used to scale time for mask rate calculation
        # but is no longer needed. Mask rate is now computed directly from time without scaling.
        mask_rate_strategy = None,
        max_sigma: float = 80,
        min_sigma: float = 0.001,
        adaptive_cat_noise_level = False,
        sampler = 'euler',
    ):

        self._check_cat_sampling_strategy(type_strategy, type_mask_index, "type")
        self._check_cat_sampling_strategy(bond_strategy, bond_mask_index, "bond")
        assert type_strategy == bond_strategy, "type_strategy and bond_strategy must be the same"

        self.steps = steps
        self.coord_noise_std = coord_noise_std
        self.type_strategy = type_strategy
        self.bond_strategy = bond_strategy
        self.cat_noise_level = cat_noise_level
        self.type_mask_index = type_mask_index
        self.bond_mask_index = bond_mask_index
        self.eps = eps
        self.prior_sampler = prior_sampler
        self.data_module = data_module
        # mask_times_factor removed - see comment above
        self.mask_rate_strategy = mask_rate_strategy
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.adaptive_cat_noise_level = adaptive_cat_noise_level
        self.sampler = sampler
        if self.mask_rate_strategy == 'log_uniform':
            self.time_mean = np.log((self.min_sigma*self.max_sigma)**0.5)
            self.time_sigma = (np.log(self.max_sigma) - np.log(self.min_sigma))/8
    @property
    def hparams(self):
        return {
            "integration-steps": self.steps,
            "integration-coord-noise-std": self.coord_noise_std,
            "integration-type-strategy": self.type_strategy,
            "integration-bond-strategy": self.bond_strategy,
            "integration-cat-noise-level": self.cat_noise_level,
            "integration-mask-rate-strategy": self.mask_rate_strategy,
            "integration-max-sigma": self.max_sigma,
            "integration-min-sigma": self.min_sigma,
        }
    def cat_step(self, curr, predicted, t, step_size):
        for key in ["atomics", "bonds", "charges"]:
            velocity = (predicted[key] - curr[key]) / t[0].item()#/(t.view(-1, 1, 1)+1+step_size)
            curr[key] = curr[key] + step_size * velocity
        return curr

    def step(self, curr: _BatchT, predicted: _BatchT, prior: _BatchT, t: _T, step_size: float) -> _BatchT:



        # *** Coord update step ***
        coord_velocity = (predicted["coords"] - curr["coords"]) / t.view(-1, 1, 1)
        coords = curr["coords"] + step_size * coord_velocity
        # *** Atom type update step ***
        # Note: mask_times_factor scaling removed - time is used directly without scaling

        if self.mask_rate_strategy in ['edm', 'log_uniform']:
            if t[0].item() > (self.max_sigma-self.eps):
                inf = (1/self.eps)
                step_size += t[0].item()*inf
                t = t*(inf+1)
            if self.type_strategy == 'uniform-sample':
                atomics, atomics_probs = self._uniform_sample_step(curr["atomics"], predicted["atomics"], t, step_size, self.mask_rate_strategy)
                bonds, bonds_probs = self._uniform_sample_step(curr["bonds"], predicted["bonds"], t, step_size, self.mask_rate_strategy)
            elif self.type_strategy == 'mask':
                atomics, atomics_probs = self._mask_sampling_step(curr["atomics"], predicted["atomics"], t, step_size, self.mask_rate_strategy, self.type_mask_index)
                bonds, bonds_probs = self._mask_sampling_step(curr["bonds"], predicted["bonds"], t, step_size, self.mask_rate_strategy, self.bond_mask_index)
            else:
                raise ValueError("Mask rate strategy not implemented")
        elif self.sampler == 'dfm-pc':
            atomics, atomics_probs = self._dfm_gat_sampler_step(curr["atomics"], predicted["atomics"], t, step_size)
            bonds, bonds_probs = self._dfm_gat_sampler_step(curr["bonds"], predicted["bonds"], t, step_size)
        else:
            raise ValueError("Mask rate strategy not implemented")


        updated = {
            "coords": coords,
            "atomics": atomics,
            "bonds": bonds,
            "mask": curr["mask"]
        }
        updated["atomics_logits"] = atomics_probs.log()
        updated["bonds_logits"] = bonds_probs.log()
        return updated
    def _mask_sampling_step(self, curr_dist, pred_dist, t, step_size, mask_rate_strategy, mask_index):
        n_categories = pred_dist.size(-1)
        original_mask_rates, mask_rates, _ = self._compute_mask_rate_deltas(t, t-step_size)
        
        step_size = original_mask_rates - mask_rates
        t = torch.tensor(1 - original_mask_rates)
        
        pred = torch.distributions.Categorical(pred_dist).sample()
        curr = torch.argmax(curr_dist, dim=-1)

        ones = [1] * (len(pred.shape) - 1)
        times = t.view(-1, *ones)

        # Choose elements to unmask
        limit = (step_size * (1 + (self.cat_noise_level * times)) / (1 - times)).to(pred.device)
        unmask = torch.rand_like(pred.float()) < limit
        unmask = unmask * (curr == mask_index)

        # Choose elements to mask
        mask = torch.rand_like(pred.float()) < step_size * self.cat_noise_level
        mask = mask * (curr != self.type_mask_index)
        mask[t + step_size >= 1.0] = 0.0

        # Applying unmasking and re-masking
        curr[unmask] = pred[unmask]
        curr[mask] = mask_index

        return smolF.one_hot_encode_tensor(curr, n_categories), pred_dist

    def _uniform_sample_step(self, curr_dist, pred_dist, t, step_size, mask_rate_strategy):
        n_categories = pred_dist.size(-1)

        curr = torch.argmax(curr_dist, dim=-1).unsqueeze(-1)
        pred_probs_curr = torch.gather(pred_dist, -1, curr)
        if mask_rate_strategy == 'log_uniform':
            t = t.clamp(min=self.min_sigma, max=self.max_sigma)
        # Setup batched time tensor and noise tensor
        ones = [1] * (len(pred_dist.shape) - 1)
        times = t.view(-1, *ones)
        noise = torch.zeros_like(times)
        noise[times - step_size > 0] = self.cat_noise_level
        # noise /= (t[0].item()+1)**2
        if self.adaptive_cat_noise_level:
            if self.mask_rate_strategy == 'edm':
                noise /= (t[0].item()+1)**2
            elif self.mask_rate_strategy == 'log_uniform':
                noise /= (t[0].item()*(np.log(self.max_sigma) - np.log(self.min_sigma)))
            else:
                raise ValueError("uniform Mask rate strategy not implemented")


        # Off-diagonal step probs
        if mask_rate_strategy == 'edm':
            mult = ((1 + (noise * (n_categories + times) * (times+1))) / (times)/(times+1))
        elif mask_rate_strategy == 'log_uniform':
            mult = ((1 + (noise * (n_categories*np.log(self.max_sigma) - (n_categories - 1)*torch.log(times) - np.log(self.min_sigma)))) / (times)/(torch.log(times) - np.log(self.min_sigma)))
        else:
            raise ValueError("uniform Mask rate strategy not implemented")
        
        
        first_term = step_size * mult * pred_dist
        second_term = step_size * noise * pred_probs_curr
        step_probs = (first_term + second_term).clamp(max=1.0)

        # On-diagonal step probs
        # print(step_probs[0,0], curr[0,0])
        step_probs.scatter_(-1, curr, 0.0)
        diags = (1.0 - step_probs.sum(dim=-1, keepdim=True)).clamp(min=0.0)
        step_probs.scatter_(-1, curr, diags)

        # Sample and convert back to one-hot so that all strategies represent data the same way
        # print(step_probs[0,0], diags[0,0])
        step_probs = step_probs.clamp(min=1e-10)
        step_probs /= step_probs.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        step_probs = step_probs / step_probs.sum(dim=-1, keepdim=True)
        samples = torch.distributions.Categorical(step_probs).sample()
        return smolF.one_hot_encode_tensor(samples, n_categories), step_probs

    def _check_cat_sampling_strategy(self, strategy, mask_index, name):
        if strategy not in ["linear", "dirichlet", "mask", "uniform-sample"]:
            raise ValueError(f"{name} sampling strategy '{strategy}' is not supported.")

        if strategy == "mask" and mask_index is None:
            raise ValueError(f"{name}_mask_index must be provided if using the mask sampling strategy.")
    def _compute_mask_rate_deltas(self, times, times_hat):
        """Compute mask rate changes - adapted from _compute_mask_rates in training
        
        Note on 'edm' strategy naming:
        Although named 'edm', the formula mask_rate = t/(t+1) actually comes from 
        Discrete Flow Matching (DFM) mask prior derivation:
        
        In DFM, mask prior is given as:
            f(t) = Cat(δ(x₀), w_data, w_noise)
        
        Define weight ratio: t = w_noise / w_data
        Then total weight: w_total = w_data + w_noise = w_data(1 + t)
        
        Mask rate (probability of selecting noise/mask) is:
            mask_rate = w_noise / (w_data + w_noise)
                     = w_noise / [w_data(1 + t)]
                     = (t·w_data) / [(1 + t)·w_data]
                     = t / (1 + t)
        
        This maps the weight ratio t ∈ [0, ∞) to mask_rate ∈ [0, 1):
        - When t → 0 (low noise weight): mask_rate → 0 (mostly keep original data)
        - When t → ∞ (high noise weight): mask_rate → 1 (mostly mask/noise)
        """

        if self.mask_rate_strategy == 'log_uniform':
            # Note: mask_times_factor removed - log_uniform strategy now always uses unscaled time
            original_mask_rates = (np.log(times.cpu()) - (self.time_mean - self.time_sigma*4))/(self.time_sigma*8)
            mask_rates = (np.log(times_hat.cpu()) - (self.time_mean - self.time_sigma*4))/(self.time_sigma*8)
        elif self.mask_rate_strategy == 'edm':
            # DFM mask rate formula: mask_rate = t / (t + 1)
            # Maps sigma time t ∈ [0, ∞) to mask_rate ∈ [0, 1)
            # Note: mask_times_factor removed - time is used directly without scaling
            original_mask_rates = times / (times + 1)
            mask_rates = times_hat / (times_hat + 1)
        else:
            raise ValueError("mask_rate_strategy not implemented")
        
        delta_mask_rates = (mask_rates - original_mask_rates) / (1 - original_mask_rates)
        
        return (float(original_mask_rates[0].item()), 
                float(mask_rates[0].item()), 
                float(delta_mask_rates[0].item()))
    def _compute_mask_rates(self, times):
        """Compute mask rates based on strategy
        
        For 'edm' strategy: mask_rate = t/(t+1) from DFM mask prior derivation.
        See _compute_mask_rate_deltas docstring for detailed explanation.
        """
        if self.mask_rate_strategy == 'log_uniform':
            # Extract first element if times is a tensor with multiple elements
            # Similar to _compute_mask_rate_deltas, extract scalar value first
            t = times[0].item() if isinstance(times, torch.Tensor) else times[0] if hasattr(times, '__getitem__') else times
            mask_rates = float((np.log(t) - (self.time_mean - self.time_sigma*4))/(self.time_sigma*8))
        else:  # 'edm' or None - uses DFM formula: mask_rate = t/(t+1)
            mask_rates = (times/(times + 1))[0].item()
        mask_rates = np.clip(mask_rates, self.eps, 1 - self.eps)
        return mask_rates
    
    def _dfm_gat_sampler_step(self, curr_dist, pred_dist, t, step_size):
        """
        A Predictor-Corrector sampler step based on Gat et al.'s "Discrete Flow Matching" paper.
        This function can be used as a drop-in replacement for `_uniform_sample_step`.
        
        Note: This method is from Gat et al.'s Discrete Flow Matching paper, distinct from 
        Campbell et al.'s Discrete Flow Matching paper (which uses a different approach).

        Args:
            curr_dist (torch.Tensor): The current one-hot encoded state.
            pred_dist (torch.Tensor): The model's prediction (probabilities) for the clean data, given `curr_dist` and `t`.
            t (torch.Tensor): The current native time (sigma).
            step_size (float): The size of the time step (t - t_next), always positive.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the new one-hot sample and the final transition probabilities.
        """
        n_categories = pred_dist.size(-1)
        
        # Convert from sigma time to normalized time following the reference logic
        eta = self.cat_noise_level
        original_mask_rates, mask_rates, _ = self._compute_mask_rate_deltas(t, t-step_size)
        t_normalized = 1 - original_mask_rates
        # Calculate alpha_t and beta_t as in reference
        alpha_t = eta * t_normalized**0.25 * (1-t_normalized)**0.25 + 1
        beta_t = alpha_t - 1
        
        
        # Calculate kappa terms
        # kappa_t = t_normalized**2
        # kappa_dot_t = 2 * t_normalized
        kappa_t = t_normalized
        kappa_dot_t = 1
        kappa_t_safe = max(kappa_t, 1e-9)
        one_minus_kappa_t_safe = max(1 - kappa_t, 1e-9)
        
        # Calculate scaling factors
        scale_hat = kappa_dot_t / one_minus_kappa_t_safe
        scale_tilde = kappa_dot_t / kappa_t_safe
        
        # print('t', t_normalized, 'alpha_t', alpha_t, 'scale_hat', scale_hat, 'scale_tilde', scale_tilde)
        
        # Calculate velocity components following reference logic
        u_hat_t = scale_hat * (pred_dist - curr_dist)
        
        # Prior noise distribution (uniform)
        prior_noise_dist = torch.full_like(pred_dist, 1.0 / n_categories)
        u_tilde_t = scale_tilde * (curr_dist - prior_noise_dist)
        
        # Combine velocities
        u_bar_t = (alpha_t * u_hat_t - beta_t * u_tilde_t)
        
        
        t_delta = original_mask_rates - mask_rates

        step_probs= curr_dist + t_delta * u_bar_t
        step_probs = torch.clamp(step_probs, min=1e-9, max=1)
                
        
        # Sample from the transition probabilities
        samples_ids = torch.distributions.Categorical(probs=step_probs).sample()
        final_sample_one_hot = F.one_hot(samples_ids, num_classes=n_categories).float()
        
        return final_sample_one_hot, step_probs

class MolBuilder:
    def __init__(self, vocab, n_workers=16):
        self.vocab = vocab
        self.n_workers = n_workers
        self._executor = None

    def shutdown(self):
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None

    def _startup(self):
        if self._executor is None:
            self._executor = ThreadPoolExecutor(self.n_workers)

    def mols_from_smiles(self, smiles, explicit_hs=False):
        self._startup()
        futures = [self._executor.submit(smolRD.mol_from_smiles, smi, explicit_hs) for smi in smiles]
        mols = [future.result() for future in futures]
        self.shutdown()
        return mols

    def mols_from_tensors(self, coords, atom_dists, mask, bond_dists=None, charge_dists=None, sanitise=True):
        extracted = self._extract_mols(
            coords,
            atom_dists,
            mask,
            bond_dists=bond_dists,
            charge_dists=charge_dists
        )

        self._startup()
        build_fn = partial(self._mol_from_tensors, sanitise=sanitise)
        futures = [self._executor.submit(build_fn, *items) for items in extracted]
        mols = [future.result() for future in futures]
        self.shutdown()

        return mols

    # TODO move into from_tensors method of GeometricMolBatch
    def smol_from_tensors(self, coords, atom_dists, mask, bond_dists, charge_dists):
        extracted = self._extract_mols(
            coords,
            atom_dists,
            mask,
            bond_dists=bond_dists,
            charge_dists=charge_dists
        )

        # mol_dicts = {}
        # for mol_coords, atom_dist, bond_dist, charge_dist in extracted:
        #     mol = {
        #         "coords": mol_coords,
        #         "atomics": atom_dist,
        #         "bonds": bond_dist,
        #         "charges": charge_dist
        #     }
        #     mol_dicts.append(mol)

        self._startup()
        build_fn = partial(self._smol_from_tensors)
        futures = [self._executor.submit(build_fn, *items) for items in extracted]
        smol_mols = [future.result() for future in futures]
        self.shutdown()

        return smol_mols

    def _mol_from_tensors(self, coords, atom_dists, bond_dists=None, charge_dists=None, sanitise=True):
        tokens = self._mol_extract_atomics(atom_dists)
        bonds = self._mol_extract_bonds(bond_dists) if bond_dists is not None else None
        charges = self._mol_extract_charges(charge_dists) if charge_dists is not None else None
        return smolRD.mol_from_atoms(coords.numpy(), tokens, bonds=bonds, charges=charges, sanitise=sanitise)

    def _smol_from_tensors(self, coords, atom_dists, bond_dists, charge_dists):
        n_atoms = coords.size(0)

        charges = torch.tensor(self._mol_extract_charges(charge_dists))
        bond_indices = torch.ones((n_atoms, n_atoms)).nonzero()
        bond_types = bond_dists[bond_indices[:, 0], bond_indices[:, 1], :]

        mol = GeometricMol(coords, atom_dists, bond_indices, bond_types, charges)
        return mol

    def mol_stabilities(self, coords, atom_dists, mask, bond_dists, charge_dists):
        extracted = self._extract_mols(
            coords,
            atom_dists,
            mask,
            bond_dists=bond_dists,
            charge_dists=charge_dists
        )
        mol_atom_stabilities = [self.atom_stabilities(*items) for items in extracted]
        return mol_atom_stabilities

    def atom_stabilities(self, coords, atom_dists, bond_dists, charge_dists):
        n_atoms = coords.shape[0]

        atomics = self._mol_extract_atomics(atom_dists)
        bonds = self._mol_extract_bonds(bond_dists)
        charges = self._mol_extract_charges(charge_dists).tolist()

        # Recreate the adj to ensure it is symmetric
        bond_indices = torch.tensor(bonds[:, :2])
        bond_types = torch.tensor(bonds[:, 2])
        adj = smolF.adj_from_edges(bond_indices, bond_types, n_atoms, symmetric=True)

        adj[adj == 4] = 1.5
        valencies = adj.sum(dim=-1).long()

        stabilities = []
        for i in range(n_atoms):
            atom_type = atomics[i]
            charge = charges[i]
            valence = valencies[i].item()

            if atom_type not in Metrics.ALLOWED_VALENCIES:
                stabilities.append(False)
                continue

            allowed = Metrics.ALLOWED_VALENCIES[atom_type]
            atom_stable = Metrics._is_valid_valence(valence, allowed, charge)
            stabilities.append(atom_stable)

        return stabilities

    # Separate each molecule from the batch
    def _extract_mols(self, coords, atom_dists, mask, bond_dists=None, charge_dists=None):
        coords_list = []
        atom_dists_list = []
        bond_dists_list = []
        charge_dists_list = []

        n_atoms = mask.sum(dim=1)
        for idx in range(coords.size(0)):
            mol_atoms = n_atoms[idx]
            mol_coords = coords[idx, :mol_atoms, :].cpu()
            mol_token_dists = atom_dists[idx, :mol_atoms, :].cpu()

            coords_list.append(mol_coords)
            atom_dists_list.append(mol_token_dists)

            if bond_dists is not None:
                mol_bond_dists = bond_dists[idx, :mol_atoms, :mol_atoms, :].cpu()
                bond_dists_list.append(mol_bond_dists)
            else:
                bond_dists_list.append(None)

            if charge_dists is not None:
                mol_charge_dists = charge_dists[idx, :mol_atoms, :].cpu()
                charge_dists_list.append(mol_charge_dists)
            else:
                charge_dists_list.append(None)

        zipped = zip(coords_list, atom_dists_list, bond_dists_list, charge_dists_list)
        return zipped

    # Take index with highest probability and convert to token
    def _mol_extract_atomics(self, atom_dists):
        vocab_indices = torch.argmax(atom_dists, dim=1).tolist()
        tokens = self.vocab.tokens_from_indices(vocab_indices)
        return tokens

    # Convert to atomic number bond list format
    def _mol_extract_bonds(self, bond_dists):
        bond_types = torch.argmax(bond_dists, dim=-1)
        bonds = smolF.bonds_from_adj(bond_types)
        return bonds.long().numpy()

    # Convert index from model to actual atom charge
    def _mol_extract_charges(self, charge_dists):
        charge_types = torch.argmax(charge_dists, dim=-1).tolist()
        charges = [smolRD.IDX_CHARGE_MAP[idx] for idx in charge_types]
        return np.array(charges)


# *********************************************************************************************************************
# ******************************************** Lightning Flow Matching Models *****************************************
# *********************************************************************************************************************

class MolecularCFM(L.LightningModule):
    def __init__(
        self,
        gen: MolecularGenerator,
        vocab: Vocabulary,
        lr: float,
        integrator: Integrator,
        coord_scale: float = 1.0,
        type_strategy: str = "ce",
        bond_strategy: str = "ce",
        charge_strategy: str = "ce",
        type_loss_weight: float = 1.0,
        bond_loss_weight: float = 1.0,
        charge_loss_weight: float = 1.0,
        use_fm_coord_loss: bool = False,
        use_cat_time_based_weight: bool = False,
        pairwise_metrics: bool = True,
        use_ema: bool = True,
        compile_model: bool = True,
        self_condition: bool = False,
        distill: bool = False,
        lr_schedule: str = "constant",
        sampling_scheduler: str = "arcsin",
        sampling_scheduler_factor_rho: Optional[float] = 2.0,
        warm_up_steps: Optional[int] = None,
        total_steps: Optional[int] = None,
        train_smiles: Optional[list[str]] = None,
        val_smiles: Optional[list[str]] = None,
        type_mask_index: Optional[int] = None,
        bond_mask_index: Optional[int] = None,
        max_sigma: Optional[float] = 80,
        min_sigma: Optional[float] = 0.001,
        sigma_data: Optional[float] = 1.0,
        low_confidence_remask: Optional[bool] = False,
        # x_pred_mode: None (disabled), 'constant' (standard EDM preconditioning), 
        # or 'adaptive' (with adaptive identity component subtraction)
        # Merged from use_x_pred and x_pred_type parameters
        x_pred_mode: Optional[str] = None,
        **kwargs
    ):
        super().__init__()

        if type_strategy not in ["mse", "ce", "mask"]:
            raise ValueError(f"Unsupported type training strategy '{type_strategy}'. "
                             + "Supported are `mse`, `ce` or `mask`.")

        if bond_strategy not in ["ce", "mask"]:
            raise ValueError(f"Unsupported bond training strategy '{bond_strategy}'. Supported are `ce` or `mask`.")

        if lr_schedule not in ["constant", "one-cycle"]:
            raise ValueError(f"LR scheduler {lr_schedule} not supported. Supported are `constant` or `one-cycle`.")

        if lr_schedule == "one-cycle" and total_steps is None:
            raise ValueError("total_steps must be provided when using the one-cycle LR scheduler.")

        if distill and (type_strategy == "mask" or bond_strategy == "mask"):
            raise ValueError("Distilled training with masking strategy is not supported.")

        if lr_schedule == "one-cycle" and warm_up_steps is not None:
            print("Note: warm_up_steps is currently ignored if schedule is one-cycle")

        self.gen = gen
        self.vocab = vocab
        self.lr = lr
        self.coord_scale = coord_scale
        self.type_strategy = type_strategy
        self.bond_strategy = bond_strategy
        self.type_loss_weight = type_loss_weight
        self.bond_loss_weight = bond_loss_weight
        self.charge_loss_weight = charge_loss_weight
        self.pairwise_metrics = pairwise_metrics
        self.compile_model = compile_model
        self.self_condition = self_condition
        self.distill = distill
        self.lr_schedule = lr_schedule
        self.sampling_scheduler = sampling_scheduler
        self.warm_up_steps = warm_up_steps
        self.total_steps = total_steps
        self.type_mask_index = type_mask_index
        self.bond_mask_index = bond_mask_index
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        # sampling_scheduler_factor_rho: Merged from sampling_strategy_factor and rho
        # Controls the mixing between linear and arcsin time schedules
        self.rho = sampling_scheduler_factor_rho
        self.sigma_data = sigma_data
        self.use_cat_time_based_weight = use_cat_time_based_weight
        self.soft_label = False
        self.use_fm_coord_loss = use_fm_coord_loss
        # x_pred_mode: None (disabled), 'constant' (standard EDM), or 'adaptive' (with αt subtraction)
        self.x_pred_mode = x_pred_mode
        builder = MolBuilder(vocab)
        if use_ema:
            avg_fn = torch.optim.swa_utils.get_ema_multi_avg_fn(0.999)
            ema_gen = torch.optim.swa_utils.AveragedModel(gen, multi_avg_fn=avg_fn)

        if compile_model:
            self.gen = self._compile_model(gen)

        self.integrator = integrator
        self.builder = builder
        self.ema_gen = ema_gen if use_ema else None
        # rho already set above from sampling_scheduler_factor_rho
        # x_pred_mode already set above
        if self.integrator.sampler == 'llada':
            self.use_CM_coord_step = False
            self.use_llada_discrete_step = True
        elif self.integrator.sampler == 'llada-cm':
            self.use_CM_coord_step = True
            self.use_llada_discrete_step = True
        else:
            self.use_CM_coord_step = False
            self.use_llada_discrete_step = False
        # Anything else passed into kwargs will also be saved
        hparams = {
            "lr": lr,
            "coord_scale": coord_scale,
            "type_loss_weight": type_loss_weight,
            "bond_loss_weight": bond_loss_weight,
            "type_strategy": type_strategy,
            "bond_strategy": bond_strategy,
            "self_condition": self_condition,
            "distill": distill,
            "lr_schedule": lr_schedule,
            "sampling_strategy": sampling_scheduler,
            "sampling_scheduler_factor_rho": sampling_scheduler_factor_rho,
            "use_ema": use_ema,
            "compile_model": compile_model,
            "warm_up_steps": warm_up_steps,
            "sigma_data": sigma_data,
            "use_fm_coord_loss": use_fm_coord_loss,
            "x_pred_mode": x_pred_mode,
            **gen.hparams,
            **integrator.hparams,
            **kwargs
        }
        self.save_hyperparameters(hparams)

        stability_metrics = {
            "atom-stability": Metrics.AtomStability(),
            "molecule-stability": Metrics.MoleculeStability()
        }
        gen_metrics = {
            "validity": Metrics.Validity(),
            "fc-validity": Metrics.Validity(connected=True),
            "uniqueness": Metrics.Uniqueness(),
            "energy-validity": Metrics.EnergyValidity(),
            "opt-energy-validity": Metrics.EnergyValidity(optimise=True),
            "energy": Metrics.AverageEnergy(),
            "energy-per-atom": Metrics.AverageEnergy(per_atom=True),
            "strain": Metrics.AverageStrainEnergy(),
            "strain-per-atom": Metrics.AverageStrainEnergy(per_atom=True),
            "opt-rmsd": Metrics.AverageOptRmsd()
        }

        if train_smiles is not None:
            print("Creating RDKit mols from training SMILES...")
            train_mols = self.builder.mols_from_smiles(train_smiles, explicit_hs=True)
            train_mols = [mol for mol in train_mols if mol is not None]

            print("Initialising novelty metric...")
            gen_metrics["novelty"] = Metrics.Novelty(train_mols)
            print("Novelty metric complete.")
        if val_smiles is not None:
            print("Creating RDKit mols from validation SMILES...")
            val_mols = self.builder.mols_from_smiles(val_smiles, explicit_hs=True)
            val_mols = [mol for mol in val_mols if mol is not None]
            print('len(val_mols)', len(val_mols))
            gen_metrics['fcd-score'] = Metrics.FCD_Score(val_mols)

        self.stability_metrics = MetricCollection(stability_metrics, compute_groups=False)
        self.gen_metrics = MetricCollection(gen_metrics, compute_groups=False)
        self.time_mean = self.hparams["train-time-mean"]
        self.time_sigma = self.hparams["train-time-sigma"]
        self.low_confidence_remask = low_confidence_remask
        # if pairwise_metrics:
        #     pair_metrics = {
        #         "mol-accuracy": Metrics.MolecularAccuracy(),
        #         "pair-rmsd": Metrics.MolecularPairRMSD()
        #     }
        #     self.pair_metrics = MetricCollection(pair_metrics, compute_groups=False)

        self._init_params()

    def forward(self, batch, t, training=False, cond_batch=None):
        """Predict molecular coordinates and atom types

        Args:
            batch (dict[str, Tensor]): Batched pointcloud data(noisy data to be denoised)
            t (torch.Tensor): Interpolation times between 0 and 1, shape [batch_size]
            training (bool): Whether to run forward in training mode
            cond_batch (dict[str, Tensor]): Predictions from previous step, if we are using self conditioning

        Returns:
            (predicted coordinates, atom type logits (unnormalised probabilities))
            Both torch.Tensor, shapes [batch_size, num_atoms, 3] and [batch_size, num atoms, vocab_size]
        """
        coords = batch["coords"]
        atom_types = batch["atomics"]
        bonds = batch["bonds"]
        mask = batch["mask"]
        coef = self._prepare_coef(t)

        input_coords = coef["c_in"].view(-1, 1, 1)*coords
        # print('input_coords Var',torch.var(input_coords[0]))
        
        # Prepare invariant atom features
        times = coef["c_noise"].view(-1, 1, 1).expand(-1, coords.size(1), -1)
        features = torch.cat((times, atom_types), dim=2)
        # Whether to use the EMA version of the model or not
        if not training and self.ema_gen is not None:
            model = self.ema_gen
        else:
            model = self.gen
        if cond_batch is not None:
            net_out = model(
                input_coords,
                features,
                edge_feats=bonds,
                cond_coords=cond_batch["coords"],
                cond_atomics=cond_batch["atomics"],
                cond_bonds=cond_batch["bonds"],
                atom_mask=mask
            )

        else:
            net_out = model(input_coords, features, edge_feats=bonds, atom_mask=mask)
        output_coords, output_types, output_bonds, output_charges = net_out
        
        # Apply x-prediction mode if enabled
        # 'constant': Standard EDM preconditioning - subtract input coords directly
        # 'adaptive': Adaptive identity component subtraction using αt = σd*t/(σd² + t²)
        # This reduces residual correlation with input and aligns with network's inductive bias
        if self.x_pred_mode == 'constant':
            # Standard EDM: Dθ(xt; t) = cskip*xt + cout*(Fθ(cin*xt; cnoise) - cin*xt)
            output_coords = output_coords - input_coords
        elif self.x_pred_mode == 'adaptive':
            # Adaptive: Dθ(xt; t) = cskip*xt + cout*(Fθ(cin*xt; cnoise) - αt*cin*xt)
            # where αt = σd*t/(σd² + t²) is the optimal LMMSE coefficient
            x_pred_coef = self.sigma_data*t/(self.sigma_data**2 + t**2)
            output_coords = output_coords - x_pred_coef.view(-1, 1, 1)*input_coords
        elif self.x_pred_mode is not None:
            raise ValueError(f"Unsupported x_pred_mode: {self.x_pred_mode}. Must be None, 'constant', or 'adaptive'")
        
        coords = output_coords*coef["c_out"].view(-1, 1, 1) + coords*coef["c_skip"].view(-1, 1, 1)

        return coords, output_types, output_bonds, output_charges
    def training_step(self, batch, b_idx):
        return self._training_step(batch, b_idx)
    def _get_initial_cond_batch(self, interpolated):
        """Initialize conditioning batch with zeros"""
        return {
            "coords": torch.zeros_like(interpolated["coords"]),
            "atomics": torch.zeros_like(interpolated["atomics"]),
            "bonds": torch.zeros_like(interpolated["bonds"])
        }
    def _generate_self_condition(self, interpolated, times, cond_batch):
        """Generate self conditioning batch"""
        with torch.no_grad():
            cond_coords, cond_types, cond_bonds, _ = self(
                interpolated, times, training=True,
                cond_batch=cond_batch
            )
            
            return {
                "coords": cond_coords,
                "atomics": F.softmax(cond_types, dim=-1),
                "bonds": F.softmax(cond_bonds, dim=-1)
            }
    def _training_step(self, batch, b_idx):
        current_epoch = self.trainer.current_epoch
        prior, data, interpolated, times = batch
        # data is the true data, interpolated is the noisy data, difference is computed between the predicted and the true data

        # If training with self conditioning, half the time generate a conditional batch by setting cond to zeros
        cond_batch = self._get_initial_cond_batch(interpolated) if self.self_condition else None
        # Apply self conditioning
        if self.self_condition and torch.rand(1).item() > 0.5:
            cond_batch = self._generate_self_condition(interpolated, times, cond_batch)

        
        coords, types, bonds, charges = self(
            interpolated,
            times,
            training=True,
            cond_batch=cond_batch
        )
        predicted = {
            "coords": coords,
            "atomics": types,
            "bonds": bonds,
            "charges": charges
        }
        coef = self._prepare_coef(times)
        if self.use_cat_time_based_weight:
            mask_rates = self.integrator._compute_mask_rates(times)
            cat_time_based_weight = 1/mask_rates
            cat_time_based_weight = np.clip(cat_time_based_weight, 0, 10)
        else:
            cat_time_based_weight = 1
            
        losses = self._loss(data, interpolated, predicted, 
                            coord_time_based_weight=1/coef["c_out"]**2, 
                            cat_time_based_weight=cat_time_based_weight, 
                            epoch_percentage=current_epoch/self.trainer.max_epochs,
                            prior = prior, times = times)
        loss = sum(list(losses.values()))

        for name, loss_val in losses.items():
            self.log(f"train-{name}", loss_val, on_step=True, logger=True)

        self.log("train-loss", loss, prog_bar=True, on_step=True, logger=True)

        return loss





    def on_train_batch_end(self, outputs, batch, b_idx):
        if self.ema_gen is not None:
            self.ema_gen.update_parameters(self.gen)

    def validation_step(self, batch, b_idx):
        prior, data, interpolated, times = batch
        gen_batch = self._generate(prior, self.integrator.steps, self.sampling_scheduler)
        stabilities = self._generate_stabilities(gen_batch)
        gen_mols = self._generate_mols(gen_batch)

        self.stability_metrics.update(stabilities)
        self.gen_metrics.update(gen_mols)

        # Also measure the model's ability to recreate the original molecule when a bit of prior noise has been added
        # if self.pairwise_metrics:
        #     gen_interp_steps = max(1, int((1 - times[0].item()) * self.integrator.steps))
        #     gen_interp_batch = self._generate(interpolated, gen_interp_steps)
        #     gen_interp_mols = self._generate_mols(gen_interp_batch)
        #     data_mols = self._generate_mols(data)
        #     self.pair_metrics.update(gen_interp_mols, data_mols)

    def on_validation_epoch_end(self):
        stability_metrics_results = self.stability_metrics.compute()
        gen_metrics_results = self.gen_metrics.compute()
        pair_metrics_results = self.pair_metrics.compute() if (self.pairwise_metrics and hasattr(self, 'pair_metrics')) else {}

        metrics = {
            **stability_metrics_results,
            **gen_metrics_results,
            **pair_metrics_results
        }

        for metric, value in metrics.items():
            progbar = True if metric == "validity" else False
            self.log(f"val-{metric}", value, on_epoch=True, logger=True, prog_bar=progbar)

        self.stability_metrics.reset()
        self.gen_metrics.reset()

        if self.pairwise_metrics:
            self.pair_metrics.reset()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        self.on_validation_epoch_end()

    def predict_step(self, batch, batch_idx):
        prior, _, _, _ = batch
        gen_batch = self._generate(prior, self.integrator.steps, self.sampling_scheduler)
        gen_mols = self._generate_mols(gen_batch)
        return gen_mols

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.gen.parameters(),
            lr=self.lr,
            amsgrad=True,
            foreach=True,
            weight_decay=0.0
        )

        if self.lr_schedule == "constant":
            warm_up_steps = 0 if self.warm_up_steps is None else self.warm_up_steps
            scheduler = LinearLR(opt, start_factor=1e-2, total_iters=warm_up_steps)

        # TODO could use warm_up_steps to shift peak of one cycle
        elif self.lr_schedule == "one-cycle":
            scheduler = OneCycleLR(opt, max_lr=self.lr, total_steps=self.total_steps, pct_start=0.3)

        else:
            raise ValueError("Only `constant` or `one-cycle` LR schedules are supported.")

        config = {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"}
        }
        return config
    def _prepare_coef(self, t):
        """Prepare coefficients for EDM-style noise schedule.
        
        Args:
            t: Time step tensor
            
        Returns:
            Dictionary containing noise schedule coefficients
        """
        sigma_data = self.sigma_data
        sigma = t
        c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
        c_out = sigma * sigma_data / (sigma**2 + sigma_data**2).sqrt()
        c_in = 1 / (sigma**2 + sigma_data**2).sqrt()
        c_noise = sigma.log() / 4
        
        return {
            "c_skip": c_skip,
            "c_out": c_out,
            "c_in": c_in,
            "c_noise": c_noise
        }
        
    def _prepare_cat_coef(self, t, n_dims=None):
        if self.integrator.mask_rate_strategy == 'log_uniform':
            # Note: mask_times_factor removed - log_uniform strategy now always uses unscaled time
            mask_rate = (t.log() - (self.time_mean - self.time_sigma*4))/(self.time_sigma*8)
        elif self.integrator.mask_rate_strategy == 'edm' or self.integrator.mask_rate_strategy is None:
            # DFM mask rate formula: mask_rate = t/(t+1)
            # Derived from DFM mask prior: mask_rate = w_noise/(w_data + w_noise) = t/(1+t)
            # where t = w_noise/w_data is the weight ratio
            # Note: mask_times_factor removed - time is used directly without scaling
            mask_rate = t / (t + 1)
        mask_rate = mask_rate.clamp(min=self.integrator.eps, max=1-self.integrator.eps)
        if self.soft_label:
            C = self.delta_out
            S = n_dims
            # c_out = (self.delta_star_base + ((S - (S - 1)*mask_rate**self.gamma) / mask_rate**self.gamma).log() - c_skip * ((S - (S - 1)*mask_rate) / mask_rate).log())/C
        

        c_skip = (1 - mask_rate)/2
        c_out = (1 + mask_rate)/2
        # print('mask_rate is ', mask_rate[0].item(),' c_skip is ', c_skip[0].item(),' c_out is ', c_out[0].item())

        return {
            "c_skip": c_skip,
            "c_out": c_out
        }
    def _compile_model(self, model):
        return torch.compile(model, dynamic=False, fullgraph=True, mode="reduce-overhead")
        
    def _loss(self, data, interpolated, predicted, coord_time_based_weight=1, cat_time_based_weight=1, epoch_percentage=None, prior=None, times=None):
        pred_coords = predicted["coords"]
        coords = data["coords"]
        mask = data["mask"].unsqueeze(2)
        if self.use_fm_coord_loss:
            coord_loss = self._fm_coord_loss(data, interpolated, predicted, prior, times)
        else:
            coord_loss = F.mse_loss(pred_coords, coords, reduction="none")
        coord_loss = (coord_loss * mask).mean(dim=(1, 2))

        type_loss = self._type_loss(data, interpolated, predicted)
        bond_loss = self._bond_loss(data, interpolated, predicted)
        charge_loss = self._charge_loss(data, predicted)
        
        coord_loss = (coord_loss * coord_time_based_weight).mean()
        type_loss = type_loss.mean() * self.type_loss_weight * cat_time_based_weight
        bond_loss = bond_loss.mean() * self.bond_loss_weight * cat_time_based_weight
        charge_loss = charge_loss.mean() * self.charge_loss_weight * cat_time_based_weight
        
        losses = {
            "coord-loss": coord_loss,
            "type-loss": type_loss,
            "bond-loss": bond_loss,
            "charge-loss": charge_loss
        }
        
        return losses

    def _fm_coord_loss(self, data, interpolated, predicted, prior, times):
        x_t_rescale = interpolated["coords"]/(times[0].item()**2+self.sigma_data**2)**0.5
        pred_x_rescale = predicted["coords"]/(times[0].item()**2+self.sigma_data**2)**0.5
        pred_velocity = (pred_x_rescale - x_t_rescale)
        target_velocity = (data["coords"]*times[0].item() - prior["coords"])/(times[0].item()**2+self.sigma_data**2)**1.5
        coord_loss = F.mse_loss(pred_velocity, target_velocity, reduction="none")#*(times[0].item()**2+self.sigma_data**2)**4/1000 
        return coord_loss
    def _type_loss(self, data, interpolated, predicted, eps=1e-3):
        pred_logits = predicted["atomics"]
        atomics_dist = data["atomics"]
        mask = data["mask"].unsqueeze(2)
        batch_size, num_atoms, _ = pred_logits.size()

        if self.type_strategy == "mse":
            type_loss = F.mse_loss(pred_logits, atomics_dist, reduction="none")
        # elif self.soft_label:
        #     soft_targets = data['atomics_soft']
        #     print(torch.sum(torch.abs(soft_targets - atomics_dist)))
        #     log_probs = F.log_softmax(pred_logits.flatten(0, 1), dim=-1)      # [B*N, C]
        #     type_loss = -(soft_targets.flatten(0, 1) * log_probs).sum(dim=-1)     # [B*N]
        #     type_loss = type_loss.unflatten(0, (batch_size, num_atoms)).unsqueeze(2)     # [B, N, 1]
            
        else:
            if not self.soft_label:
                data['atomics_eps'] = 0
            atomics = torch.argmax(atomics_dist, dim=-1).flatten(0, 1)
            type_loss = F.cross_entropy(pred_logits.flatten(0, 1), atomics, reduction="none", label_smoothing=data['atomics_eps'])
            type_loss = type_loss.unflatten(0, (batch_size, num_atoms)).unsqueeze(2)

        # if True:
        #     atomics = torch.argmax(atomics_dist, dim=-1).flatten(0, 1)
        #     temp_type_loss = F.cross_entropy(pred_logits.flatten(0, 1), atomics, reduction="none", label_smoothing=0.1)
        #     temp_type_loss = temp_type_loss.unflatten(0, (batch_size, num_atoms)).unsqueeze(2)
        #     print('type_loss is ', type_loss.mean().item(), ' temp_type_loss is ', temp_type_loss.mean().item())

        n_atoms = mask.sum(dim=(1, 2)) + eps

        # If we are training with masking, only compute the loss on masked types
        if self.type_strategy == "mask":
            masked_types = torch.argmax(interpolated["atomics"], dim=-1) == self.type_mask_index
            n_atoms = masked_types.sum(dim=-1) + eps
            type_loss = type_loss * masked_types.float().unsqueeze(-1)

        type_loss = (type_loss * mask).sum(dim=(1, 2)) / n_atoms
        return type_loss

    def _bond_loss(self, data, interpolated, predicted, eps=1e-3):
        pred_logits = predicted["bonds"]
        mask = data["mask"]
        bonds = torch.argmax(data["bonds"], dim=-1)
        batch_size, num_atoms, _, _ = pred_logits.size()
        # if self.soft_label:
        #     soft_targets = data['bonds_soft']
        #     log_probs = F.log_softmax(pred_logits.flatten(0, 2), dim=-1)      # [B*N, C]
        #     bond_loss = -(soft_targets.flatten(0, 2) * log_probs).sum(dim=-1)               # [B*N]
        #     bond_loss = bond_loss.unflatten(0, (batch_size, num_atoms, num_atoms))
        # else:
        if not self.soft_label:
            data['bonds_eps'] = 0
        bond_loss = F.cross_entropy(pred_logits.flatten(0, 2), bonds.flatten(0, 2), reduction="none", label_smoothing=data['bonds_eps'])
        bond_loss = bond_loss.unflatten(0, (batch_size, num_atoms, num_atoms))

        adj_matrix = smolF.adj_from_node_mask(mask, self_connect=True)
        n_bonds = adj_matrix.sum(dim=(1, 2)) + eps

        # Only compute loss on masked bonds if we are training with masking strategy
        if self.bond_strategy == "mask":
            masked_bonds = torch.argmax(interpolated["bonds"], dim=-1) == self.bond_mask_index
            n_bonds = masked_bonds.sum(dim=(1, 2)) + eps
            bond_loss = bond_loss * masked_bonds.float()

        bond_loss = (bond_loss * adj_matrix).sum(dim=(1, 2)) / n_bonds
        return bond_loss

    def _charge_loss(self, data, predicted, eps=1e-3):
        pred_logits = predicted["charges"]
        charges = data["charges"]
        mask = data["mask"]
        batch_size, num_atoms, _ = pred_logits.size()

        charges = torch.argmax(charges, dim=-1).flatten(0, 1)
        charge_loss = F.cross_entropy(pred_logits.flatten(0, 1), charges, reduction="none")
        charge_loss = charge_loss.unflatten(0, (batch_size, num_atoms))

        n_atoms = mask.sum(dim=1) + eps
        charge_loss = (charge_loss * mask).sum(dim=1) / n_atoms
        return charge_loss
    def _create_time_schedule(self, steps, strategy):
        """Create time schedule based on strategy"""
        if strategy == "log":
            time_points = np.geomspace(self.min_sigma, self.max_sigma, steps + 1).tolist()
            time_points.reverse()
        elif strategy == "arcsin":
            transformed = 2 * np.arcsin(np.linspace(0, 1, steps + 1)**0.5) / np.pi
            mixed = (1 - self.rho) * np.linspace(0, 1, steps + 1) + self.rho * transformed
            time_points = mixed * (np.log(self.max_sigma) - np.log(self.min_sigma)) + np.log(self.min_sigma)
            time_points = np.exp(time_points).tolist()
            time_points.reverse()
        else:
            raise ValueError(f"Unknown ODE integration strategy '{strategy}'")
        time_points[-1] = 0
        return time_points
    def _generate(self, prior, steps, strategy="arcsin"):
        
        time_start = time.time()
        if self.distill:
            return self._distill_generate(prior)
        time_points = self._create_time_schedule(steps, strategy)
        times = torch.ones(prior["coords"].size(0), device=self.device) * self.max_sigma
        step_sizes = [t0 - t1 for t0, t1 in zip(time_points[:-1], time_points[1:])]
        curr = {
            k: (v.clone() * self.max_sigma) if k == 'coords' else v.clone()
            for k, v in prior.items()
        }
        time_end = time.time()
        print(f"Time taken for time schedule: {time_end - time_start} seconds")
        time_start = time.time()
        cond_batch = self._get_initial_cond_batch(prior) if self.self_condition else None
        with torch.no_grad():
            
            for t_idx, step_size in enumerate(step_sizes):
                cond = cond_batch if self.self_condition else None
                gamma = self.integrator.coord_noise_std
                gamma = 1
                # gamma = 1 + gamma if 0.03 <= (times[0].item()) <= 3 else 1
                times_hat = times*gamma

                if gamma > 1:
                    
                    step_size += (times_hat[0].item() - times[0].item())
                    curr = self._interpolate_with_noise(curr, times, times_hat)

                coords, type_logits, bond_logits, charge_logits = self(curr, times_hat, training=False, cond_batch=cond)
                times = times_hat - step_size
                times = torch.clamp(times, min=1e-6)
                type_probs = F.softmax(type_logits, dim=-1)
                bond_probs = F.softmax(bond_logits, dim=-1)
                charge_probs = F.softmax(charge_logits, dim=-1)

                cond_batch = {
                    "coords": coords,
                    "atomics": type_probs,
                    "bonds": bond_probs
                }
                predicted = {
                    "coords": coords,
                    "atomics": type_probs,
                    "bonds": bond_probs,
                    "charges": charge_probs,
                    "mask": curr["mask"]
                }

                if self.type_strategy == "mask":
                    self.llada_discrete_step = True
                    self.low_confidence_remask = False
                else:
                    self.llada_discrete_step = False
                    self.low_confidence_remask = False
                # New logic: if llada_discrete_step is True, use discrete features directly
                curr = self._euler_step(curr, predicted, prior, times_hat, step_size)
                if self.llada_discrete_step:
                    
                    # Use predicted discrete features to replace curr, but keep coords unchanged
                    # print('curr coords before', curr["coords"][0])
                    curr_coords = curr["coords"].clone()  # Save original coords
                    curr.update({
                        "atomics": F.one_hot(torch.argmax(predicted['atomics'], dim=-1), num_classes=predicted['atomics'].size(-1)).float(),
                        "bonds": F.one_hot(torch.argmax(predicted['bonds'], dim=-1), num_classes=predicted['bonds'].size(-1)).float(),
                        "charges": F.one_hot(torch.argmax(predicted['charges'], dim=-1), num_classes=predicted['charges'].size(-1)).float()
                    })
                    # Remask discrete features using _interpolate_with_noise
                    # Interpolate from time=0 to current time step
                    zero_times = torch.zeros_like(times)
                    curr_times = times.clone()
                    curr = self._interpolate_with_noise(curr, zero_times, curr_times, curr_probs = predicted)
                    
                    # Restore coords to keep them unchanged during interpolation
                    curr["coords"] = curr_coords                    
                    # print('curr coords after', curr["coords"][0])

        predicted["coords"] = predicted["coords"] * self.coord_scale
        return predicted
    def _interpolate_with_noise(self, curr, times, times_hat, curr_probs = None):
        """Interpolate current state with noise"""
        # Sample from molecules
        batch_size = curr["coords"].size(0)
        num_atoms = curr["coords"].size(1)
        from_mols = [self.integrator.prior_sampler.sample_molecule(num_atoms) for _ in range(batch_size)]
        # Reuse mask rate computation logic from training
        _, _, delta_mask_rates = self.integrator._compute_mask_rate_deltas(times, times_hat)
        delta_times = float(torch.sqrt(times_hat[0]**2 - times[0]**2).item())
        
        print('t and t_hat', times[0].item(), times_hat[0].item(), delta_mask_rates)
        print('before interpolate var:', torch.var(curr["coords"][0]), times[0].item())
        
        # Prepare from_mols
        mol_batch = GeometricMolBatch(from_mols, device=curr["coords"].device)

        # from_mols = self.integrator.data_module._batch_to_dict(mol_batch)
        
        from_mols = {"coords": mol_batch.coords, "atomics": mol_batch.atomics, "bonds": mol_batch.adjacency, "charges": mol_batch.charges, "mask": mol_batch.mask.long()}
        if from_mols["charges"] is not None:
            n_charges = len(smolRD.CHARGE_IDX_MAP.keys())
            from_mols["charges"] = smolF.one_hot_encode_tensor(from_mols["charges"], n_charges)

        # from_mols = {k: v.clone() for k, v in from_mols.items()}
        from_mols = {k: v for k, v in from_mols.items()}

        curr = self._interpolate_mol(from_mols, curr, delta_times, delta_mask_rates, curr_probs = curr_probs)
        
        print('after interpolate var:', torch.var(curr["coords"][0]), times_hat[0].item())
        
        return curr
    def _euler_step(self, curr, predicted, prior, times_hat, step_size):
        """Perform Euler integration step"""
        curr = self.integrator.step(curr, predicted, prior, times_hat, step_size)
        return curr
        

    def _generate_mols(self, generated, sanitise=True):
        coords = generated["coords"]
        atom_dists = generated["atomics"]
        bond_dists = generated["bonds"]
        charge_dists = generated["charges"]
        masks = generated["mask"]

        mols = self.builder.mols_from_tensors(
            coords,
            atom_dists,
            masks,
            bond_dists=bond_dists,
            charge_dists=charge_dists,
            sanitise=sanitise
        )
        return mols

    def _generate_stabilities(self, generated):
        coords = generated["coords"]
        atom_dists = generated["atomics"]
        bond_dists = generated["bonds"]
        charge_dists = generated["charges"]
        masks = generated["mask"]
        stabilities = self.builder.mol_stabilities(coords, atom_dists, masks, bond_dists, charge_dists)
        return stabilities

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
    def _interpolate_mol(self, from_mols: dict, to_mols: dict, delta_t: float, delta_mask_rate: float, curr_probs = None) -> dict:
        """Interpolates mols which have already been sampled according to OT map, if required"""
        #to_mols is the sample, from_mol is the noise

        if from_mols["coords"].size(0) != to_mols["coords"].size(0):
            raise RuntimeError("Both molecules to be interpolated must have the same number of atoms.")
        # Interpolate coords and add gaussian noise
        coords = to_mols["coords"] + from_mols["coords"]*delta_t
# Handle atomics
        if self.llada_discrete_step and self.low_confidence_remask and curr_probs is not None:
            atomics_confidence = torch.max(curr_probs["atomics"], dim=-1)[0]
            atom_mask = atomics_confidence < torch.quantile(atomics_confidence.flatten(), delta_mask_rate)
        else:
            atom_mask = torch.rand(from_mols["atomics"].size(0), from_mols["atomics"].size(1)) < delta_mask_rate
    
        # atom_mask = torch.rand(from_mols["atomics"].size(0), from_mols["atomics"].size(1)) < delta_mask_rate
        
        
        to_atomics = torch.argmax(to_mols["atomics"], dim=-1)
        from_atomics = torch.argmax(from_mols["atomics"], dim=-1)
        to_atomics[atom_mask] = from_atomics[atom_mask]
        atomics = smolF.one_hot_encode_tensor(to_atomics, to_mols["atomics"].size(-1))

        # Interpolate bonds
            # Handle bonds
        if self.llada_discrete_step and self.low_confidence_remask and curr_probs is not None:
            bonds_confidence = torch.max(curr_probs["bonds"], dim=-1)[0]
            bond_mask = bonds_confidence < torch.quantile(bonds_confidence.flatten(), delta_mask_rate)
            to_adj = torch.argmax(to_mols["bonds"], dim=-1)
        else:
            to_adj = torch.argmax(to_mols["bonds"], dim=-1)
            bond_mask = torch.rand_like(to_adj.float()) < delta_mask_rate

        from_adj = torch.argmax(from_mols["bonds"], dim=-1)
        to_adj[bond_mask] = from_adj[bond_mask]
        interp_adj = smolF.one_hot_encode_tensor(to_adj, to_mols["bonds"].size(-1))

        # bond_indices = torch.ones((from_mols["bonds"].size(0), from_mols["bonds"].size(1), from_mols["bonds"].size(1))).nonzero()
        # bond_types = interp_adj[bond_indices[:, 0], bond_indices[:, 1]]

        # interp_mol = GeometricMol(coords, atomics, bond_indices=bond_indices, bond_types=bond_types)

        interp_mol = {"coords": coords, "atomics": atomics, "bonds": interp_adj, "mask": to_mols["mask"]}
        return interp_mol