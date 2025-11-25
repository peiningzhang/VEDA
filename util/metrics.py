import os
from concurrent.futures import ProcessPoolExecutor

import torch
from rdkit import Chem
from torchmetrics import Metric

from util import rdkit as smolRD

ALLOWED_VALENCIES = {
    "H": {0: 1, 1: 0, -1: 0},
    "C": {0: [3, 4], 1: 3, -1: 3},
    "N": {0: [2, 3], 1: [2, 3, 4], -1: 2},  # In QM9, N+ seems to be present in the form NH+ and NH2+
    "O": {0: 2, 1: 3, -1: 1},
    "F": {0: 1, -1: 0},
    "B": 3,
    "Al": 3,
    "Si": 4,
    "P": {0: [3, 5], 1: 4},
    "S": {0: [2, 6], 1: [2, 3], 2: 4, 3: 5, -1: 3},
    "Cl": 1,
    "As": 3,
    "Br": {0: 1, 1: 2},
    "I": 1,
    "Hg": [1, 2],
    "Bi": [3, 5],
    "Se": [2, 4, 6],
}


def calc_atom_stabilities(mol):
    stabilities = []

    for atom in mol.GetAtoms():
        atom_type = atom.GetSymbol()
        valence = atom.GetExplicitValence()
        charge = atom.GetFormalCharge()

        if atom_type not in ALLOWED_VALENCIES:
            stabilities.append(False)
            continue

        allowed = ALLOWED_VALENCIES[atom_type]
        atom_stable = _is_valid_valence(valence, allowed, charge)
        stabilities.append(atom_stable)

    return stabilities


def _is_valid_valence(valence, allowed, charge):
    if isinstance(allowed, int):
        valid = allowed == valence

    elif isinstance(allowed, list):
        valid = valence in allowed

    elif isinstance(allowed, dict):
        allowed = allowed.get(charge)
        if allowed is None:
            return False

        valid = _is_valid_valence(valence, allowed, charge)

    return valid


def _is_valid_float(num):
    return num not in [None, float("inf"), float("-inf"), float("nan")]


class GenerativeMetric(Metric):
    # TODO add metric attributes - see torchmetrics doc

    def __init__(self, **kwargs):
        # Pass extra kwargs (defined in Metric class) to parent
        super().__init__(**kwargs)

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        raise NotImplementedError()

    def compute(self) -> torch.Tensor:
        raise NotImplementedError()


class PairMetric(Metric):
    def __init__(self, **kwargs):
        # Pass extra kwargs (defined in Metric class) to parent
        super().__init__(**kwargs)

    def update(self, predicted: list[Chem.rdchem.Mol], actual: list[Chem.rdchem.Mol]) -> None:
        raise NotImplementedError()

    def compute(self) -> torch.Tensor:
        raise NotImplementedError()


class AtomStability(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state("atom_stable", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, stabilities: list[list[bool]]) -> None:
        all_atom_stables = [atom_stable for atom_stbs in stabilities for atom_stable in atom_stbs]
        self.atom_stable += sum(all_atom_stables)
        self.total += len(all_atom_stables)

    def compute(self) -> torch.Tensor:
        return self.atom_stable.float() / self.total


class MoleculeStability(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state("mol_stable", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, stabilities: list[list[bool]]) -> None:
        mol_stables = [sum(atom_stbs) == len(atom_stbs) for atom_stbs in stabilities]
        self.mol_stable += sum(mol_stables)
        self.total += len(mol_stables)

    def compute(self) -> torch.Tensor:
        return self.mol_stable.float() / self.total


class Validity(GenerativeMetric):
    def __init__(self, connected=False, **kwargs):
        super().__init__(**kwargs)
        self.connected = connected

        self.add_state("valid", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        is_valid = [smolRD.mol_is_valid(mol, connected=self.connected) for mol in mols if mol is not None]
        self.valid += sum(is_valid)
        self.total += len(mols)

    def compute(self) -> torch.Tensor:
        return self.valid.float() / self.total


# TODO I don't think this will work with DDP
class Uniqueness(GenerativeMetric):
    """Note: only tracks uniqueness of molecules which can be converted into SMILES"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.valid_smiles = []

    def reset(self):
        self.valid_smiles = []

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        smiles = [smolRD.smiles_from_mol(mol, canonical=True) for mol in mols if mol is not None]
        valid_smiles = [smi for smi in smiles if smi is not None]
        self.valid_smiles.extend(valid_smiles)

    def compute(self) -> torch.Tensor:
        num_unique = len(set(self.valid_smiles))
        uniqueness = torch.tensor(num_unique) / len(self.valid_smiles)
        return uniqueness


class Novelty(GenerativeMetric):
    def __init__(self, existing_mols: list[Chem.rdchem.Mol], **kwargs):
        super().__init__(**kwargs)

        n_workers = min(8, len(os.sched_getaffinity(0)))
        executor = ProcessPoolExecutor(max_workers=n_workers)

        futures = [executor.submit(smolRD.smiles_from_mol, mol, canonical=True) for mol in existing_mols]
        smiles = [future.result() for future in futures]
        smiles = [smi for smi in smiles if smi is not None]

        executor.shutdown()

        self.smiles = set(smiles)

        self.add_state("novel", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        smiles = [smolRD.smiles_from_mol(mol, canonical=True) for mol in mols if mol is not None]
        valid_smiles = [smi for smi in smiles if smi is not None]
        novel = [smi not in self.smiles for smi in valid_smiles]

        self.novel += sum(novel)
        self.total += len(novel)

    def compute(self) -> torch.Tensor:
        return self.novel.float() / self.total


class EnergyValidity(GenerativeMetric):
    def __init__(self, optimise=False, **kwargs):
        super().__init__(**kwargs)

        self.optimise = optimise

        self.add_state("n_valid", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        num_mols = len(mols)

        if self.optimise:
            mols = [smolRD.optimise_mol(mol) for mol in mols if mol is not None]

        energies = [smolRD.calc_energy(mol) for mol in mols if mol is not None]
        valid_energies = [energy for energy in energies if _is_valid_float(energy)]

        self.n_valid += len(valid_energies)
        self.total += num_mols

    def compute(self) -> torch.Tensor:
        return self.n_valid.float() / self.total


class AverageEnergy(GenerativeMetric):
    """Average energy for molecules for which energy can be calculated

    Note that the energy cannot be calculated for some molecules (specifically invalid ones) and the pose optimisation
    is not guaranteed to succeed. Molecules for which the energy cannot be calculated do not count towards the metric.

    This metric doesn't require that input molecules have been sanitised by RDKit, however, it is usually a good idea
    to do this anyway to ensure that all of the required molecular and atom properties are calculated and stored.
    """

    def __init__(self, optimise=False, per_atom=False, **kwargs):
        super().__init__(**kwargs)

        self.optimise = optimise
        self.per_atom = per_atom

        self.add_state("energy", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_valid_energies", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        if self.optimise:
            mols = [smolRD.optimise_mol(mol) for mol in mols if mol is not None]

        energies = [smolRD.calc_energy(mol, per_atom=self.per_atom) for mol in mols if mol is not None]
        valid_energies = [energy for energy in energies if _is_valid_float(energy)]

        self.energy += sum(valid_energies)
        self.n_valid_energies += len(valid_energies)

    def compute(self) -> torch.Tensor:
        return self.energy / self.n_valid_energies


class AverageStrainEnergy(GenerativeMetric):
    """
    The strain energy is the energy difference between a molecule's pose and its optimised pose. Estimated using RDKit.
    Only calculated when all of the following are true:
    1. The molecule is valid and an energy can be calculated
    2. The pose optimisation succeeds
    3. The energy can be calculated for the optimised pose

    Note that molecules which do not meet these criteria will not count towards the metric and can therefore give
    unexpected results. Use the EnergyValidity metric with the optimise flag set to True to track the proportion of
    molecules for which this metric can be calculated.

    This metric doesn't require that input molecules have been sanitised by RDKit, however, it is usually a good idea
    to do this anyway to ensure that all of the required molecular and atom properties are calculated and stored.
    """

    def __init__(self, per_atom=False, **kwargs):
        super().__init__(**kwargs)

        self.per_atom = per_atom

        self.add_state("total_energy_diff", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_valid", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        opt_mols = [(idx, smolRD.optimise_mol(mol)) for idx, mol in list(enumerate(mols)) if mol is not None]
        energies = [(idx, smolRD.calc_energy(mol, per_atom=self.per_atom)) for idx, mol in opt_mols if mol is not None]
        valids = [(idx, energy) for idx, energy in energies if energy is not None]

        if len(valids) == 0:
            return

        valid_indices, valid_energies = tuple(zip(*valids))
        original_energies = [smolRD.calc_energy(mols[idx], per_atom=self.per_atom) for idx in valid_indices]
        energy_diffs = [orig - opt for orig, opt in zip(original_energies, valid_energies)]

        self.total_energy_diff += sum(energy_diffs)
        self.n_valid += len(energy_diffs)

    def compute(self) -> torch.Tensor:
        return self.total_energy_diff / self.n_valid


class AverageOptRmsd(GenerativeMetric):
    """
    Average RMSD between a molecule and its optimised pose. Only calculated when all of the following are true:
    1. The molecule is valid
    2. The pose optimisation succeeds

    Note that molecules which do not meet these criteria will not count towards the metric and can therefore give
    unexpected results.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state("total_rmsd", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_valid", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        valids = [(idx, smolRD.optimise_mol(mol)) for idx, mol in list(enumerate(mols)) if mol is not None]
        valids = [(idx, mol) for idx, mol in valids if mol is not None]

        if len(valids) == 0:
            return

        valid_indices, opt_mols = tuple(zip(*valids))
        original_mols = [mols[idx] for idx in valid_indices]
        rmsds = [smolRD.conf_distance(mol1, mol2) for mol1, mol2 in zip(original_mols, opt_mols)]

        self.total_rmsd += sum(rmsds)
        self.n_valid += len(rmsds)

    def compute(self) -> torch.Tensor:
        return self.total_rmsd / self.n_valid


class MolecularAccuracy(PairMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state("n_correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predicted: list[Chem.rdchem.Mol], actual: list[Chem.rdchem.Mol]) -> None:
        predicted_smiles = [smolRD.smiles_from_mol(pred, canonical=True) for pred in predicted]
        actual_smiles = [smolRD.smiles_from_mol(act, canonical=True) for act in actual]
        matches = [pred == act for pred, act in zip(predicted_smiles, actual_smiles) if act is not None]

        self.n_correct += sum(matches)
        self.total += len(matches)

    def compute(self) -> torch.Tensor:
        return self.n_correct.float() / self.total


class MolecularPairRMSD(PairMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state("total_rmsd", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_valid", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predicted: list[Chem.rdchem.Mol], actual: list[Chem.rdchem.Mol]) -> None:
        valid_pairs = [(pred, act) for pred, act in zip(predicted, actual) if pred is not None and act is not None]
        rmsds = [smolRD.conf_distance(pred, act) for pred, act in valid_pairs]
        rmsds = [rmsd for rmsd in rmsds if rmsd is not None]

        self.total_rmsd += sum(rmsds)
        self.n_valid += len(rmsds)

    def compute(self) -> torch.tensor:
        return self.total_rmsd / self.n_valid


from fcd_torch import FCD
class FCD_Score(GenerativeMetric):
    def __init__(
        self,
        existing_mols,
        device: str = 'cuda:0',
        n_jobs: int = 8,
        **kwargs
    ):
        """
        reference_sdf_path: path to your reference .sdf file.
        device: torch device for FCD ('cpu' or 'cuda:0', etc.).
        n_jobs: number of workers for FCD internal calculations.
        """
        super().__init__(**kwargs)
        n_workers = min(8, len(os.sched_getaffinity(0)))
        executor = ProcessPoolExecutor(max_workers=n_workers)

        futures = [executor.submit(smolRD.smiles_from_mol, mol, canonical=True) for mol in existing_mols]
        smiles = [future.result() for future in futures]
        smiles = [smi for smi in smiles if smi is not None]
        # Initialize FCD calculator
        self.fcd = FCD(device=device, n_jobs=n_jobs)
        # Precompute the reference distributions
        self.pgen_ref = self.fcd.precalc(smiles)
        # Accumulate generated smiles here
        self.generated_smiles: list[str] = []


    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        """
        Collects SMILES from newly generated molecules.
        """
        smiles = [smolRD.smiles_from_mol(mol, canonical=True) for mol in mols if mol is not None]
        self.generated_smiles = [smi for smi in smiles if smi is not None]

    def compute(self) -> torch.Tensor:
        """
        Returns the FCD score between all accumulated generated SMILES
        and the reference distribution.
        """
        print('generated smiles', self.generated_smiles)
        try:
            score = self.fcd(self.generated_smiles, pgen=self.pgen_ref)
        except Exception as e:
            if not self.generated_smiles:
                print("No valid generated molecules to score.")
            print(f"Error calculating FCD score: {e}")
            return torch.tensor(1000000)
        # wrap in a tensor for compatibility
        return score
    
import numpy as np
from rdkit.Chem import rdMolTransforms
#TODO: add a metric to calculate the bond length, angle, and torsion deviation from the reference molecule


    
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from scipy import stats


class BondLengthTypeSpecificDistribution(GenerativeMetric):
    """
    按键类型分别计算分布差异（稍微复杂一点但更准确）
    """
    
    def __init__(self, reference_mols: list[Chem.rdchem.Mol], **kwargs):
        super().__init__(**kwargs)
        
        # 按键类型分组的参考键长
        self.reference_by_type = self._group_bonds_by_type(reference_mols)
        self.generated_by_type = {bond_type: [] for bond_type in self.reference_by_type.keys()}
    
    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        """按键类型收集生成分子的键长"""
        batch_by_type = self._group_bonds_by_type(mols)
        
        for bond_type, lengths in batch_by_type.items():
            if bond_type in self.generated_by_type:
                self.generated_by_type[bond_type].extend(lengths)
    
    def _group_bonds_by_type(self, mols):
        """按键类型分组提取键长"""
        bonds_by_type = {}
        
        for mol in mols:
            if mol is None or mol.GetNumConformers() == 0:
                continue
                
            conf = mol.GetConformer()
            for bond in mol.GetBonds():
                # 创建键类型标识符
                atom1 = mol.GetAtomWithIdx(bond.GetBeginAtomIdx())
                atom2 = mol.GetAtomWithIdx(bond.GetEndAtomIdx())
                
                # 标准化键类型 (例如: C-C, C-N, C=C)
                symbols = sorted([atom1.GetSymbol(), atom2.GetSymbol()])
                bond_key = f"{symbols[0]}-{symbols[1]}_{bond.GetBondType()}"
                
                length = rdMolTransforms.GetBondLength(
                    conf, bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                )
                
                if bond_key not in bonds_by_type:
                    bonds_by_type[bond_key] = []
                bonds_by_type[bond_key].append(length)
        
        return bonds_by_type
    
    def compute(self) -> torch.Tensor:
        """计算所有键类型的平均分布差异"""
        total_distance = 0.0
        valid_types = 0
        
        for bond_type in self.reference_by_type.keys():
            if (bond_type in self.generated_by_type and 
                len(self.generated_by_type[bond_type]) > 0):
                
                distance = stats.wasserstein_distance(
                    self.generated_by_type[bond_type],
                    self.reference_by_type[bond_type]
                )
                total_distance += distance
                valid_types += 1
        
        if valid_types == 0:
            return torch.tensor(float('inf'))
        
        return torch.tensor(total_distance / valid_types)




class BondAngleTypeSpecificDistribution(GenerativeMetric):
    """
    按键角类型分别计算分布差异
    """
    
    def __init__(self, reference_mols: list[Chem.rdchem.Mol], **kwargs):
        super().__init__(**kwargs)
        
        # 按键角类型分组的参考键角
        self.reference_by_type = self._group_angles_by_type(reference_mols)
        self.generated_by_type = {angle_type: [] for angle_type in self.reference_by_type.keys()}
    
    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        """按键角类型收集生成分子的键角"""
        batch_by_type = self._group_angles_by_type(mols)
        
        for angle_type, angles in batch_by_type.items():
            if angle_type in self.generated_by_type:
                self.generated_by_type[angle_type].extend(angles)
    
    def _group_angles_by_type(self, mols):
        """按键角类型分组提取键角"""
        angles_by_type = {}
        
        for mol in mols:
            if mol is None or mol.GetNumConformers() == 0:
                continue
                
            conf = mol.GetConformer()
            
            for atom in mol.GetAtoms():
                neighbors = [neighbor.GetIdx() for neighbor in atom.GetNeighbors()]
                
                # 生成所有可能的键角，以当前原子为中心
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        # 计算键角
                        angle_rad = rdMolTransforms.GetAngleRad(conf,
                                                              neighbors[i],
                                                              atom.GetIdx(),
                                                              neighbors[j])
                        angle_deg = np.degrees(angle_rad)
                        
                        # 创建键角类型标识符
                        atom1_symbol = mol.GetAtomWithIdx(neighbors[i]).GetSymbol()
                        atom2_symbol = atom.GetSymbol()  # 中心原子
                        atom3_symbol = mol.GetAtomWithIdx(neighbors[j]).GetSymbol()
                        
                        # 标准化键角类型 (例如: C-C-C, N-C-C)
                        symbols = sorted([atom1_symbol, atom3_symbol])
                        angle_key = f"{symbols[0]}-{atom2_symbol}-{symbols[1]}"
                        
                        if angle_key not in angles_by_type:
                            angles_by_type[angle_key] = []
                        angles_by_type[angle_key].append(angle_deg)
        
        return angles_by_type
    
    def compute(self) -> torch.Tensor:
        """计算所有键角类型的平均分布差异"""
        total_distance = 0.0
        valid_types = 0
        
        for angle_type in self.reference_by_type.keys():
            if (angle_type in self.generated_by_type and 
                len(self.generated_by_type[angle_type]) > 0):
                
                distance = stats.wasserstein_distance(
                    self.generated_by_type[angle_type],
                    self.reference_by_type[angle_type]
                )
                total_distance += distance
                valid_types += 1
        
        if valid_types == 0:
            return torch.tensor(float('inf'))
        
        return torch.tensor(total_distance / valid_types)


class TorsionAngleTypeSpecificDistribution(GenerativeMetric):
    """
    按扭转角类型分别计算分布差异
    """
    
    def __init__(self, reference_mols: list[Chem.rdchem.Mol], **kwargs):
        super().__init__(**kwargs)
        
        # 按扭转角类型分组的参考扭转角
        self.reference_by_type = self._group_torsions_by_type(reference_mols)
        self.generated_by_type = {torsion_type: [] for torsion_type in self.reference_by_type.keys()}
    
    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        """按扭转角类型收集生成分子的扭转角"""
        batch_by_type = self._group_torsions_by_type(mols)
        
        for torsion_type, torsions in batch_by_type.items():
            if torsion_type in self.generated_by_type:
                self.generated_by_type[torsion_type].extend(torsions)
    
    def _group_torsions_by_type(self, mols):
        """按扭转角类型分组提取扭转角"""
        torsions_by_type = {}
        
        for mol in mols:
            if mol is None or mol.GetNumConformers() == 0:
                continue
                
            conf = mol.GetConformer()
            
            # 找到所有4个连接原子的组合
            for bond1 in mol.GetBonds():
                atom2_idx = bond1.GetBeginAtomIdx()
                atom3_idx = bond1.GetEndAtomIdx()
                
                atom2 = mol.GetAtomWithIdx(atom2_idx)
                atom3 = mol.GetAtomWithIdx(atom3_idx)
                
                # 获取atom2的邻居（除了atom3）
                atom2_neighbors = [n.GetIdx() for n in atom2.GetNeighbors() if n.GetIdx() != atom3_idx]
                
                # 获取atom3的邻居（除了atom2）
                atom3_neighbors = [n.GetIdx() for n in atom3.GetNeighbors() if n.GetIdx() != atom2_idx]
                
                # 生成所有可能的扭转角
                for atom1_idx in atom2_neighbors:
                    for atom4_idx in atom3_neighbors:
                        # 计算扭转角
                        try:
                            torsion_rad = rdMolTransforms.GetDihedralRad(conf,
                                                                       atom1_idx,
                                                                       atom2_idx,
                                                                       atom3_idx,
                                                                       atom4_idx)
                            torsion_deg = np.degrees(torsion_rad)
                            
                            # 创建扭转角类型标识符
                            atom1_symbol = mol.GetAtomWithIdx(atom1_idx).GetSymbol()
                            atom2_symbol = mol.GetAtomWithIdx(atom2_idx).GetSymbol()
                            atom3_symbol = mol.GetAtomWithIdx(atom3_idx).GetSymbol()
                            atom4_symbol = mol.GetAtomWithIdx(atom4_idx).GetSymbol()
                            
                            # 标准化扭转角类型 (例如: C-C-C-C, C-C-N-C)
                            # 使用字典序最小的排列来确保一致性
                            pattern1 = f"{atom1_symbol}-{atom2_symbol}-{atom3_symbol}-{atom4_symbol}"
                            pattern2 = f"{atom4_symbol}-{atom3_symbol}-{atom2_symbol}-{atom1_symbol}"
                            torsion_key = min(pattern1, pattern2)
                            
                            if torsion_key not in torsions_by_type:
                                torsions_by_type[torsion_key] = []
                            torsions_by_type[torsion_key].append(torsion_deg)
                            
                        except:
                            # 跳过无效的扭转角计算
                            continue
        
        return torsions_by_type
    
    def _circular_distance(self, angles1, angles2):
        """计算两个角度分布的循环距离"""
        # 将角度转换为复数表示以处理周期性
        complex1 = [np.exp(1j * np.radians(angle)) for angle in angles1]
        complex2 = [np.exp(1j * np.radians(angle)) for angle in angles2]
        
        # 计算角度的平均方向
        mean1 = np.angle(np.mean(complex1), deg=True)
        mean2 = np.angle(np.mean(complex2), deg=True)
        
        # 计算循环差异
        diff = abs(mean1 - mean2)
        circular_diff = min(diff, 360 - diff)
        
        return circular_diff
    
    def compute(self) -> torch.Tensor:
        """计算所有扭转角类型的平均分布差异"""
        total_distance = 0.0
        valid_types = 0
        
        for torsion_type in self.reference_by_type.keys():
            if (torsion_type in self.generated_by_type and 
                len(self.generated_by_type[torsion_type]) > 0):
                
                # 对于扭转角，使用循环距离更合适
                distance = self._circular_distance(
                    self.generated_by_type[torsion_type],
                    self.reference_by_type[torsion_type]
                )
                total_distance += distance
                valid_types += 1
        
        if valid_types == 0:
            return torch.tensor(float('inf'))
        
        return torch.tensor(total_distance / valid_types)
    

class MolecularShapeDistribution(GenerativeMetric):
    """
    分子形状指标 - 基于主成分分析的形状描述符
    """
    
    def __init__(self, reference_mols: list[Chem.rdchem.Mol], **kwargs):
        super().__init__(**kwargs)
        
        self.reference_shapes = self._extract_shape_descriptors(reference_mols)
        self.generated_shapes = []
    
    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        """收集生成分子的形状描述符"""
        batch_shapes = self._extract_shape_descriptors(mols)
        self.generated_shapes.extend(batch_shapes)
    
    def _extract_shape_descriptors(self, mols):
        """提取分子形状描述符"""
        shape_descriptors = []
        
        for mol in mols:
            if mol is None or mol.GetNumConformers() == 0:
                continue
                
            conf = mol.GetConformer()
            positions = []
            
            # 获取所有原子坐标
            for i in range(mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(i)
                positions.append([pos.x, pos.y, pos.z])
            
            if len(positions) < 3:
                continue
                
            positions = np.array(positions)
            
            # 计算重心
            centroid = np.mean(positions, axis=0)
            centered_positions = positions - centroid
            
            # 主成分分析
            try:
                cov_matrix = np.cov(centered_positions.T)
                eigenvalues = np.linalg.eigvals(cov_matrix)
                eigenvalues = np.sort(eigenvalues)[::-1]  # 降序排列
                
                # 形状描述符：特征值比值
                if len(eigenvalues) >= 3 and eigenvalues[0] > 1e-6:
                    asphericity = eigenvalues[0] - 0.5 * (eigenvalues[1] + eigenvalues[2])
                    acylindricity = eigenvalues[1] - eigenvalues[2]
                    relative_shape_anisotropy = (asphericity**2 + 0.75 * acylindricity**2) / (eigenvalues.sum()**2)
                    
                    shape_descriptors.append({
                        'asphericity': asphericity,
                        'acylindricity': acylindricity, 
                        'relative_shape_anisotropy': relative_shape_anisotropy
                    })
            except:
                continue
                
        return shape_descriptors
    
    def compute(self) -> torch.Tensor:
        """计算形状分布差异"""
        if len(self.generated_shapes) == 0 or len(self.reference_shapes) == 0:
            return torch.tensor(float('inf'))
        
        # 分别计算各个形状描述符的距离
        total_distance = 0.0
        
        for descriptor in ['asphericity', 'acylindricity', 'relative_shape_anisotropy']:
            ref_values = [shape[descriptor] for shape in self.reference_shapes]
            gen_values = [shape[descriptor] for shape in self.generated_shapes]
            
            distance = stats.wasserstein_distance(gen_values, ref_values)
            total_distance += distance
        
        return torch.tensor(total_distance / 3)


class HydrogenBondingGeometry(GenerativeMetric):
    """
    氢键几何分布指标
    """
    
    def __init__(self, reference_mols: list[Chem.rdchem.Mol], **kwargs):
        super().__init__(**kwargs)
        
        self.reference_hbond_angles = self._extract_hbond_geometry(reference_mols)
        self.generated_hbond_angles = []
    
    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        """收集生成分子的氢键几何"""
        batch_hbond_angles = self._extract_hbond_geometry(mols)
        self.generated_hbond_angles.extend(batch_hbond_angles)
    
    def _extract_hbond_geometry(self, mols):
        """提取可能的氢键几何信息"""
        hbond_angles = []
        
        for mol in mols:
            if mol is None or mol.GetNumConformers() == 0:
                continue
                
            conf = mol.GetConformer()
            
            # 找到所有氢原子
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == 'H':
                    h_idx = atom.GetIdx()
                    
                    # 找到氢原子连接的重原子（供体）
                    neighbors = list(atom.GetNeighbors())
                    if len(neighbors) != 1:
                        continue
                        
                    donor_atom = neighbors[0]
                    donor_idx = donor_atom.GetIdx()
                    
                    # 只考虑N-H, O-H类型的氢键供体
                    if donor_atom.GetSymbol() not in ['N', 'O']:
                        continue
                    
                    # 寻找可能的氢键受体（N, O原子）
                    for acceptor_atom in mol.GetAtoms():
                        if (acceptor_atom.GetSymbol() in ['N', 'O'] and 
                            acceptor_atom.GetIdx() != donor_idx):
                            
                            acceptor_idx = acceptor_atom.GetIdx()
                            
                            # 计算距离
                            h_pos = conf.GetAtomPosition(h_idx)
                            acceptor_pos = conf.GetAtomPosition(acceptor_idx)
                            distance = h_pos.Distance(acceptor_pos)
                            
                            # 如果距离合理（可能的氢键）
                            if 1.5 < distance < 3.0:
                                # 计算 D-H...A 角度
                                try:
                                    angle_rad = rdMolTransforms.GetAngleRad(
                                        conf, donor_idx, h_idx, acceptor_idx
                                    )
                                    angle_deg = np.degrees(angle_rad)
                                    hbond_angles.append(angle_deg)
                                except:
                                    continue
        
        return hbond_angles
    
    def compute(self) -> torch.Tensor:
        """计算氢键几何分布差异"""
        if len(self.generated_hbond_angles) == 0 and len(self.reference_hbond_angles) == 0:
            return torch.tensor(0.0)
        elif len(self.generated_hbond_angles) == 0 or len(self.reference_hbond_angles) == 0:
            return torch.tensor(float('inf'))
            
        distance = stats.wasserstein_distance(
            self.generated_hbond_angles,
            self.reference_hbond_angles
        )
        
        return torch.tensor(distance)


class ImprovedTorsionAngleDistribution(GenerativeMetric):
    """
    改进的扭转角分布指标 - 解决你提到的torsion效果差的问题
    """
    
    def __init__(self, reference_mols: list[Chem.rdchem.Mol], **kwargs):
        super().__init__(**kwargs)
        
        self.reference_by_type = self._group_torsions_by_type(reference_mols)
        self.generated_by_type = {torsion_type: [] for torsion_type in self.reference_by_type.keys()}
    
    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        """按扭转角类型收集生成分子的扭转角"""
        batch_by_type = self._group_torsions_by_type(mols)
        
        for torsion_type, torsions in batch_by_type.items():
            if torsion_type in self.generated_by_type:
                self.generated_by_type[torsion_type].extend(torsions)
    
    def _group_torsions_by_type(self, mols):
        """改进的扭转角分组方法"""
        torsions_by_type = {}
        
        for mol in mols:
            if mol is None or mol.GetNumConformers() == 0:
                continue
                
            conf = mol.GetConformer()
            
            # 只考虑非端基的键（避免自由旋转的噪声）
            for bond in mol.GetBonds():
                atom2_idx = bond.GetBeginAtomIdx()
                atom3_idx = bond.GetEndAtomIdx()
                
                atom2 = mol.GetAtomWithIdx(atom2_idx)
                atom3 = mol.GetAtomWithIdx(atom3_idx)
                
                # 跳过端基键（度数为1的原子）
                if atom2.GetDegree() == 1 or atom3.GetDegree() == 1:
                    continue
                
                # 获取邻居原子
                atom2_neighbors = [n.GetIdx() for n in atom2.GetNeighbors() if n.GetIdx() != atom3_idx]
                atom3_neighbors = [n.GetIdx() for n in atom3.GetNeighbors() if n.GetIdx() != atom2_idx]
                
                if len(atom2_neighbors) == 0 or len(atom3_neighbors) == 0:
                    continue
                
                # 选择最重的原子作为扭转角端点（减少构象自由度影响）
                atom1_idx = max(atom2_neighbors, key=lambda x: mol.GetAtomWithIdx(x).GetAtomicNum())
                atom4_idx = max(atom3_neighbors, key=lambda x: mol.GetAtomWithIdx(x).GetAtomicNum())
                
                try:
                    torsion_rad = rdMolTransforms.GetDihedralRad(conf,
                                                               atom1_idx,
                                                               atom2_idx,
                                                               atom3_idx,
                                                               atom4_idx)
                    torsion_deg = np.degrees(torsion_rad)
                    
                    # 创建更详细的扭转角类型标识符（包含键级）
                    atom1_symbol = mol.GetAtomWithIdx(atom1_idx).GetSymbol()
                    atom2_symbol = mol.GetAtomWithIdx(atom2_idx).GetSymbol()
                    atom3_symbol = mol.GetAtomWithIdx(atom3_idx).GetSymbol()
                    atom4_symbol = mol.GetAtomWithIdx(atom4_idx).GetSymbol()
                    
                    # 获取中心键的类型
                    center_bond_type = bond.GetBondType()
                    
                    # 标准化扭转角类型
                    pattern1 = f"{atom1_symbol}-{atom2_symbol}={center_bond_type}={atom3_symbol}-{atom4_symbol}"
                    pattern2 = f"{atom4_symbol}-{atom3_symbol}={center_bond_type}={atom2_symbol}-{atom1_symbol}"
                    torsion_key = min(pattern1, pattern2)
                    
                    if torsion_key not in torsions_by_type:
                        torsions_by_type[torsion_key] = []
                    torsions_by_type[torsion_key].append(torsion_deg)
                    
                except:
                    continue
        
        return torsions_by_type
    
    def _improved_circular_distance(self, angles1, angles2):
        """改进的循环距离计算"""
        if len(angles1) == 0 or len(angles2) == 0:
            return float('inf')
        
        # 将角度标准化到 [-180, 180] 区间
        angles1_norm = [(a + 180) % 360 - 180 for a in angles1]
        angles2_norm = [(a + 180) % 360 - 180 for a in angles2]
        
        # 使用Von Mises分布参数来描述角度分布
        def circular_mean_and_concentration(angles):
            complex_angles = [np.exp(1j * np.radians(a)) for a in angles]
            mean_complex = np.mean(complex_angles)
            mean_angle = np.degrees(np.angle(mean_complex))
            concentration = abs(mean_complex)  # R值，表示集中程度
            return mean_angle, concentration
        
        mean1, conc1 = circular_mean_and_concentration(angles1_norm)
        mean2, conc2 = circular_mean_and_concentration(angles2_norm)
        
        # 计算角度差异
        angle_diff = abs(mean1 - mean2)
        angle_diff = min(angle_diff, 360 - angle_diff)
        
        # 计算集中度差异
        conc_diff = abs(conc1 - conc2)
        
        # 综合距离
        return angle_diff + 100 * conc_diff  # 给集中度差异更大权重
    
    def compute(self) -> torch.Tensor:
        """计算改进的扭转角分布差异"""
        total_distance = 0.0
        valid_types = 0
        
        for torsion_type in self.reference_by_type.keys():
            if (torsion_type in self.generated_by_type and 
                len(self.generated_by_type[torsion_type]) > 10):  # 需要足够的样本
                
                distance = self._improved_circular_distance(
                    self.generated_by_type[torsion_type],
                    self.reference_by_type[torsion_type]
                )
                
                if distance != float('inf'):
                    total_distance += distance
                    valid_types += 1
        
        if valid_types == 0:
            return torch.tensor(float('inf'))
        
        return torch.tensor(total_distance / valid_types)
    
    
    
    
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms
from xtb.ase.calculator import XTB

class XTBEnergy(GenerativeMetric):
    def __init__(self, per_atom=False, method="GFN2-xTB", **kwargs):
        super().__init__(**kwargs)
        self.per_atom = per_atom
        self.method = method

        self.add_state("energy_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def _calculate_xtb_energy(self, mol, confId=0):
        if mol is None:
            return None
        
        mol_copy = Chem.Mol(mol)
        
        try:
            if mol_copy.GetNumConformers() == 0:
                AllChem.EmbedMolecule(mol_copy)
                AllChem.MMFFOptimizeMolecule(mol_copy)
            
            conf = mol_copy.GetConformer(confId)
            positions = []
            symbols = []
            
            for atom in mol_copy.GetAtoms():
                pos = conf.GetAtomPosition(atom.GetIdx())
                positions.append([pos.x, pos.y, pos.z])
                symbols.append(atom.GetSymbol())
            
            positions = np.array(positions)
            atoms = Atoms(positions=positions, symbols=symbols)
            atoms.calc = XTB(method=self.method)
            
            energy = atoms.get_potential_energy()
            energy = energy / mol_copy.GetNumAtoms() if self.per_atom else energy
            
            return energy
            
        except Exception:
            return None

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        energies = [self._calculate_xtb_energy(mol) for mol in mols if mol is not None]
        valid_energies = [e for e in energies if e is not None]
        
        if valid_energies:
            self.energy_sum += sum(valid_energies)
            self.total += len(valid_energies)

    def compute(self) -> torch.Tensor:
        if self.total == 0:
            return torch.tensor(0.0)
        return self.energy_sum.float() / self.total
    
    
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import QED, Crippen, Descriptors, Lipinski
from rdkit.Chem import rdMolDescriptors
from rdkit.Contrib.SA_Score import sascorer

class QED_Score(GenerativeMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("qed_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        valid_mols = [mol for mol in mols if mol is not None]
        if not valid_mols:
            return
            
        qed_scores = []
        for mol in valid_mols:
            try:
                qed_score = QED.qed(mol)
                qed_scores.append(qed_score)
            except Exception:
                continue
        
        if qed_scores:
            self.qed_sum += sum(qed_scores)
            self.total += len(qed_scores)

    def compute(self) -> torch.Tensor:
        if self.total == 0:
            return torch.tensor(0.0)
        return self.qed_sum.float() / self.total


class SA_Score(GenerativeMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("sa_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        valid_mols = [mol for mol in mols if mol is not None]
        if not valid_mols:
            return
            
        sa_scores = []
        for mol in valid_mols:
            try:
                sa = self.calculate_sa_score(mol)
                sa_score = round((10 - sa) / 9, 2)
                sa_scores.append(sa_score)
            except Exception:
                continue
        if sa_scores:
            self.sa_sum += sum(sa_scores)
            self.total += len(sa_scores)


    def calculate_sa_score(self, mol):
        """SA score"""
        # 使用RDKit的SA scorer
        sa_score = sascorer.calculateScore(mol)
        return sa_score

    
    def compute(self) -> torch.Tensor:
        if self.total == 0:
            return torch.tensor(0.0)
        return self.sa_sum.float() / self.total


class LogP_Score(GenerativeMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("logp_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        valid_mols = [mol for mol in mols if mol is not None]
        if not valid_mols:
            return
            
        logp_scores = []
        for mol in valid_mols:
            try:
                logp = Crippen.MolLogP(mol)
                logp_scores.append(logp)
            except Exception:
                continue
        
        if logp_scores:
            self.logp_sum += sum(logp_scores)
            self.total += len(logp_scores)

    def compute(self) -> torch.Tensor:
        if self.total == 0:
            return torch.tensor(0.0)
        return self.logp_sum.float() / self.total


class Lipinski_Score(GenerativeMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("lipinski_sum", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        valid_mols = [mol for mol in mols if mol is not None]
        if not valid_mols:
            return
            
        lipinski_scores = []
        for mol in valid_mols:
            try:
                rule_1 = Descriptors.ExactMolWt(mol) < 500
                rule_2 = Lipinski.NumHDonors(mol) <= 5
                rule_3 = Lipinski.NumHAcceptors(mol) <= 10
                logp = Crippen.MolLogP(mol)
                rule_4 = (logp >= -2) & (logp <= 5)
                rule_5 = rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10
                lipinski_score = sum([int(a) for a in [rule_1, rule_2, rule_3, rule_4, rule_5]])
                lipinski_scores.append(lipinski_score)
            except Exception:
                continue
        
        if lipinski_scores:
            self.lipinski_sum += sum(lipinski_scores)
            self.total += len(lipinski_scores)

    def compute(self) -> torch.Tensor:
        if self.total == 0:
            return torch.tensor(0.0)
        return self.lipinski_sum.float() / self.total